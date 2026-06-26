import os
import time
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm

def load_data(base_dir):
    data_path = os.path.join(base_dir, '..', 'data', 'mock_nmr_data_10k.csv')
    gt_path = os.path.join(base_dir, '..', 'data', 'mock_ground_truth_10k.csv')
    ref_path = os.path.join(base_dir, '..', 'data', 'hmdb_reference', 'hmdb_reference_library.csv')
    
    print("Loading datasets...")
    df_data = pd.read_csv(data_path)
    df_gt = pd.read_csv(gt_path)
    df_ref = pd.read_csv(ref_path)
    
    return df_data, df_gt, df_ref

def build_reference_spectra_matrix(df_ref, ppm_values):
    """Build a (N_metabolites x 10001) reference matrix using vectorized Lorentzian."""
    print("Building reference spectra matrix (vectorized)...")
    metabolites = df_ref['Name'].unique()
    ref_matrix = []
    
    for meta in tqdm(metabolites, desc="Compiling ref library"):
        meta_peaks = df_ref[df_ref['Name'] == meta]
        locs = meta_peaks['loc'].values[:, np.newaxis]
        heights = meta_peaks['height'].values[:, np.newaxis]
        widths = (meta_peaks['width'].values / 450.0)[:, np.newaxis]
        
        x = ppm_values[np.newaxis, :]
        specs = heights / (1 + ((x - locs) / (widths / 2))**2)
        spec = np.sum(specs, axis=0)
        
        area = np.trapezoid(spec, ppm_values)
        if area > 0:
            spec = spec / area
        ref_matrix.append(spec)
        
    return np.array(ref_matrix, dtype=np.float32), metabolites


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    df_data, df_gt, df_ref = load_data(base_dir)
    ppm_values = df_data['ppm'].values

    # Build reference matrix (N_metabolites, 10001)
    ref_matrix, metabolites = build_reference_spectra_matrix(df_ref, ppm_values)
    n_metabolites = len(metabolites)

    # Prepare sample matrix (N_samples, 10001)
    sample_cols = [c for c in df_data.columns if c != 'ppm']
    n_samples = len(sample_cols)
    print(f"Extracting sample matrix for {n_samples} samples...")
    # Shape: (10001, 10000)
    sample_matrix = df_data[sample_cols].values.T.astype(np.float32)  # (10000, 10001)
    
    # ==================================================================
    # EXP-A: NMF-based approach
    # Strategy: 
    #   1. Run NMF on the entire sample matrix with n_components = K
    #      to find K latent "pure" components (W: samples x K, H: K x 10001)
    #   2. Match each latent component H[k] against the reference library
    #      using cosine similarity to find the closest metabolite
    #   3. For each sample, check which components have non-negligible weight (W[s,k] > threshold)
    #      → those metabolites are predicted present
    # ==================================================================
    K = 50  # Number of latent components. Typical mix has ~15 metabolites; K=50 gives headroom
    WEIGHT_THRESHOLD = 0.01  # Minimum W value to consider a component "present" in a sample
    SIM_THRESHOLD = 0.3      # Minimum cosine similarity to accept a component-metabolite match
    
    print(f"\nRunning NMF with n_components={K} on {n_samples} samples...")
    start_time = time.time()
    
    nmf_model = NMF(
        n_components=K,
        init='nndsvda',   # Better initialization for sparse NMR data
        solver='mu',      # Multiplicative update, better for spectral data
        max_iter=200,
        random_state=42,
        l1_ratio=0.5,
        alpha_W=0.01,
        alpha_H=0.01
    )
    
    # W: (10000, K) — how much each component contributes to each sample
    # H: (K, 10001) — the shape of each latent component spectrum
    W = nmf_model.fit_transform(sample_matrix)
    H = nmf_model.components_  # (K, 10001)
    
    nmf_time = time.time() - start_time
    print(f"NMF completed in {nmf_time:.1f}s. Matching components to metabolites...")
    
    # Match each NMF component to a metabolite using cosine similarity
    # H: (K, 10001), ref_matrix: (N_metabolites, 10001)
    # Result: (K, N_metabolites)
    comp_to_meta_sim = cosine_similarity(H, ref_matrix)  # (K, N_metabolites)
    
    # For each component, find the best-matching metabolite
    # and discard if max similarity is too low (likely noise/artifact)
    comp_to_meta_idx = np.argmax(comp_to_meta_sim, axis=1)  # (K,)
    comp_max_sim = np.max(comp_to_meta_sim, axis=1)         # (K,)
    
    valid_components = comp_max_sim >= SIM_THRESHOLD
    print(f"Valid components (sim>={SIM_THRESHOLD}): {np.sum(valid_components)}/{K}")
    
    # Build prediction matrix
    # For each sample s, predict metabolite m as present if:
    #   ∃ valid component k mapping to m, AND W[s,k] > WEIGHT_THRESHOLD
    eval_start = time.time()
    
    # Normalize W so each sample sums to 1 (relative contribution)
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-9)
    
    y_pred = np.zeros((n_samples, n_metabolites), dtype=int)
    
    for k in range(K):
        if not valid_components[k]:
            continue
        meta_idx = comp_to_meta_idx[k]
        # Samples where this component has enough weight
        present_samples = W_norm[:, k] > WEIGHT_THRESHOLD
        y_pred[present_samples, meta_idx] = 1
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("Evaluating Results...")
    
    # Ground Truth binary matrix
    metabolite_list = list(metabolites)
    meta_to_idx = {m: i for i, m in enumerate(metabolite_list)}
    
    y_true = np.zeros((n_samples, n_metabolites), dtype=int)
    for s_idx, (_, gt_row) in enumerate(df_gt.iterrows()):
        for meta, idx in meta_to_idx.items():
            if meta in gt_row and gt_row[meta] > 0:
                y_true[s_idx, idx] = 1
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    exact_match_ratio = np.sum(np.all(y_true == y_pred, axis=1)) / n_samples
    
    results = {
        'Experiment': 'EXP-A: NMF Decomposition',
        'NMF_Components': K,
        'Valid_Components': int(np.sum(valid_components)),
        'Weight_Threshold': WEIGHT_THRESHOLD,
        'Sim_Threshold': SIM_THRESHOLD,
        'Time_Seconds': round(elapsed, 2),
        'Time_Per_Sample_Ms': round((elapsed / n_samples) * 1000, 2),
        'Precision_Macro': round(precision, 4),
        'Recall_Macro': round(recall, 4),
        'F1_Macro': round(f1, 4),
        'Exact_Match_Ratio': round(exact_match_ratio, 4)
    }
    
    print("\n--- RESULTS ---")
    for k_name, v in results.items():
        print(f"{k_name}: {v}")
        
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {results_dir}")

if __name__ == '__main__':
    main()
