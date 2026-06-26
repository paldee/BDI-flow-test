"""
EXP-D: Multi-Window FAISS Retrieval
Strategy: 
  Instead of querying the FULL mixture spectrum against pure references (which fails 
  because the mixture is a superposition of many spectra), we divide each spectrum
  into overlapping LOCAL WINDOWS and query each window independently against the
  reference library. A metabolite is predicted 'Present' if its reference spectrum
  appears as a nearest neighbor in enough windows.

  This exploits the LOCAL structure of NMR peaks:
  - Each metabolite has peaks in specific ppm regions
  - A local window containing that region will be "close" to the reference spectrum
    in that window, even if the global mixture looks very different.

Steps:
  1. Build FAISS index from reference spectra LOCAL WINDOWS (N_metabolites x N_windows)
  2. For each sample, extract same LOCAL WINDOWS
  3. Query FAISS per window → collect votes per metabolite
  4. Predict 'Present' if votes >= VOTE_THRESHOLD
"""
import os
import time
import numpy as np
import pandas as pd
import faiss
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import TruncatedSVD
import json
from tqdm import tqdm


def lorentzian_vectorized(ppm_values, df_ref_meta):
    locs = df_ref_meta['loc'].values[:, np.newaxis]
    heights = df_ref_meta['height'].values[:, np.newaxis]
    widths = (df_ref_meta['width'].values / 450.0)[:, np.newaxis]
    x = ppm_values[np.newaxis, :]
    specs = heights / (1 + ((x - locs) / (widths / 2))**2)
    spec = np.sum(specs, axis=0)
    area = np.trapezoid(spec, ppm_values)
    if area > 0:
        spec /= area
    return spec.astype(np.float32)


def load_data(base_dir):
    data_path = os.path.join(base_dir, '..', 'data', 'mock_nmr_data_10k.csv')
    gt_path = os.path.join(base_dir, '..', 'data', 'mock_ground_truth_10k.csv')
    ref_path = os.path.join(base_dir, '..', 'data', 'hmdb_reference', 'hmdb_reference_library.csv')
    print("Loading datasets...")
    df_data = pd.read_csv(data_path)
    df_gt = pd.read_csv(gt_path)
    df_ref = pd.read_csv(ref_path)
    return df_data, df_gt, df_ref


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    df_data, df_gt, df_ref = load_data(base_dir)
    ppm_values = df_data['ppm'].values
    n_points = len(ppm_values)  # 10001
    metabolites = df_ref['Name'].unique()
    n_metabolites = len(metabolites)

    # ---- Build reference matrix ----
    print("Building reference spectra matrix...")
    ref_matrix = np.zeros((n_metabolites, n_points), dtype=np.float32)
    for i, meta in enumerate(tqdm(metabolites, desc="Compiling ref library")):
        meta_peaks = df_ref[df_ref['Name'] == meta]
        ref_matrix[i] = lorentzian_vectorized(ppm_values, meta_peaks)

    # ---- Define windows ----
    # 20 overlapping windows, each covering 1 ppm (1000 points), step = 0.5 ppm (500 points)
    WINDOW_SIZE = 1000   # 1 ppm
    WINDOW_STEP = 500    # 0.5 ppm overlap
    windows = []
    start = 0
    while start + WINDOW_SIZE <= n_points:
        windows.append((start, start + WINDOW_SIZE))
        start += WINDOW_STEP
    n_windows = len(windows)
    print(f"Using {n_windows} overlapping windows (1 ppm each, 0.5 ppm step).")

    # ---- Dimensionality reduction per window ----
    # Reduce each window from 1000 → 64 dims using TruncatedSVD for FAISS efficiency
    DIM = 64
    VOTE_THRESHOLD = 2       # How many windows must vote for a metabolite
    TOP_K_PER_WINDOW = 5     # Top K nearest neighbors per window

    print(f"Fitting TruncatedSVD (dim={DIM}) on reference windows...")
    # Stack all window slices of all reference spectra → fit global SVD
    ref_windows_all = []
    for w_start, w_end in windows:
        ref_windows_all.append(ref_matrix[:, w_start:w_end])
    ref_windows_stack = np.vstack(ref_windows_all)  # (n_metabolites * n_windows, WINDOW_SIZE)
    
    svd = TruncatedSVD(n_components=DIM, random_state=42)
    svd.fit(ref_windows_stack)
    print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")

    # ---- Build FAISS indices (one per window) ----
    print("Building FAISS indices...")
    faiss_indices = []
    ref_embeddings_per_window = []
    
    for w_idx, (w_start, w_end) in enumerate(windows):
        window_data = ref_matrix[:, w_start:w_end]  # (n_metabolites, WINDOW_SIZE)
        embeddings = svd.transform(window_data).astype(np.float32)
        
        # Normalize for cosine similarity search
        faiss.normalize_L2(embeddings)
        
        index = faiss.IndexFlatIP(DIM)  # Inner product = cosine similarity after L2 norm
        index.add(embeddings)
        faiss_indices.append(index)
        ref_embeddings_per_window.append(embeddings)
    
    print(f"Built {len(faiss_indices)} FAISS indices, each with {n_metabolites} vectors.")

    # ---- Sample matrix ----
    sample_cols = [c for c in df_data.columns if c != 'ppm']
    n_samples = len(sample_cols)
    print(f"\nProcessing {n_samples} samples with Multi-Window FAISS...")
    sample_matrix = df_data[sample_cols].values.T.astype(np.float32)  # (10000, 10001)

    start_time = time.time()
    y_pred = np.zeros((n_samples, n_metabolites), dtype=int)

    # Process all samples batch by batch through each window
    BATCH_SIZE = 500
    vote_matrix = np.zeros((n_samples, n_metabolites), dtype=np.int16)

    for w_idx, (w_start, w_end) in enumerate(tqdm(windows, desc="FAISS search per window")):
        for batch_start in range(0, n_samples, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_samples)
            batch = sample_matrix[batch_start:batch_end, w_start:w_end]
            
            # Project to DIM
            batch_emb = svd.transform(batch).astype(np.float32)
            faiss.normalize_L2(batch_emb)
            
            # Search FAISS
            D, I = faiss_indices[w_idx].search(batch_emb, TOP_K_PER_WINDOW)
            
            # Accumulate votes
            for s_local, neighbors in enumerate(I):
                s_global = batch_start + s_local
                for meta_idx in neighbors:
                    vote_matrix[s_global, meta_idx] += 1

    # Threshold votes to get final predictions
    y_pred = (vote_matrix >= VOTE_THRESHOLD).astype(int)
    
    end_time = time.time()
    elapsed = end_time - start_time

    # ---- Evaluate ----
    print("Evaluating Results...")
    meta_to_idx = {m: i for i, m in enumerate(metabolites)}
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
        'Experiment': 'EXP-D: Multi-Window FAISS',
        'N_Windows': n_windows,
        'Window_Size_ppm': 1.0,
        'Embedding_Dim': DIM,
        'Top_K_Per_Window': TOP_K_PER_WINDOW,
        'Vote_Threshold': VOTE_THRESHOLD,
        'Time_Seconds': round(elapsed, 2),
        'Time_Per_Sample_Ms': round((elapsed / n_samples) * 1000, 2),
        'Precision_Macro': round(precision, 4),
        'Recall_Macro': round(recall, 4),
        'F1_Macro': round(f1, 4),
        'Exact_Match_Ratio': round(exact_match_ratio, 4)
    }

    print("\n--- RESULTS ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
