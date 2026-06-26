import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm

def lorentzian(x, x0, a, w):
    return a / (1 + ((x - x0) / (w / 2))**2)

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
    print("Building reference spectra matrix...")
    metabolites = df_ref['Name'].unique()
    ref_matrix = []
    
    # Vectorized computation
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
        
    return np.array(ref_matrix), metabolites

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    df_data, df_gt, df_ref = load_data(base_dir)
    ppm_values = df_data['ppm'].values
    
    # 1. Build Reference Matrix (N_metabolites, 10001)
    ref_matrix, metabolites = build_reference_spectra_matrix(df_ref, ppm_values)
    
    # 2. Prepare Sample Matrix (N_samples, 10001)
    sample_cols = [c for c in df_data.columns if c != 'ppm']
    print(f"Extracting sample matrix for {len(sample_cols)} samples...")
    sample_matrix = df_data[sample_cols].values.T  # Shape: (10000, 10001)
    
    print("Computing Cosine Similarity Matrix...")
    start_time = time.time()
    
    # Calculate cosine similarity (10000, 1328)
    sim_matrix = cosine_similarity(sample_matrix, ref_matrix)
    
    # Thresholding: If similarity > 0.1, we predict 'Present'
    # 0.1 is chosen because in complex mixtures, the pure spectrum is heavily diluted and masked, 
    # resulting in a low absolute cosine score against the entire mixture.
    THRESHOLD = 0.05
    y_pred = (sim_matrix > THRESHOLD).astype(int)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("Evaluating Results...")
    
    # Prepare Ground Truth Binary Matrix
    y_true = []
    for _, gt_row in df_gt.iterrows():
        true_row = []
        for meta in metabolites:
            if meta in gt_row and gt_row[meta] > 0:
                true_row.append(1)
            else:
                true_row.append(0)
        y_true.append(true_row)
        
    y_true = np.array(y_true)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    exact_match_ratio = np.sum(np.all(y_true == y_pred, axis=1)) / len(sample_cols)
    
    results = {
        'Experiment': 'EXP-B: Cosine Similarity',
        'Time_Seconds': round(elapsed, 2),
        'Time_Per_Sample_Ms': round((elapsed / len(sample_cols)) * 1000, 2),
        'Precision_Macro': round(precision, 4),
        'Recall_Macro': round(recall, 4),
        'F1_Macro': round(f1, 4),
        'Exact_Match_Ratio': round(exact_match_ratio, 4),
        'Threshold_Used': THRESHOLD
    }
    
    print("\n--- RESULTS ---")
    for k, v in results.items():
        print(f"{k}: {v}")
        
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {results_dir}")

if __name__ == '__main__':
    main()
