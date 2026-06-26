import os
import time
import numpy as np
import pandas as pd
import faiss
from sklearn.metrics import f1_score
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
    df_data = pd.read_csv(data_path)
    df_gt = pd.read_csv(gt_path)
    df_ref = pd.read_csv(ref_path)
    return df_data, df_gt, df_ref

def run_faiss_pipeline(sample_matrix, ref_matrix, y_true, metabolites, n_points, window_size_ppm=1.0, top_k=5, vote_thresh=2, dim=64):
    WINDOW_SIZE = int(window_size_ppm * 1000)
    WINDOW_STEP = int(WINDOW_SIZE / 2)
    n_samples = sample_matrix.shape[0]
    n_metabolites = len(metabolites)
    
    windows = []
    start = 0
    while start + WINDOW_SIZE <= n_points:
        windows.append((start, start + WINDOW_SIZE))
        start += WINDOW_STEP
    
    # SVD
    ref_windows_all = []
    for w_start, w_end in windows:
        ref_windows_all.append(ref_matrix[:, w_start:w_end])
    ref_windows_stack = np.vstack(ref_windows_all)
    
    svd = TruncatedSVD(n_components=dim, random_state=42)
    svd.fit(ref_windows_stack)
    
    # Build indices
    faiss_indices = []
    for w_start, w_end in windows:
        window_data = ref_matrix[:, w_start:w_end]
        embeddings = svd.transform(window_data).astype(np.float32)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss_indices.append(index)
        
    # Search
    BATCH_SIZE = 1000
    vote_matrix = np.zeros((n_samples, n_metabolites), dtype=np.int16)
    
    for w_idx, (w_start, w_end) in enumerate(windows):
        for batch_start in range(0, n_samples, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_samples)
            batch = sample_matrix[batch_start:batch_end, w_start:w_end]
            
            batch_emb = svd.transform(batch).astype(np.float32)
            faiss.normalize_L2(batch_emb)
            
            D, I = faiss_indices[w_idx].search(batch_emb, top_k)
            
            for s_local, neighbors in enumerate(I):
                s_global = batch_start + s_local
                for meta_idx in neighbors:
                    vote_matrix[s_global, meta_idx] += 1
                    
    y_pred = (vote_matrix >= vote_thresh).astype(int)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return f1

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_data, df_gt, df_ref = load_data(base_dir)
    
    ppm_values = df_data['ppm'].values
    n_points = len(ppm_values)
    metabolites = df_ref['Name'].unique()
    n_metabolites = len(metabolites)
    
    print("Building ref matrix...")
    ref_matrix = np.zeros((n_metabolites, n_points), dtype=np.float32)
    for i, meta in enumerate(metabolites):
        meta_peaks = df_ref[df_ref['Name'] == meta]
        ref_matrix[i] = lorentzian_vectorized(ppm_values, meta_peaks)
        
    sample_cols = [c for c in df_data.columns if c != 'ppm']
    sample_matrix = df_data[sample_cols].values.T.astype(np.float32)
    
    meta_to_idx = {m: i for i, m in enumerate(metabolites)}
    y_true = np.zeros((len(sample_cols), n_metabolites), dtype=int)
    for s_idx, (_, gt_row) in enumerate(df_gt.iterrows()):
        for meta, idx in meta_to_idx.items():
            if meta in gt_row and gt_row[meta] > 0:
                y_true[s_idx, idx] = 1

    # Grid Search Space
    windows_ppm = [0.2, 0.5, 1.0, 1.5]
    top_ks = [3, 5, 10, 15]
    vote_thresholds = [1, 2, 3, 5]
    
    best_f1 = 0
    best_params = {}
    
    print("Starting Grid Search for FAISS...")
    for w in windows_ppm:
        for k in top_ks:
            for v in vote_thresholds:
                # Basic logical constraint: can't demand more votes than windows
                num_windows = int(10.0 / (w/2)) - 1
                if v > num_windows: continue
                
                f1 = run_faiss_pipeline(sample_matrix, ref_matrix, y_true, metabolites, n_points, 
                                        window_size_ppm=w, top_k=k, vote_thresh=v)
                
                print(f"W={w}ppm, Top={k}, Vote={v} --> F1 = {f1:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {'Window_ppm': w, 'Top_K': k, 'Vote_Thresh': v}
                    
    print("\n--- GRID SEARCH RESULTS ---")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best Params: {best_params}")

if __name__ == '__main__':
    main()
