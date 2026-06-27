import os
import time
import numpy as np
import pandas as pd
import faiss
import scipy.optimize
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

ref_matrix_T = None

def init_worker(cache_path):
    global ref_matrix_T
    ref_matrix_T = np.load(cache_path, mmap_mode='r').T

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
    
    print("Loading data...")
    df_data = pd.read_csv(data_path)
    df_gt = pd.read_csv(gt_path)
    df_ref = pd.read_csv(ref_path)
    return df_data, df_gt, df_ref

def process_sample_nnls(args):
    s_idx, candidates, X_sample, fixed_thresh, adaptive_ratio = args
    if len(candidates) == 0:
        return s_idx, []
        
    global ref_matrix_T
    Phi_sub = ref_matrix_T[:, candidates]
    w, _ = scipy.optimize.nnls(Phi_sub, X_sample)
    
    final_candidates = []
    max_w = np.max(w) if len(w) > 0 else 0
    
    for local_idx, weight in enumerate(w):
        if weight > fixed_thresh and weight > (max_w * adaptive_ratio):
            global_idx = candidates[local_idx]
            final_candidates.append(global_idx)
            
    return s_idx, final_candidates

def run_hybrid_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_data, df_gt, df_ref = load_data(base_dir)
    
    ppm_values = df_data['ppm'].values
    n_points = len(ppm_values)
    metabolites = df_ref['Name'].unique()
    n_metabolites = len(metabolites)
    
    cache_path = os.path.join(base_dir, 'ref_matrix_cache.npy')
    
    if os.path.exists(cache_path):
        print(f"Loading cached ref_matrix from {cache_path}...")
        ref_matrix = np.load(cache_path)
    else:
        print("Building ref matrix...")
        ref_matrix = np.zeros((n_metabolites, n_points), dtype=np.float32)
        for i, meta in enumerate(tqdm(metabolites, desc="Building Refs")):
            meta_peaks = df_ref[df_ref['Name'] == meta]
            ref_matrix[i] = lorentzian_vectorized(ppm_values, meta_peaks)
        np.save(cache_path, ref_matrix)
        print("Cache saved!")
        
    sample_cols = [c for c in df_data.columns if c != 'ppm']
    sample_matrix = df_data[sample_cols].values.T.astype(np.float32)
    n_samples = sample_matrix.shape[0]
    
    meta_to_idx = {m: i for i, m in enumerate(metabolites)}
    y_true = np.zeros((n_samples, n_metabolites), dtype=int)
    for s_idx, (_, gt_row) in enumerate(df_gt.head(n_samples).iterrows()):
        for meta, idx in meta_to_idx.items():
            if meta in gt_row and gt_row[meta] > 0:
                y_true[s_idx, idx] = 1

    print("\n--- STAGE 1: FAISS Filtering ---")
    start_time = time.time()
    window_size_ppm = 0.2
    top_k = 5
    vote_thresh = 1
    dim = 64
    
    WINDOW_SIZE = int(window_size_ppm * 1000)
    WINDOW_STEP = int(WINDOW_SIZE / 2)
    windows = [(start, start + WINDOW_SIZE) for start in range(0, n_points - WINDOW_SIZE + 1, WINDOW_STEP)]
    
    ref_windows_stack = np.vstack([ref_matrix[:, w_start:w_end] for w_start, w_end in windows])
    svd = TruncatedSVD(n_components=dim, random_state=42)
    svd.fit(ref_windows_stack)
    
    faiss_indices = []
    for w_start, w_end in windows:
        window_data = ref_matrix[:, w_start:w_end]
        embeddings = svd.transform(window_data).astype(np.float32)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss_indices.append(index)
        
    BATCH_SIZE = 1000
    vote_matrix = np.zeros((n_samples, n_metabolites), dtype=np.int16)
    
    for w_idx, (w_start, w_end) in enumerate(tqdm(windows, desc="FAISS Searching")):
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
                    
    y_pred_faiss = (vote_matrix >= vote_thresh).astype(int)
    faiss_time = time.time() - start_time
    
    print(f"FAISS Stage 1 Time: {faiss_time:.2f} s")

    print("\n--- STAGE 2: NNLS + Adaptive Thresholding ---")
    start_time_nnls = time.time()
    
    # WINNING PARAMETERS FROM GRID SEARCH
    fixed_thresh = 0.25
    adaptive_ratio = 0.10
    y_pred_final = np.zeros_like(y_true)
    
    tasks = []
    for s_idx in range(n_samples):
        candidates = np.where(y_pred_faiss[s_idx] == 1)[0]
        if len(candidates) > 0:
            X_sample = sample_matrix[s_idx]
            tasks.append((s_idx, candidates, X_sample, fixed_thresh, adaptive_ratio))
    
    max_workers = 4
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(cache_path,)) as executor:
        results = list(tqdm(executor.map(process_sample_nnls, tasks), total=len(tasks), desc="NNLS Reranking", smoothing=0.1))
        
    for s_idx, final_cands in results:
        for global_idx in final_cands:
            y_pred_final[s_idx, global_idx] = 1

    nnls_time = time.time() - start_time_nnls
    
    final_f1 = f1_score(y_true, y_pred_final, average='macro', zero_division=0)
    final_prec = precision_score(y_true, y_pred_final, average='macro', zero_division=0)
    final_rec = recall_score(y_true, y_pred_final, average='macro', zero_division=0)
    
    print(f"\nFINAL FAISS+NNLS (Adaptive Ratio > {adaptive_ratio}, Fixed > {fixed_thresh}):")
    print(f"Macro F1: {final_f1:.4f} | Prec: {final_prec:.4f} | Rec: {final_rec:.4f}")
    print(f"NNLS Time per sample: {(nnls_time / n_samples) * 1000:.2f} ms")
    print(f"Total Pipeline Time per sample: {((faiss_time + nnls_time) / n_samples) * 1000:.2f} ms")

if __name__ == '__main__':
    run_hybrid_pipeline()
