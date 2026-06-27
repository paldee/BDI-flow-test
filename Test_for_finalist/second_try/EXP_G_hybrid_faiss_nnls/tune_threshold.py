import os
import time
import numpy as np
import pandas as pd
import faiss
import scipy.optimize
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

ref_matrix_T = None

def init_worker(cache_path):
    global ref_matrix_T
    ref_matrix_T = np.load(cache_path, mmap_mode='r').T

def process_sample_nnls(args):
    global ref_matrix_T
    s_idx, candidates, X_sample = args
    if len(candidates) == 0:
        return s_idx, np.array([])
        
    Phi_sub = ref_matrix_T[:, candidates]
    w, _ = scipy.optimize.nnls(Phi_sub, X_sample)
    return s_idx, w

def run_threshold_tuning():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'mock_nmr_data_10k.csv')
    gt_path = os.path.join(base_dir, '..', 'data', 'mock_ground_truth_10k.csv')
    ref_path = os.path.join(base_dir, '..', 'data', 'hmdb_reference', 'hmdb_reference_library.csv')
    
    print("Loading data...")
    df_data = pd.read_csv(data_path)
    df_gt = pd.read_csv(gt_path)
    df_ref = pd.read_csv(ref_path)
    
    ppm_values = df_data['ppm'].values
    n_points = len(ppm_values)
    metabolites = df_ref['Name'].unique()
    n_metabolites = len(metabolites)
    
    global ref_matrix_T
    cache_path = os.path.join(base_dir, 'ref_matrix_cache.npy')
    ref_matrix = np.load(cache_path)
    ref_matrix_T = ref_matrix.T
        
    sample_cols = [c for c in df_data.columns if c != 'ppm']
    N_SEARCH = 1000
    sample_cols = sample_cols[:N_SEARCH]
    
    sample_matrix = df_data[sample_cols].values.T.astype(np.float32)
    n_samples = sample_matrix.shape[0]
    
    meta_to_idx = {m: i for i, m in enumerate(metabolites)}
    y_true = np.zeros((n_samples, n_metabolites), dtype=int)
    for s_idx, (_, gt_row) in enumerate(df_gt.head(n_samples).iterrows()):
        for meta, idx in meta_to_idx.items():
            if meta in gt_row and gt_row[meta] > 0:
                y_true[s_idx, idx] = 1

    print("\n--- STAGE 1: FAISS Filtering ---")
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
                    
    y_pred_faiss = (vote_matrix >= vote_thresh).astype(int)

    print("\n--- STAGE 2: Running Fast NNLS ---")
    tasks = []
    for s_idx in range(n_samples):
        candidates = np.where(y_pred_faiss[s_idx] == 1)[0]
        if len(candidates) > 0:
            X_sample = sample_matrix[s_idx]
            tasks.append((s_idx, candidates, X_sample))
    
    max_workers = 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_sample_nnls, tasks), total=len(tasks), desc="NNLS Computing"))

    print("\n--- STAGE 3: Threshold Grid Search ---")
    fixed_thresholds = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
    adaptive_ratios = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20]
    
    best_f1 = 0
    best_params = {}
    
    for fixed in fixed_thresholds:
        for ratio in adaptive_ratios:
            y_pred_final = np.zeros_like(y_true)
            
            for s_idx, w in results:
                if len(w) == 0: continue
                candidates = np.where(y_pred_faiss[s_idx] == 1)[0]
                
                max_w = np.max(w) if len(w) > 0 else 0
                
                for local_idx, weight in enumerate(w):
                    # Condition: must pass BOTH fixed absolute threshold AND adaptive relative threshold
                    if weight > fixed and weight > (max_w * ratio):
                        global_idx = candidates[local_idx]
                        y_pred_final[s_idx, global_idx] = 1
                        
            f1 = f1_score(y_true, y_pred_final, average='macro', zero_division=0)
            prec = precision_score(y_true, y_pred_final, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred_final, average='macro', zero_division=0)
            
            print(f"Fixed: {fixed:.2f} | Ratio: {ratio:.2f} -> F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = {'fixed': fixed, 'ratio': ratio}
                
    print(f"\n🏆 Best Parameters: {best_params} with F1: {best_f1:.4f}")

if __name__ == '__main__':
    run_threshold_tuning()
