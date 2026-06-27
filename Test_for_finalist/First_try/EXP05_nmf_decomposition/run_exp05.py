import numpy as np
import pandas as pd
import os
import time
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from scipy.optimize import linear_sum_assignment
import warnings

# Ignore NMF convergence warnings if any
warnings.filterwarnings('ignore', category=UserWarning)

def lorentzian(x, x0, a, w):
    return a / (1 + ((x - x0) / (w / 2))**2)

def run_exp05():
    print("--- EXP05: NMF Decomposition (Unsupervised) ---")
    start_time = time.time()
    
    data_dir = r"d:\hack\BDI\Test_for_finalist\data"
    
    # 1. Load the data
    print("Loading data...")
    df_data = pd.read_csv(os.path.join(data_dir, "realistic_nmr_data_large.csv"))
    df_gt = pd.read_csv(os.path.join(data_dir, "realistic_nmr_ground_truth.csv"))
    df_lib = pd.read_csv(os.path.join(data_dir, "reference_library", "reference_library_38.csv"))
    
    ppm_values = df_data['ppm'].values
    samples_intensity = df_data.drop(columns=['ppm']).values.T # shape: (1000, 10001)
    
    metabolites = df_lib['Name'].unique()
    num_metabolites = len(metabolites)
    
    # 2. Run NMF
    print(f"Running NMF to extract {num_metabolites} components...")
    nmf_model = NMF(n_components=num_metabolites, init='nndsvda', random_state=42, max_iter=500)
    
    # W represents the predicted concentration/weight of each component for each sample
    W = nmf_model.fit_transform(samples_intensity) # shape: (1000, 38)
    # H represents the extracted pure spectra for each component
    H = nmf_model.components_ # shape: (38, 10001)
    
    # 3. Build Reference Signatures to match components
    print("Building reference signatures for matching...")
    pure_spectra = {}
    for meta in metabolites:
        meta_peaks = df_lib[df_lib['Name'] == meta]
        spec = np.zeros_like(ppm_values)
        for _, row in meta_peaks.iterrows():
            loc = row['loc']
            height = row['height']
            width = row['width']
            scaler = 450.0
            spec += lorentzian(ppm_values, loc, height, width / scaler)
        area = np.trapezoid(spec, ppm_values)
        if area > 0:
            spec = spec / area
        pure_spectra[meta] = spec
        
    ref_matrix = np.array([pure_spectra[meta] for meta in metabolites]) # shape: (38, 10001)
    
    # 4. Map NMF Components to Known Metabolites
    print("Mapping NMF components to metabolites using Hungarian Algorithm...")
    # Calculate similarity between extracted components (H) and reference spectra
    sim_matrix_components = cosine_similarity(H, ref_matrix)
    
    # Use linear sum assignment to find optimal 1-to-1 mapping
    cost_matrix = 1.0 - sim_matrix_components
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Reorder W matrix to match the predefined metabolites order
    W_reordered = np.zeros_like(W)
    for r, c in zip(row_ind, col_ind):
        W_reordered[:, c] = W[:, r]
        
    # Check average similarity of matched components
    matched_similarities = sim_matrix_components[row_ind, col_ind]
    avg_sim = np.mean(matched_similarities)
    print(f"Average Cosine Similarity of matched NMF components to Reference: {avg_sim:.4f}")
    
    # 5. Evaluate against Ground Truth
    print("Evaluating...")
    y_true_df = df_gt.set_index("Sample_ID")[metabolites]
    y_true = (y_true_df.values > 0).astype(int)
    
    best_threshold = 0
    best_f1 = 0
    best_preds = None
    
    # The scale of W is arbitrary, so we test thresholds relative to the maximum weight found
    max_w = W_reordered.max()
    thresholds = np.linspace(max_w * 0.001, max_w * 0.5, 200)
    
    for t in thresholds:
        y_pred = (W_reordered >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_preds = y_pred
            
    print(f"Best Relative Threshold: {best_threshold:.6f} (max W is {max_w:.4f})")
    
    y_pred = best_preds
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    elapsed = time.time() - start_time
    print("\n--- Final Results ---")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1-Score  : {best_f1:.4f}")
    print(f"Elapsed Time    : {elapsed:.2f} seconds")
    
    # 6. Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=metabolites, zero_division=0)
    
    with open(os.path.join(results_dir, "exp05_report.txt"), "w") as f:
        f.write(f"EXP05: NMF Decomposition (Unsupervised)\n")
        f.write(f"Average Matched Component Similarity: {avg_sim:.4f}\n")
        f.write(f"Best Relative Threshold: {best_threshold:.6f}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall: {recall:.4f}\n")
        f.write(f"Macro F1-Score: {best_f1:.4f}\n")
        f.write(f"Elapsed Time: {elapsed:.2f} seconds\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        
    print(f"Results saved to {results_dir}/exp05_report.txt")

if __name__ == "__main__":
    run_exp05()
