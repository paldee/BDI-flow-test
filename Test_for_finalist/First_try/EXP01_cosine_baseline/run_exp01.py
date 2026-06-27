import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import time

def lorentzian(x, x0, a, w):
    return a / (1 + ((x - x0) / (w / 2))**2)

def run_exp01():
    print("--- EXP01: Cosine Similarity Baseline ---")
    start_time = time.time()
    
    data_dir = r"d:\hack\BDI\Test_for_finalist\data"
    
    # 1. Load the data
    print("Loading data...")
    df_data = pd.read_csv(os.path.join(data_dir, "realistic_nmr_data_large.csv"))
    df_gt = pd.read_csv(os.path.join(data_dir, "realistic_nmr_ground_truth.csv"))
    df_lib = pd.read_csv(os.path.join(data_dir, "reference_library", "reference_library_38.csv"))
    
    ppm_values = df_data['ppm'].values
    
    # Drop ppm column to get just the sample intensities (shape: 10001 x 1000)
    samples_intensity = df_data.drop(columns=['ppm']).values
    # Transpose to (1000 x 10001) for easier iteration
    samples_intensity = samples_intensity.T 
    sample_ids = df_data.columns.drop('ppm').tolist()
    
    # 2. Build Reference Signatures
    print("Building reference signatures...")
    metabolites = df_lib['Name'].unique()
    pure_spectra = {}
    
    for meta in metabolites:
        meta_peaks = df_lib[df_lib['Name'] == meta]
        spec = np.zeros_like(ppm_values)
        for _, row in meta_peaks.iterrows():
            loc = row['loc']
            height = row['height']
            width = row['width']
            scaler = 450.0
            W = width / scaler
            spec += lorentzian(ppm_values, loc, height, W)
            
        area = np.trapezoid(spec, ppm_values)
        if area > 0:
            spec = spec / area
            
        pure_spectra[meta] = spec
        
    # Stack reference spectra into matrix (38 x 10001)
    ref_matrix = np.array([pure_spectra[meta] for meta in metabolites])
    
    # 3. Calculate Cosine Similarity
    print("Calculating cosine similarities...")
    # Normalize samples and references for dot product
    # norm = sqrt(sum(x^2))
    samples_norm = np.linalg.norm(samples_intensity, axis=1, keepdims=True)
    # Avoid division by zero
    samples_norm[samples_norm == 0] = 1e-10
    samples_normalized = samples_intensity / samples_norm
    
    ref_norm = np.linalg.norm(ref_matrix, axis=1, keepdims=True)
    ref_norm[ref_norm == 0] = 1e-10
    ref_normalized = ref_matrix / ref_norm
    
    # Cosine similarity matrix (1000 samples x 38 metabolites)
    # shape: (1000, 10001) dot (10001, 38) -> (1000, 38)
    sim_matrix = np.dot(samples_normalized, ref_normalized.T)
    
    # 4. Evaluate against Ground Truth
    print("Evaluating...")
    # Ground truth: if concentration > 0, then 1, else 0
    y_true_df = df_gt.set_index("Sample_ID")
    # Ensure columns match metabolites order
    y_true_df = y_true_df[metabolites]
    y_true = (y_true_df.values > 0).astype(int)
    
    # Iterate thresholds to find the best F1
    best_threshold = 0
    best_f1 = 0
    best_preds = None
    
    # Thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        y_pred = (sim_matrix >= t).astype(int)
        # Calculate macro F1
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_preds = y_pred
            
    print(f"Best Threshold: {best_threshold:.4f}")
    
    # Get detailed metrics using the best threshold
    y_pred = best_preds
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    print("\n--- Final Results ---")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1-Score  : {best_f1:.4f}")
    print(f"Elapsed Time    : {time.time() - start_time:.2f} seconds")
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate classification report per metabolite
    report = classification_report(y_true, y_pred, target_names=metabolites, zero_division=0)
    
    with open(os.path.join(results_dir, "exp01_report.txt"), "w") as f:
        f.write(f"EXP01: Cosine Similarity Baseline\n")
        f.write(f"Best Threshold: {best_threshold:.4f}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall: {recall:.4f}\n")
        f.write(f"Macro F1-Score: {best_f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        
    print(f"Results saved to {results_dir}/exp01_report.txt")

if __name__ == "__main__":
    run_exp01()
