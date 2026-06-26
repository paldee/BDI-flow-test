import os
import time
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import classification_report, precision_recall_fscore_support
import json

def load_data(base_dir):
    data_path = os.path.join(base_dir, '..', 'data', 'mock_nmr_data_10k.csv')
    gt_path = os.path.join(base_dir, '..', 'data', 'mock_ground_truth_10k.csv')
    ref_path = os.path.join(base_dir, '..', 'data', 'hmdb_reference', 'hmdb_reference_library.csv')
    
    print("Loading data...")
    df_data = pd.read_csv(data_path)
    df_gt = pd.read_csv(gt_path)
    df_ref = pd.read_csv(ref_path)
    
    return df_data, df_gt, df_ref

def build_reference_dict(df_ref):
    """
    Convert reference DataFrame into a dictionary for fast lookup.
    Structure: { metabolite_name: [list of peak ppms] }
    """
    ref_dict = {}
    for name, group in df_ref.groupby('Name'):
        # Sort by height descending, take top 15 peaks to avoid fitting noise
        top_peaks = group.sort_values('height', ascending=False).head(15)
        ref_dict[name] = top_peaks['loc'].values
    return ref_dict

def evaluate_sample(sample_intensity, ppm_values, ref_dict, tolerance=0.015, min_match_ratio=0.6):
    """
    1. Find peaks in the sample spectrum
    2. Match found peaks against the reference library
    """
    # Peak picking
    # height threshold avoids pure noise. distance prevents overlapping identical peaks
    peaks, _ = find_peaks(sample_intensity, height=0.02, distance=5)
    found_ppms = ppm_values[peaks]
    
    predicted_metabolites = []
    
    # Matching against reference
    for meta, ref_ppms in ref_dict.items():
        if len(ref_ppms) == 0:
            continue
            
        matched_count = 0
        for ref_ppm in ref_ppms:
            # Check if any found peak is within tolerance of ref_ppm
            if np.any(np.abs(found_ppms - ref_ppm) <= tolerance):
                matched_count += 1
                
        match_ratio = matched_count / len(ref_ppms)
        if match_ratio >= min_match_ratio:
            predicted_metabolites.append(meta)
            
    return predicted_metabolites

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    df_data, df_gt, df_ref = load_data(base_dir)
    ref_dict = build_reference_dict(df_ref)
    
    ppm_values = df_data['ppm'].values
    sample_cols = [c for c in df_data.columns if c != 'ppm']
    
    print(f"Running Peak Picking Evaluation on {len(sample_cols)} samples...")
    start_time = time.time()
    
    all_predictions = []
    
    for sample_col in sample_cols:
        sample_intensity = df_data[sample_col].values
        preds = evaluate_sample(sample_intensity, ppm_values, ref_dict)
        all_predictions.append({
            'Sample_ID': sample_col,
            'Predicted': preds
        })
        
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Evaluation
    print("Evaluating Results...")
    metabolites = list(ref_dict.keys())
    
    # Binarize Ground Truth (1 if concentration > 0 else 0)
    y_true = []
    y_pred = []
    
    # Map predictions back to binary matrix format
    pred_df = pd.DataFrame(all_predictions)
    
    for _, gt_row in df_gt.iterrows():
        sample_id = gt_row['Sample_ID']
        preds = pred_df[pred_df['Sample_ID'] == sample_id]['Predicted'].values[0]
        
        true_row = []
        pred_row = []
        for meta in metabolites:
            # Check if meta is a column in GT (it might not be if 0 concentration)
            if meta in gt_row and gt_row[meta] > 0:
                true_row.append(1)
            else:
                true_row.append(0)
                
            if meta in preds:
                pred_row.append(1)
            else:
                pred_row.append(0)
                
        y_true.append(true_row)
        y_pred.append(pred_row)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # We calculate macro F1 across all metabolites
    # Some metabolites might never appear in GT, which throws warnings. 
    # Use zero_division=0 to handle them silently.
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # Find exact match ratio (how many samples had EXACTLY correct metabolites)
    exact_matches = np.sum(np.all(y_true == y_pred, axis=1))
    exact_match_ratio = exact_matches / len(sample_cols)
    
    results = {
        'Experiment': 'EXP-E: Peak Picking',
        'Time_Seconds': round(elapsed, 2),
        'Time_Per_Sample_Ms': round((elapsed / len(sample_cols)) * 1000, 2),
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
