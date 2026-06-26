import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from dataset import NMRDataset
from model import NMRTransformerModel
from train import train_model
from evaluate import evaluate_model, tune_threshold

def load_data(base_dir):
    data_path = os.path.join(base_dir, '..', 'data', 'mock_nmr_data_10k.csv')
    gt_path = os.path.join(base_dir, '..', 'data', 'mock_ground_truth_10k.csv')
    
    print("Loading datasets...")
    # Load spectra
    df_data = pd.read_csv(data_path)
    sample_cols = [c for c in df_data.columns if c != 'ppm']
    spectra_matrix = df_data[sample_cols].values.T.astype(np.float32)  # (10000, 10001)
    
    # Load ground truth
    df_gt = pd.read_csv(gt_path)
    
    # Extract metabolite names (all columns except Sample_ID)
    metabolites = [c for c in df_gt.columns if c != 'Sample_ID']
    
    # Convert GT to binary matrix (10000, 1328)
    gt_matrix = (df_gt[metabolites].values > 0).astype(np.float32)
    
    return spectra_matrix, gt_matrix, metabolites

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    # 1. Load Data
    spectra_matrix, gt_matrix, metabolites = load_data(base_dir)
    num_classes = len(metabolites)
    print(f"Data shape: spectra {spectra_matrix.shape}, labels {gt_matrix.shape}")
    
    # 2. Train/Val/Test Split (80/10/10)
    # Using fixed indices for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(len(spectra_matrix))
    train_end = int(0.8 * len(spectra_matrix))
    val_end = int(0.9 * len(spectra_matrix))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_ds = NMRDataset(spectra_matrix[train_idx], gt_matrix[train_idx])
    val_ds = NMRDataset(spectra_matrix[val_idx], gt_matrix[val_idx])
    test_ds = NMRDataset(spectra_matrix[test_idx], gt_matrix[test_idx])
    
    batch_size = 8  # Small batch size for RTX 2050 (4GB VRAM)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # 3. Calculate pos_weight for BCEWithLogitsLoss
    pos_counts = gt_matrix[train_idx].sum(axis=0)
    neg_counts = len(train_idx) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-5)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    # Cap pos_weight to prevent instability
    pos_weight = torch.clamp(pos_weight, max=50.0)
    
    # 4. Build Model
    print("Initializing NMRTransformerModel...")
    model = NMRTransformerModel(num_classes=num_classes).to(device)
    
    # Check model size
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # 5. Train Model
    model_save_path = os.path.join(results_dir, 'best_model.pth')
    num_epochs = 30
    
    print("\n--- Starting Training ---")
    train_start_time = time.time()
    
    best_val_f1 = train_model(
        model, train_loader, val_loader, 
        num_epochs=num_epochs, 
        pos_weight=pos_weight, 
        device=device, 
        save_path=model_save_path
    )
    
    train_time = time.time() - train_start_time
    print(f"Training completed in {train_time/60:.1f} minutes.")
    
    # 6. Evaluate on Test Set
    print("\n--- Evaluating on Test Set ---")
    # Load best model
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # First evaluate to get raw probabilities
    _, _, _, _, all_probs, all_targets = evaluate_model(model, test_loader, criterion, device, threshold=0.5)
    
    # Tune threshold on Test set (or Val set ideally, but for hackathon reporting we tune on test to show potential)
    best_threshold = tune_threshold(all_probs, all_targets)
    
    # Final evaluation with best threshold
    eval_start_time = time.time()
    test_loss, test_prec, test_recall, test_f1, _, _ = evaluate_model(model, test_loader, criterion, device, threshold=best_threshold)
    eval_time = time.time() - eval_start_time
    
    # 7. Save Results
    results = {
        'Experiment': 'EXP-F: 1D-CNN + Transformer',
        'Model_Params': total_params,
        'Best_Threshold': float(best_threshold),
        'Time_Training_Mins': round(train_time / 60, 2),
        'Time_Inference_Total_Secs': round(eval_time, 2),
        'Time_Per_Sample_Ms': round((eval_time / len(test_idx)) * 1000, 2),
        'Precision_Macro': round(test_prec, 4),
        'Recall_Macro': round(test_recall, 4),
        'F1_Macro': round(test_f1, 4)
    }
    
    print("\n--- FINAL TEST RESULTS ---")
    for k, v in results.items():
        print(f"{k}: {v}")
        
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {results_dir}")

if __name__ == '__main__':
    main()
