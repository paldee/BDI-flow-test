import sys
import os

# Add the path to the resnet1d module
sys.path.append(r"d:\hack\BDI")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from resnet1d.resnet1d import ResNet1D

class NMRDataset(Dataset):
    def __init__(self, X, y):
        # Shape of X: (samples, 10001)
        # Reshape to (samples, channels, length) for Conv1D
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def run_exp03():
    print("--- EXP03: 1D-ResNet Architecture ---")
    start_time = time.time()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = r"d:\hack\BDI\Test_for_finalist\data"
    
    print("Loading data...")
    df_data = pd.read_csv(os.path.join(data_dir, "realistic_nmr_data_large.csv"))
    df_gt = pd.read_csv(os.path.join(data_dir, "realistic_nmr_ground_truth.csv"))
    df_lib = pd.read_csv(os.path.join(data_dir, "reference_library", "reference_library_38.csv"))
    
    metabolites = df_lib['Name'].unique()
    num_classes = len(metabolites)
    
    X = df_data.drop(columns=['ppm']).values.T # (1000, 10001)
    
    y_df = df_gt.set_index("Sample_ID")[metabolites]
    y = (y_df.values > 0).astype(int) # Multi-label targets (1/0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale X data
    X_train_max = X_train.max(axis=1, keepdims=True)
    X_train_max[X_train_max == 0] = 1e-10
    X_train = X_train / X_train_max
    
    X_test_max = X_test.max(axis=1, keepdims=True)
    X_test_max[X_test_max == 0] = 1e-10
    X_test = X_test / X_test_max
    
    train_dataset = NMRDataset(X_train, y_train)
    test_dataset = NMRDataset(X_test, y_test)
    
    batch_size = 16 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Building ResNet1D model...")
    model = ResNet1D(
        in_channels=1, 
        base_filters=64, 
        kernel_size=15, 
        stride=2, 
        groups=1, 
        n_block=8, 
        n_classes=num_classes,
        use_bn=True,
        use_do=True
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 50 
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
    print("Evaluating on Test Set...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.append(preds)
            all_targets.append(targets.numpy())
            
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    elapsed = time.time() - start_time
    print("\n--- Final Results (Test Set) ---")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1-Score  : {f1:.4f}")
    print(f"Elapsed Time    : {elapsed:.2f} seconds")
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=metabolites, zero_division=0)
    
    with open(os.path.join(results_dir, "exp03_report.txt"), "w") as f:
        f.write(f"EXP03: 1D-ResNet\n")
        f.write(f"Test Set Size: {len(y_true)}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall: {recall:.4f}\n")
        f.write(f"Macro F1-Score: {f1:.4f}\n")
        f.write(f"Elapsed Time: {elapsed:.2f} seconds\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        
    torch.save(model.state_dict(), os.path.join(results_dir, "model_exp03.pt"))
    print(f"Results saved to {results_dir}/exp03_report.txt")

if __name__ == "__main__":
    run_exp03()
