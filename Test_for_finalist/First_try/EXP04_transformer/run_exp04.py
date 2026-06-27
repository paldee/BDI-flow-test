import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

class NMRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # (B, 1, 10001)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0) # (1, max_len, d_model)

    def forward(self, x):
        # x shape: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class NMRTransformer(nn.Module):
    def __init__(self, num_classes):
        super(NMRTransformer, self).__init__()
        
        # Downsample CNN to reduce sequence length from 10001 to manageable size
        # 10001 -> pool/conv -> length ~100
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), # Length / 20
            
            nn.Conv1d(16, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) # Length / 80 overall -> ~125
        )
        
        self.d_model = 64
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=1000)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x: (B, 1, seq_len)
        features = self.feature_extractor(x) # (B, 64, ~125)
        
        # Reshape for transformer: (B, seq_len, d_model)
        features = features.permute(0, 2, 1) # (B, ~125, 64)
        
        # Add positional encoding
        features = self.pos_encoder(features)
        
        # Transformer
        encoded = self.transformer_encoder(features) # (B, seq_len, d_model)
        
        # Global Average Pooling over the sequence
        pooled = encoded.mean(dim=1) # (B, d_model)
        
        out = self.classifier(pooled)
        return out

def run_exp04():
    print("--- EXP04: Transformer Architecture ---")
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
    
    X = df_data.drop(columns=['ppm']).values.T 
    y_df = df_gt.set_index("Sample_ID")[metabolites]
    y = (y_df.values > 0).astype(int) 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    
    print("Building CNN+Transformer model...")
    model = NMRTransformer(num_classes).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
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
    
    with open(os.path.join(results_dir, "exp04_report.txt"), "w") as f:
        f.write(f"EXP04: CNN + Transformer\n")
        f.write(f"Test Set Size: {len(y_true)}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall: {recall:.4f}\n")
        f.write(f"Macro F1-Score: {f1:.4f}\n")
        f.write(f"Elapsed Time: {elapsed:.2f} seconds\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        
    torch.save(model.state_dict(), os.path.join(results_dir, "model_exp04.pt"))
    print(f"Results saved to {results_dir}/exp04_report.txt")

if __name__ == "__main__":
    run_exp04()
