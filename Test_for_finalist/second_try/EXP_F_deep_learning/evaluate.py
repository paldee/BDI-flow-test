import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

def evaluate_model(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_spectra, batch_labels in dataloader:
            batch_spectra = batch_spectra.to(device)
            batch_labels = batch_labels.to(device)
            
            with torch.amp.autocast('cuda'):
                logits = model(batch_spectra)
                loss = criterion(logits, batch_labels)
            
            total_loss += loss.item() * batch_spectra.size(0)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(batch_labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_preds = (all_probs >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    
    return avg_loss, precision, recall, f1, all_probs, all_targets

def tune_threshold(all_probs, all_targets):
    best_threshold = 0.5
    best_f1 = 0
    
    print("Tuning threshold...")
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_targets, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    print(f"Best threshold found: {best_threshold:.2f} with F1: {best_f1:.4f}")
    return best_threshold
