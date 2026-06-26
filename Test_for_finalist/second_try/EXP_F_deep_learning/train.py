import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import time
from evaluate import evaluate_model
import os

def train_model(model, train_loader, val_loader, num_epochs, pos_weight, device, save_path):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Using a modest learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    scheduler = OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=num_epochs)
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = 0.0
    patience_counter = 0
    PATIENCE = 5
    
    accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        optimizer.zero_grad()
        
        for i, (batch_spectra, batch_labels) in enumerate(train_loader):
            batch_spectra = batch_spectra.to(device)
            batch_labels = batch_labels.to(device)
            
            with torch.amp.autocast('cuda'):
                logits = model(batch_spectra)
                loss = criterion(logits, batch_labels)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            total_loss += loss.item() * accumulation_steps * batch_spectra.size(0)
            
        avg_train_loss = total_loss / len(train_loader.dataset)
        
        # Validation
        val_loss, val_prec, val_recall, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device, threshold=0.5)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.1f}s] - Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} (P: {val_prec:.4f}, R: {val_recall:.4f})")
        
        # Early stopping and model saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model (F1 improved to {best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    return best_val_f1
