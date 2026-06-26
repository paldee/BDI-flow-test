# 🧠 Phase 2: Deep Learning Model — Implementation Plan

> **เป้าหมาย:** สร้างโมเดล 1D-CNN + Transformer สำหรับ Multi-Label Classification  
> เพื่อรับ NMR Spectrum (10,001 จุด) แล้วทำนายว่ามีสาร Metabolite ตัวไหนอยู่บ้าง (1,328 คลาส)  
> เป้าหมาย F1-Score ≥ 70% | เทรนบน RTX 2050 (4GB) | Inference บน H100 (80GB)

---

## 📌 สรุปบริบท (Context)

### ข้อมูลที่มี
| รายการ | ค่า |
|:---|:---|
| Mock Data | `mock_nmr_data_10k.csv` — Shape: (10001, 10001) — 800 MB in memory |
| Ground Truth | `mock_ground_truth_10k.csv` — Shape: (10000, 1329) |
| จำนวน Metabolites | **1,328 คลาส** (Multi-Label) |
| จำนวนสารต่อ Sample (เฉลี่ย) | 15 สาร (สูงสุด 31 สาร) |
| ppm Points per Sample | 10,001 จุด (0.000 - 10.000 ppm ทุก 0.001) |

### Hardware
| เครื่อง | GPU | VRAM | หมายเหตุ |
|:---|:---|:---|:---|
| Dev (ตอนนี้) | RTX 2050 | 4 GB | ต้อง batch size เล็กมาก, mixed precision จำเป็น |
| Prod (วันแข่ง) | H100 | 80 GB | Scale เต็มที่ |

### ทำไม Deep Learning ถึงจำเป็น?
จากการทดลอง Phase 1 ทั้ง 5 วิธี (Peak Picking, Cosine, NMF, DTW, FAISS) วิธีที่ดีที่สุดคือ DTW/FAISS ได้ F1=0.32 เท่านั้น เพราะ:
- **Mixture Problem:** สเปกตรัมแต่ละ Sample คือ superposition ของ 15-30 สาร → กราฟซ้อนทับกันยุ่งเหยิง
- **Similarity Problem:** สาร 1,328 ตัว มีหลายตัวที่ Peak ตำแหน่งใกล้กัน → Rule-based ทายผิดเพียบ
- **Only DL can learn:** เฉพาะ Neural Network เท่านั้นที่จะเรียนรู้ "ฟังก์ชันการแยกสาร" ที่ซับซ้อนขนาดนี้ได้จากตัวอย่าง

---

## 🏗️ Architecture: 1D-CNN + Transformer Encoder

### ทำไมถึงเลือกสถาปัตยกรรมนี้?
1. **1D-CNN** ย่อสเปกตรัม 10,001 จุด → sequence สั้นลง (เช่น 312 tokens) → ลด memory และจับ local peak patterns
2. **Transformer Encoder** เรียนรู้ความสัมพันธ์ระหว่าง peaks ที่อยู่ห่างกัน (long-range dependencies)
3. **จากผลทดลองรอบก่อน (38 สาร):** CNN+Transformer ได้ F1=85.9% ดีที่สุดในกลุ่ม DL

### Model Architecture (Pseudocode)
```
Input: (batch_size, 10001) — raw NMR spectrum

1. Reshape → (batch_size, 1, 10001)   # 1 channel

2. CNN Feature Extractor:
   Conv1d(1→32, kernel=7, stride=2, pad=3)  → (batch, 32, 5001) → BatchNorm → ReLU
   Conv1d(32→64, kernel=5, stride=2, pad=2) → (batch, 64, 2501) → BatchNorm → ReLU
   Conv1d(64→128, kernel=5, stride=2, pad=2) → (batch, 128, 1251) → BatchNorm → ReLU
   Conv1d(128→256, kernel=3, stride=2, pad=1) → (batch, 256, 626) → BatchNorm → ReLU
   Conv1d(256→256, kernel=3, stride=2, pad=1) → (batch, 256, 313) → BatchNorm → ReLU
   
   Output: (batch_size, 313, 256) after transpose  # 313 tokens, 256 dims each

3. Positional Encoding:
   Learnable positional embeddings (313 positions, 256 dims)

4. Transformer Encoder:
   4 layers, 8 heads, dim_feedforward=512, dropout=0.1
   Output: (batch_size, 313, 256)

5. Global Average Pooling:
   mean(dim=1) → (batch_size, 256)

6. Classification Head:
   Linear(256 → 512) → ReLU → Dropout(0.3)
   Linear(512 → 1328) → Sigmoid   # Multi-label output

Output: (batch_size, 1328) — probability per metabolite
```

### Key Design Decisions
| ตัวเลือก | ค่าที่เลือก | เหตุผล |
|:---|:---|:---|
| Loss Function | `BCEWithLogitsLoss` (pos_weight adjusted) | Multi-label + class imbalance (15/1328 = 1.1% positive rate) |
| Threshold | 0.5 (ปรับภายหลัง) | จูนด้วย validation set หลังเทรนเสร็จ |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) | Standard for Transformer |
| Scheduler | OneCycleLR (max_lr=5e-4) | ช่วย converge เร็วขึ้น |
| Mixed Precision | `torch.amp.autocast('cuda')` + GradScaler | จำเป็นสำหรับ RTX 2050 4GB |
| Batch Size (Dev) | 8-16 | จำกัดด้วย VRAM 4GB |
| Batch Size (Prod/H100) | 256-512 | Scale เต็มที่บน 80GB |
| Epochs | 30-50 | ดู EarlyStopping |

### Class Imbalance Handling
ปัญหาสำคัญ: แต่ละ Sample มีสารเฉลี่ย 15 ตัว จากทั้งหมด 1,328 → **Positive rate = 1.1%**
ถ้าโมเดลทายว่า "ไม่มีสารเลย" ทุก Sample จะถูก 98.9% แต่ F1 = 0

**วิธีแก้:**
```python
# คำนวณ pos_weight จาก training data
pos_counts = y_train.sum(axis=0)  # จำนวน positive per class
neg_counts = len(y_train) - pos_counts
pos_weight = neg_counts / (pos_counts + 1)  # +1 to avoid div/0
pos_weight = torch.clamp(pos_weight, max=50.0)  # cap to prevent explosion

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## 📁 File Structure

```
d:\hack\BDI\Test_for_finalist\second_try\
├── EXP_F_deep_learning/
│   ├── model.py              # Model architecture (CNN + Transformer)
│   ├── dataset.py            # PyTorch Dataset class
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation & threshold tuning
│   ├── run_exp_f.py          # Main entry point
│   └── results/
│       └── metrics.json      # Final results
```

---

## 📝 Implementation Details (ทีละไฟล์)

### 1. `dataset.py` — NMR Dataset
```python
class NMRDataset(torch.utils.data.Dataset):
    """
    Loads spectra and labels from CSV files.
    Uses memory mapping or chunked loading because the data CSV is 800MB.
    """
    def __init__(self, spectra_matrix, labels_matrix):
        # spectra_matrix: np.array (N_samples, 10001) float32
        # labels_matrix: np.array (N_samples, 1328) int (0/1)
        self.spectra = torch.from_numpy(spectra_matrix).float()
        self.labels = torch.from_numpy(labels_matrix).float()
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        return self.spectra[idx], self.labels[idx]
```

**Data Split:**
- Train: 8,000 samples (80%)
- Validation: 1,000 samples (10%)
- Test: 1,000 samples (10%)
- Stratified split based on number of metabolites per sample

### 2. `model.py` — Architecture
ตาม Architecture section ด้านบน ประกอบด้วย:
- `CNNFeatureExtractor`: 5 layers Conv1d ลดขนาดจาก 10001 → 313 tokens
- `PositionalEncoding`: Learnable positional embeddings
- `NMRTransformer`: Transformer Encoder + Global Average Pooling + Classification Head

### 3. `train.py` — Training Loop
```python
# Pseudocode
for epoch in range(num_epochs):
    model.train()
    for batch_spectra, batch_labels in train_loader:
        with torch.amp.autocast('cuda'):
            logits = model(batch_spectra)
            loss = criterion(logits, batch_labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    
    # Validation
    model.eval()
    val_f1 = evaluate(model, val_loader, threshold=0.5)
    
    # Early stopping
    if val_f1 > best_f1:
        best_f1 = val_f1
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 5:
            break
```

**Key Training Features:**
- Mixed Precision (AMP) — จำเป็นสำหรับ VRAM 4GB
- Gradient Accumulation (accumulate 4 steps) — เพื่อ effective batch size 32-64
- EarlyStopping (patience=5) — ป้องกัน overfitting
- Save best model checkpoint

### 4. `evaluate.py` — Evaluation & Threshold Tuning
```python
# หลังเทรนเสร็จ ปรับ threshold เพื่อหา F1 สูงสุด
best_threshold = 0.5
best_f1 = 0
for t in np.arange(0.1, 0.9, 0.05):
    preds = (probs > t).astype(int)
    f1 = f1_score(y_true, preds, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t
```

### 5. `run_exp_f.py` — Main Entry Point
```
1. Load data (mock_nmr_data_10k.csv + mock_ground_truth_10k.csv)
2. Split into train/val/test
3. Create DataLoaders
4. Build model
5. Train with AMP + early stopping
6. Evaluate on test set
7. Tune threshold
8. Save metrics to results/metrics.json
```

---

## ⏱️ Estimated Timeline

| ขั้นตอน | เวลาที่คาด (RTX 2050) | เวลาที่คาด (H100) |
|:---|:---|:---|
| Load & preprocess data | 30 วินาที | 10 วินาที |
| Train 30 epochs (batch=8, AMP) | ~15-20 นาที | ~2 นาที |
| Evaluate + Threshold tuning | 30 วินาที | 5 วินาที |
| **รวม** | **~20 นาที** | **~3 นาที** |

---

## ⚠️ Risks & Mitigations

| ความเสี่ยง | ผลกระทบ | วิธีรับมือ |
|:---|:---|:---|
| VRAM ไม่พอ (RTX 2050 4GB) | Training crash | ใช้ AMP + batch_size=8 + gradient accumulation |
| Overfitting (10,000 samples for 1,328 classes) | F1 ดีตอน train แต่พังตอน test | Dropout(0.3), weight_decay, early stopping |
| Class imbalance (1.1% positive rate) | โมเดลทายว่า "ไม่มี" ทุกอย่าง | pos_weight ใน BCEWithLogitsLoss |
| Mock Data ไม่สมจริงพอ | โมเดลเรียนรู้ pattern ผิด | ถ้า F1 สูงบน Mock แต่ต่ำบนข้อมูลจริง → ต้องปรับ Mock Data Generator |

---

## 🎯 Success Criteria

| เกณฑ์ | ค่าที่ต้องการ |
|:---|:---|
| F1-Score (Macro) on Test Set | ≥ 0.70 (70%) |
| Inference Time per Sample | < 10 ms (H100) |
| VRAM Usage (Training, RTX 2050) | < 3.5 GB |
| Model Size | < 50 MB |

---

## 🔄 Fallback Plan

ถ้า F1-Score < 50% หลังเทรนเสร็จ:
1. **เพิ่ม Mock Data** เป็น 50,000-100,000 samples (รันบน H100 วันแข่ง)
2. **ลดจำนวน Output Classes** ให้เหลือเฉพาะสาร top-200 ที่พบบ่อยที่สุด
3. **ใช้ Ensemble:** ผสม FAISS (EXP-D) กับ DL predictions → weighted voting
4. **กลับไปใช้ EXP-D (FAISS)** เป็น Submission หลัก (F1=0.32 ดีกว่าส่งโมเดลที่พัง)

---

## ✅ Checklist

- [ ] สร้าง `EXP_F_deep_learning/dataset.py`
- [ ] สร้าง `EXP_F_deep_learning/model.py`
- [ ] สร้าง `EXP_F_deep_learning/train.py`
- [ ] สร้าง `EXP_F_deep_learning/evaluate.py`
- [ ] สร้าง `EXP_F_deep_learning/run_exp_f.py`
- [ ] รัน Training (30 epochs, AMP, batch=8)
- [ ] ประเมินผลบน Test Set
- [ ] Tune Threshold
- [ ] บันทึกผลใน `results/metrics.json`
- [ ] อัปเดต `results_comparison.md`
