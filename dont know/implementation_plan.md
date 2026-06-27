# NMR Hybrid Physics-Aware AI Pipeline — Flow Validation Test

ทดสอบว่า pipeline 4 stages ตาม [NMR_DEEP_HANDBOOK.md](file:///d:/hack/BDI/NMR_prototype/NMR_DEEP_HANDBOOK.md) สามารถ train และ inference ได้จริงบนข้อมูล simulated จาก NMRQNet

## User Review Required

> [!IMPORTANT]
> **ต้องติดตั้ง PyTorch CPU + torchdiffeq + dtw-python** — จะรัน `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` และ `pip install torchdiffeq dtw-python matplotlib`

> [!WARNING]
> **ข้อมูล CSV มี 40,000 features** — จะ downsample 40x เหลือ ~1,000 จุดและใช้ 15 samples เท่านั้น เพื่อความเร็วในการทดลอง

---

## Proposed Changes

### Environment Setup

#### pip install
- `torch` (CPU-only), `torchaudio`, `torchvision` via PyTorch CPU index
- `torchdiffeq` — สำหรับ Neural ODE solver
- `dtw-python` — สำหรับ Dynamic Time Warping
- `matplotlib` — สำหรับ visualization

---

### Data Pipeline

#### [NEW] Data Loading & Preprocessing (Cell 1-2 ใน notebook)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Samples | 15 จาก 100 | ลด compute time |
| Features | ~1,000 จาก 40,000 | Downsample 40x โดยใช้ `scipy.signal.decimate` หรือ simple step slicing |
| ppm range | -1 ถึง 11 ppm (ครอบคลุมทั้งหมด) | ตาม data documentation |
| Targets | 9 metabolites (Choline, Cysteine, Glucose, Glutamate, Glycine, Leucine, Lysine, Myo_inositol, Tryptophan) | ตาม ground_truth_concentrations.csv |

#### [NEW] Noise/Drift/Ghost Peak Injection (Cell 3)

ตาม handbook Chapter 2 — Sim2Real:
1. **Stochastic Shift Drift**: สุ่มเลื่อน spectrum ±5 จุด (≈ 0.06 ppm at downsampled resolution)
2. **Ghost Peak Injection**: เพิ่ม narrow Lorentzian peak ที่ ~4.15 ppm (carbohydrate zone)
3. **Additive Gaussian Noise**: σ = 0.01 × max_signal

---

### Stage 1 — SequenceAwareEncoder (Pre-trained 1D-ResNet)

#### [NEW] Model definition (Cell 4)

ใช้ **1D-ResNet Backbone** ที่ผ่านการ Pre-train บนข้อมูล ECG / Physiological Signals (จาก `hsd1503/resnet1d` ที่ผ่านการเทรนบนข้อมูล PhysioNet Challenge 2017):
- **Input**: `(batch, 1, 1000)` (ขนาดที่ downsample แล้ว สอดคล้องกับ input size ของ 1D-ResNet)
- **Freeze strategy**: Freeze เลเยอร์ Convolution แรกๆ ของ backbone (เพื่อใช้เป็น feature extractor ทั่วไปสำหรับสัญญาณ 1D)
- **Adaptation & Projection Head**: 
  - ตัด Classification head เดิมทิ้ง
  - เพิ่ม **Linear projection head** เพื่อแปลงสัญญาณออกมาเป็น latent embedding ขนาด **512 มิติ**
- **Output**: `(batch, 512)`

---

### Stage 2 — LatentSpaceODESolver (Neural ODE)

#### [NEW] ODE function + solver (Cell 5)

ตาม handbook Chapter 4:
```python
class ODEFunc(nn.Module):
    # 512 → 1024 → 512 MLP
    # เรียนรู้ vector field gradient dh/dt

class LatentSpaceODESolver:
    # Euler integration, 4 steps, dt=0.1
    # h(T) = h(0) + ∫ f_θ(h(t)) dt
```

- ใช้ `torchdiffeq.odeint` หรือ manual Euler 4-step (ตามที่ handbook ระบุ)
- Input: latent (batch, 512) จาก Encoder
- Output: aligned latent (batch, 512)

---

### Stage 3 — SpectrumDecoder + LocalizedPatchEBM

#### [NEW] Decoder + EBM (Cell 6-7)

**SpectrumDecoder** (ตาม handbook Chapter 5.1):
```
Input: (batch, 512)
  → Linear(512, 64*128) → Reshape(batch, 64, 128)
  → ConvTranspose1D(64→16, kernel=7, stride=2) → GELU
  → ConvTranspose1D(16→1, kernel=15, stride=2) → GELU
  → Linear adjustment → (batch, 1, 1000)
```

**LocalizedPatchEBM** (ตาม handbook Chapter 5.2-5.3):
- แบ่ง spectrum เป็น 3 chemical zones:
  1. Aliphatic (0.5-3.0 ppm) — weight 0.4
  2. Carbohydrate (3.0-5.5 ppm) — weight 0.4  
  3. Aromatic (5.5-9.0 ppm) — weight 0.2
- แต่ละ zone ผ่าน small MLP → scalar energy E(patch)
- E_global = 0.4 × E₁ + 0.4 × E₂ + 0.2 × E₃
- Ghost peak detection: E_global > threshold (1.1)

---

### Stage 4 — Constrained DTW + Hybrid Matching

#### [NEW] DTW + Hybrid Score (Cell 8)

ตาม handbook Chapter 6:
1. **Constrained DTW**: Sakoe-Chiba band (radius=10), distance = `|a - b|`
2. **Peak Bipartite Assignment**: scipy.optimize.linear_sum_assignment, tolerance ±0.03 ppm  
3. **Hybrid Score**: 
   ```
   Match Confidence = 0.45 × PeakAssignment + 0.35 × DTW_similarity + 0.20 × σ(-E_ebm)
   ```

- ใช้ reference library: สร้างจาก single-compound spectra (pure Lorentzian peaks ตามข้อมูลสาร 9 ชนิด)
- Output: ranked compound matches with confidence scores

---

### Pretraining & Fine-tuning Loop

#### [NEW] Phase 1 — Sim2Real Pretraining (Cell 9a)
- **Data**: ใช้ **1,000 synthetic samples** (สุ่ม Lorentzian peaks, noise, and drift) เพื่อจำลองความหลากหลายของ NMR spectra
- **Target**: เทรนเพื่อกู้คืนสัญญาณสะอาด (MSE reconstruction loss ระหว่าง output ของ Decoder กับ clean spectrum)
- **Frozen weights**: Freeze เลเยอร์ Convolution แรกๆ ของ ResNet1D backbone และเทรนเฉพาะ Linear projection head (และ Decoder)
- **Compute duration**: คาดว่าจะเสร็จภายใน ~5-10 นาที
- **Goal**: ช่วยให้ Latent space (512 มิติ) เรียนรู้โครงสร้างทางฟิสิกส์เบื้องต้นของ NMR spectra

#### [NEW] Phase 2 — Fine-tuning with Real Data & EBM (Cell 9b)
- **Data**: ข้อมูล NMRQNet จริง (ใช้ 15 samples เพื่อความรวดเร็วในการรัน flow)
- **Models**: Fine-tune โมเดลทั้งหมดร่วมกัน (Projection Head + ODE + Decoder + LocalizedPatchEBM)
- **Loss**: Loss = MSE(reconstructed_spectrum, clean_spectrum) + λ × EBM_contrastive_loss
- **EBM Training**:
  - Positive samples: clean spectra → low energy
  - Negative samples: noisy+ghost spectra → high energy
  - Contrastive margin loss

---

### Inference & Visualization

#### [NEW] Full pipeline inference + plots (Cell 10-12)

แต่ละ stage จะแสดง:
1. **Raw vs Noisy spectrum** comparison plot
2. **Encoder latent space** visualization (t-SNE or PCA of 512-dim vectors)
3. **Before/After Neural ODE alignment** overlay plot
4. **EBM energy heatmap** across chemical zones — แสดง ghost peak detection
5. **DTW alignment path** visualization
6. **Final Hybrid Score** table — ranked compound identification results
7. **Diagnostic JSON report** — ตาม clinical_report.json format

---

### Project Structure

```
d:\hack\BDI\nmr_flow_test\
├── nmr_pipeline_test.ipynb     ← Main Jupyter Notebook (all cells)
└── reference_peaks.json        ← Reference library สำหรับ 9 metabolites
```

---

## Verification Plan

### Automated Tests
1. รัน notebook ครบทุก cell รวมถึง Pretraining cells และ Fine-tuning cells โดยไม่มี error
2. ตรวจสอบการโหลด weight ของ Pre-trained ResNet1D และการ freeze เลเยอร์ (เช็ค `requires_grad == False` ในเลเยอร์แรกๆ)
3. ตรวจสอบ output shapes ทุก stage:
   - ResNet1D output + Linear projection: (batch, 512)
   - ODE: (batch, 512) 
   - Decoder: (batch, 1, 1000)
   - EBM: scalar energy per sample
4. Pretraining loss (MSE reconstruction) ลดลงเมื่อเทรนบน synthetic 1,000 samples
5. Fine-tuning loss ลดลง
6. Ghost peak ถูก detect (E_global > threshold สำหรับ injected ghost peaks)
7. DTW + Hybrid Score ให้ผลลัพธ์ที่ reasonable (compounds ที่มีอยู่จริงได้ score สูงกว่า)

### Manual Verification
- ดู visualization plots ทุก stage ใน notebook
- ตรวจสอบว่า flow ทั้งหมดสมเหตุสมผลตาม NMR_DEEP_HANDBOOK.md
