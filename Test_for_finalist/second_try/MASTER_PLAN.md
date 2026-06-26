# 🧪 MASTER PLAN: NMR Peak Annotation — Second Try

> **เอกสารนี้เขียนขึ้นเพื่อให้ Agent ตัวอื่นรับไปรันต่อได้โดยไม่ต้องคิดเอง**
> อ่านทั้งหมดก่อนเริ่มทำงาน

---

## 📌 บริบทโดยย่อ (Context)

### โจทย์การแข่ง
- รับ **Binning Table** (ตาราง NMR 1D ¹H Spectroscopy): แกน Y = ppm (0-10 ppm, ทุก 0.001 ppm = **10,001 จุด**), แกน X = Sample IDs, ค่าในช่อง = Intensity
- แต่ละ Sample คือ **สเปกตรัมสารผสม (mixture)** — **ไม่รู้** ว่ามีกี่สารปนกัน จำนวนสารต่อ Sample อาจไม่เท่ากัน
- **ไม่มี Label** บอกว่ามีสารอะไร — ต้องเชื่อมต่อไปดึงข้อมูลจาก **ฐานข้อมูลภายนอก (HMDB)** เอง
- จำนวน Sample อาจเป็น **หมื่น - แสน**
- วันจริงรันบน **H100 (80GB VRAM)** มีเวลา **1 วัน**
- **D-Day: 2 กรกฎาคม 2569** (อีก 6 วัน)

### สิ่งที่ผู้จัดเตรียมมาให้แล้ว
- ข้อมูลถูก **Clean** (ลด noise แล้ว) + **Align TMS** (จัดตำแหน่ง ppm แล้ว)
- ยังอาจมี **Residual chemical shift drift ±0.01-0.05 ppm** จากความแตกต่างของ pH หรือ Overlapping peaks 

### ผลลัพธ์ที่ต้องส่ง
- **รายชื่อสารเคมี (Metabolite Names)** ที่พบในแต่ละ Sample
- ถ้าระบุ **ความเข้มข้น (Quantification)** ได้ด้วย จะเป็น bonus

### ข้อจำกัดสำคัญ
> ⚠️ **ทุก Solution ต้องเป็น Unsupervised / Self-supervised**
> - ห้ามใช้ Classification ที่ต้องรู้จำนวนคลาสล่วงหน้า
> - ห้ามใช้ Label / Ground Truth ตอน Inference (ใช้ได้เฉพาะตอน Evaluation)

---

## 🖥️ สภาพแวดล้อม (Environment)

### Hardware
| สภาพแวดล้อม | GPU | VRAM | หมายเหตุ |
| :--- | :--- | :--- | :--- |
| **Dev (ปัจจุบัน)** | NVIDIA GeForce RTX 2050 | 4 GB | batch size ต้องเล็ก |
| **Prod (วันจริง)** | NVIDIA H100 | 80 GB | Scale เต็มที่ |

### Software
| Package | Version |
| :--- | :--- |
| Python | venv ที่ `d:\hack\BDI\.venv\Scripts\python.exe` |
| PyTorch | 2.7.1+cu118 |
| CUDA | Available (RTX 2050) |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |

### คำสั่งรัน Python
```bash
d:\hack\BDI\.venv\Scripts\python.exe <script.py>
```

### Package ที่อาจต้องติดตั้งเพิ่ม
```bash
d:\hack\BDI\.venv\Scripts\pip.exe install faiss-cpu dtaidistance requests lxml tqdm
```
> ถ้า FAISS ต้องการ GPU version ใช้ `faiss-gpu` แทน แต่ `faiss-cpu` ก็เร็วพอสำหรับ 500 สาร

---

## 📁 โครงสร้างโฟลเดอร์

```
d:\hack\BDI\Test_for_finalist\second_try\
├── MASTER_PLAN.md                          ← ไฟล์นี้
├── data/
│   ├── hmdb_reference/
│   │   ├── download_hmdb.py                ← [Phase 0A] ดึง HMDB data
│   │   ├── build_reference_spectra.py      ← [Phase 0A] แปลงเป็น spectra
│   │   └── hmdb_reference_library.csv      ← [OUTPUT] Reference Library
│   ├── generate_mock_data.py               ← [Phase 0B] สร้าง Mock Data
│   ├── mock_nmr_data_10k.csv               ← [OUTPUT] Mock Data 10,000 Samples
│   └── mock_ground_truth_10k.csv           ← [OUTPUT] Ground Truth
├── EXPA_nmf_decomposition/
│   ├── run_expA.py
│   └── results/
├── EXPB_cosine_hmdb/
│   ├── run_expB.py
│   └── results/
├── EXPC_dtw_matching/
│   ├── run_expC.py
│   └── results/
├── EXPD_autoencoder_faiss/
│   ├── run_expD.py
│   └── results/
├── EXPE_peak_picking/
│   ├── run_expE.py
│   └── results/
└── results_comparison.md                   ← [OUTPUT] ผลเปรียบเทียบทุก Experiment
```

---

## 📂 ข้อมูลอ้างอิงที่มีอยู่แล้ว (Existing Resources)

### Reference Library 38 สาร (จาก NMRQNet)
- **Path:** `d:\hack\BDI\NMRQNet\data\simulation_data\reference_library_38.csv`
- **Format:** CSV, columns = `"", "Name", "cluster", "loc", "height", "width"`
  - `Name`: ชื่อสาร (e.g., "Alanine", "Creatine")
  - `cluster`: ตำแหน่ง cluster center ของพีคกลุ่มนั้น (ppm)
  - `loc`: ตำแหน่งพีคจริง (ppm)
  - `height`: ความสูงของพีค
  - `width`: ความกว้างของพีค (ใช้กับ Lorentzian, ต้องหารด้วย scaler=450)
- **จำนวนสาร:** 38 ตัว (1,321 peak entries)
- **รายชื่อสาร:** Alanine, Creatine, Choline, 2-Aminobutyrate, 2-Hydroxybutyrate, 2-Oxoglutarate, 3-Hydroxybutyrate, Acetate, Acetoacetate, Acetone, Asparagine, Citrate, Creatinine, Dimethyl sulfone, Ethanol, Glycine, Glycerol, Galactose, Glutamate, Glutamine, Methionine, N,N-Dimethylglycine, lactate, succinate, Trimethylamine N-oxide, Threonine, Pyruvate, Valine, Histidine, Tyrosine, Leucine, Isoleucine, Glucose, Phenylalanine, Ornithine, Lysine, Proline, Sarcosine

### Lorentzian Function (ใช้สร้าง spectrum จาก peak list)
```python
def lorentzian(x, x0, a, w):
    """
    x: ppm axis (array)
    x0: peak location (ppm)
    a: peak height
    w: peak width (already divided by scaler)
    """
    return a / (1 + ((x - x0) / (w / 2))**2)
```

### การสร้าง Pure Spectrum จาก Reference Library
```python
ppm_values = np.round(np.arange(0.0, 10.001, 0.001), 3)  # 10,001 points
scaler = 450.0

for meta in metabolites:
    meta_peaks = df_lib[df_lib['Name'] == meta]
    spec = np.zeros_like(ppm_values)
    for _, row in meta_peaks.iterrows():
        W = row['width'] / scaler
        spec += lorentzian(ppm_values, row['loc'], row['height'], W)
    # Normalize area under curve to 1
    area = np.trapezoid(spec, ppm_values)
    if area > 0:
        spec = spec / area
    pure_spectra[meta] = spec
```

### Mock Data Generator ตัวเก่า (อ้างอิง)
- **Path:** `d:\hack\BDI\Test_for_finalist\First_try\data\generate_realistic_nmr_data.py`
- **ดูโค้ดเต็มเป็น reference** สำหรับ Phase 0B

### ผลการทดลองรอบแรก (ข้อมูลอ้างอิง ไม่ต้องทำซ้ำ)
| Experiment | F1-Score | หมายเหตุ |
| :--- | :--- | :--- |
| EXP01 Cosine Similarity | 91.1% | Baseline ดีที่สุด, 38 สาร |
| EXP05 NMF/ICA | 68.1% | Recall ต่ำ 62.6% |

---

## 🚀 ลำดับการดำเนินงาน

```
Phase 0A (HMDB Reference Library) → Phase 0B (Mock Data) → EXP-E → EXP-B → EXP-A → EXP-C → EXP-D → สรุปผล
```

**ทำทีละ Phase ตามลำดับ ห้ามข้าม**

---

# ═══════════════════════════════════════════
# Phase 0A: สร้าง HMDB Reference Library
# ═══════════════════════════════════════════

## เป้าหมาย
ดึงข้อมูล ¹H NMR peak positions จาก HMDB สำหรับสาร metabolite ที่พบใน biological samples (เลือด, ปัสสาวะ, เซรั่ม) แล้วแปลงเป็น Reference Spectra ในรูปแบบเดียวกับ `reference_library_38.csv`

## ไฟล์ที่ต้องสร้าง

### `data/hmdb_reference/download_hmdb.py`

**วิธีการ:**
1. HMDB มี REST API ที่ `https://hmdb.ca/metabolites/<HMDB_ID>.xml`
2. ดาวน์โหลด metabolite XML files ที่มี ¹H NMR spectra data
3. Parse XML เพื่อสกัด peak list (ppm position, intensity/height)

**Fallback strategy (ถ้า HMDB API ช้าหรือถูกบล็อก):**
- ใช้ HMDB bulk download files จาก `https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip` (~7GB)
- หรือดึงจาก HMDB NMR spectra JSON API: `https://hmdb.ca/metabolites/<HMDB_ID>/nmr_spectra`
- หรือเก็บ peak positions จาก HMDB spectra search page

**สิ่งที่ต้อง Extract จาก HMDB:**
- `metabolite_name`: ชื่อสาร
- `hmdb_id`: HMDB accession number (e.g., HMDB0000161)
- `peak_ppm`: ตำแหน่งพีค (ppm)
- `peak_height`: ความสูงของพีค (ถ้ามี ถ้าไม่มีให้ default = 1.0)
- `peak_width`: ความกว้างพีค (ถ้าไม่มีให้ default = 1.0 — จะใช้ scaler=450 เหมือน NMRQNet)

**เป้าหมายจำนวน:** 200-500 สาร (metabolites) ที่มี ¹H NMR data
**Filter:** เลือกเฉพาะ ¹H NMR (proton), ไม่เอา ¹³C NMR หรือ 2D NMR
**Filter เพิ่มเติม:** เน้นสารที่เจอใน biological samples (biofluid type = blood/urine/serum)

**Format ผลลัพธ์:** บันทึกเป็น CSV

### `data/hmdb_reference/build_reference_spectra.py`

**วิธีการ:**
1. อ่าน peak data ที่ได้จาก `download_hmdb.py`
2. สร้าง pure spectrum สำหรับแต่ละสารโดยใช้ Lorentzian function (เหมือน NMRQNet)
3. Normalize area under curve = 1

**Input:** HMDB peak data (CSV from step above)
**Output:** `hmdb_reference_library.csv`

**Format ของ `hmdb_reference_library.csv`:**
```csv
"","Name","cluster","loc","height","width"
"1","Glucose",3.24,3.24,1.0,1.0
"2","Glucose",3.24,3.25,0.8,1.0
...
```
> ใช้ format เดียวกับ `reference_library_38.csv` ทุกประการ เพื่อให้โค้ด Experiment ทุกตัวใช้ร่วมกันได้

**สำคัญ:** ต้องรวม 38 สารเดิมจาก NMRQNet เข้าไปด้วย (merge กัน) เพื่อไม่สูญเสียข้อมูลที่มีอยู่แล้ว

### Fallback Plan
ถ้า HMDB ดึงข้อมูลยาก ให้:
1. ใช้ `reference_library_38.csv` เดิมเป็น base
2. เพิ่มสารจาก HMDB เท่าที่ดึงได้
3. อย่างน้อยต้องมี 38 สาร (เท่าเดิม) ไม่ควรน้อยกว่า

---

# ═══════════════════════════════════════════
# Phase 0B: สร้าง Mock Data 10,000 Samples
# ═══════════════════════════════════════════

## เป้าหมาย
สร้างข้อมูลจำลองที่สมจริง 10,000 Samples โดยใช้ HMDB Reference Library (จาก Phase 0A)

## ไฟล์ที่ต้องสร้าง

### `data/generate_mock_data.py`

**Based on:** `d:\hack\BDI\Test_for_finalist\First_try\data\generate_realistic_nmr_data.py`

**สิ่งที่ต้องเปลี่ยนจากเดิม:**

| พารามิเตอร์ | ค่าเดิม | ค่าใหม่ |
| :--- | :--- | :--- |
| Reference Library path | `NMRQNet/.../reference_library_38.csv` | `data/hmdb_reference/hmdb_reference_library.csv` |
| num_samples | 1,000 | **10,000** |
| จำนวนสารต่อ Sample | `np.random.randint(10, 38+1)` สม่ำเสมอ | `np.clip(np.random.poisson(15), 3, 50)` |
| Noise | `np.random.normal(0, 0.01, ...)` | **คงไว้ `σ=0.01` หรือลดลง** (ข้อมูลจริงถูก clean แล้ว) |
| Shift error | `np.random.normal(0, 0.0005)` | **คงไว้** (จำลอง residual drift จาก pH) |

**สคริปต์ต้องทำงานตามขั้นตอนนี้:**
1. อ่าน `hmdb_reference_library.csv`
2. สร้าง pure spectrum สำหรับสารแต่ละตัว (Lorentzian, scaler=450, area normalized to 1)
3. วนลูป 10,000 samples:
   - สุ่มจำนวนสาร: `n = np.clip(np.random.poisson(15), 3, min(50, total_metabolites))`
   - สุ่มเลือกสาร `n` ตัวจากทั้งหมด (replace=False)
   - สุ่ม concentration ด้วย Gamma(shape=2.0, scale=1.0) สำหรับแต่ละสาร
   - เพิ่ม shift error: `np.random.normal(0, 0.0005)` ppm
   - รวมสเปกตรัมทุกสาร + noise เบาๆ (σ=0.01)
   - clip ให้ ≥ 0
4. บันทึก `mock_nmr_data_10k.csv` (ppm column + 10,000 sample columns)
5. บันทึก `mock_ground_truth_10k.csv` (Sample_ID + ทุกสารเป็น columns, ค่า = concentration หรือ 0)

**Output files:**
- `data/mock_nmr_data_10k.csv` — ขนาดประมาณ **650 MB**
- `data/mock_ground_truth_10k.csv` — ขนาดประมาณ **2-5 MB**

**คำสั่งรัน:**
```bash
d:\hack\BDI\.venv\Scripts\python.exe data/generate_mock_data.py --samples 10000 --output_dir data
```

**⏱ เวลาที่คาดการณ์:** 10-30 นาที (ขึ้นกับจำนวนสาร)

---

# ═══════════════════════════════════════════
# EXP-E: Peak Picking + Rule-Based Matching
# ═══════════════════════════════════════════

## ทำก่อนเพราะ: ง่ายที่สุด เร็วที่สุด ได้ baseline ทันที

## หลักการ
1. หาพีคในสเปกตรัมตัวอย่างด้วย `scipy.signal.find_peaks`
2. สร้าง "fingerprint" ของตำแหน่งพีค (ppm positions)
3. เทียบตำแหน่งพีคกับ known peak positions ของสารใน Reference Library
4. ให้คะแนน = จำนวนพีคของสารที่ match / จำนวนพีคทั้งหมดของสาร

## ไฟล์: `EXPE_peak_picking/run_expE.py`

### Pseudocode ละเอียด

```python
import numpy as np
import pandas as pd
import os, time
from scipy.signal import find_peaks
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def run_expE():
    DATA_DIR = r"d:\hack\BDI\Test_for_finalist\second_try\data"
    
    # 1. Load data
    df_data = pd.read_csv(os.path.join(DATA_DIR, "mock_nmr_data_10k.csv"))
    df_gt = pd.read_csv(os.path.join(DATA_DIR, "mock_ground_truth_10k.csv"))
    df_lib = pd.read_csv(os.path.join(DATA_DIR, "hmdb_reference", "hmdb_reference_library.csv"))
    
    ppm_values = df_data['ppm'].values
    samples = df_data.drop(columns=['ppm']).values.T  # (10000, 10001)
    
    metabolites = df_lib['Name'].unique()
    
    # 2. Build known peak positions for each metabolite
    #    Group by Name, get unique cluster positions (ppm)
    known_peaks = {}
    for meta in metabolites:
        meta_rows = df_lib[df_lib['Name'] == meta]
        # Use cluster column for main peak positions
        positions = meta_rows['cluster'].unique()
        known_peaks[meta] = positions
    
    # 3. For each sample, find peaks and match
    PEAK_TOLERANCE = 0.03  # ±0.03 ppm tolerance for matching
    
    # Build prediction matrix (num_samples x num_metabolites)
    scores_matrix = np.zeros((len(samples), len(metabolites)))
    
    for i, sample in enumerate(samples):
        # Find peaks in sample spectrum
        peaks_idx, properties = find_peaks(sample, 
                                            prominence=0.01,  # ปรับตามข้อมูล
                                            height=0.05,      # ปรับตามข้อมูล
                                            distance=5)       # ห่างกันอย่างน้อย 5 จุด (0.005 ppm)
        sample_peak_positions = ppm_values[peaks_idx]
        
        # Match against each metabolite
        for j, meta in enumerate(metabolites):
            ref_positions = known_peaks[meta]
            if len(ref_positions) == 0:
                continue
            
            matched = 0
            for ref_pos in ref_positions:
                # Check if any sample peak is within tolerance
                if np.any(np.abs(sample_peak_positions - ref_pos) <= PEAK_TOLERANCE):
                    matched += 1
            
            scores_matrix[i, j] = matched / len(ref_positions)
        
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1}/{len(samples)} samples...")
    
    # 4. Find best threshold
    y_true = (df_gt.set_index("Sample_ID")[metabolites].values > 0).astype(int)
    
    best_threshold, best_f1, best_preds = 0, 0, None
    for t in np.linspace(0.1, 0.95, 50):
        y_pred = (scores_matrix >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold, best_preds = f1, t, y_pred
    
    # 5. Report and save
    precision = precision_score(y_true, best_preds, average='macro', zero_division=0)
    recall = recall_score(y_true, best_preds, average='macro', zero_division=0)
    report = classification_report(y_true, best_preds, target_names=metabolites, zero_division=0)
    
    # Save to results/expE_report.txt (same format as previous experiments)
```

### พารามิเตอร์ที่ต้องปรับจูน
- `prominence`: ขั้นต่ำของความโดดเด่นของพีค (ลอง 0.005, 0.01, 0.05)
- `height`: ความสูงขั้นต่ำ (ลอง 0.01, 0.05, 0.1)
- `PEAK_TOLERANCE`: tolerance สำหรับ matching (ลอง 0.02, 0.03, 0.05)
- `threshold`: ค่าคะแนนขั้นต่ำที่จะถือว่ามีสาร (ใช้ grid search)

### ผลลัพธ์ที่ต้องบันทึก: `results/expE_report.txt`
```
EXP-E: Peak Picking + Rule-Based Matching
Best Threshold: X.XXXX
Peak Tolerance: ±0.XX ppm
Macro Precision: X.XXXX
Macro Recall: X.XXXX
Macro F1-Score: X.XXXX
Elapsed Time: X.XX seconds
Inference Time per Sample: X.XXXX seconds

Classification Report:
<per-metabolite report>
```

**⏱ เวลาที่คาดการณ์:** 1-2 ชั่วโมง
**GPU:** ไม่ใช้

---

# ═══════════════════════════════════════════
# EXP-B: Cosine Similarity + HMDB Full Library
# ═══════════════════════════════════════════

## หลักการ
เหมือน EXP01 เดิมทุกประการ แต่ scale ขึ้นจาก 38 สารเป็น HMDB full library (200-500+ สาร) + 10,000 Samples

## ไฟล์: `EXPB_cosine_hmdb/run_expB.py`

### Pseudocode ละเอียด

```python
def run_expB():
    # 1. Load data (เหมือน EXP01)
    # 2. Build reference spectra for ALL metabolites in HMDB library
    #    ใช้ Lorentzian function เดียวกัน (scaler=450, area normalized)
    # 3. Stack reference spectra: ref_matrix shape (N_metabolites, 10001)
    # 4. Normalize samples and references (L2 norm)
    # 5. Cosine similarity: sim_matrix = samples_normalized @ ref_normalized.T
    #    shape: (10000, N_metabolites)
    # 6. Grid search threshold (0.01 to 0.99, step 0.01)
    # 7. Evaluate against ground truth
```

**ข้อแตกต่างจาก EXP01 เดิม:**
- ใช้ `hmdb_reference_library.csv` แทน `reference_library_38.csv`
- ทุก path ต้องชี้ไปที่ `d:\hack\BDI\Test_for_finalist\second_try\data\`
- ข้อมูล 10,000 Samples → sim_matrix จะใหญ่ขึ้น → ใช้ vectorized numpy (ไม่ต้อง loop)

**โค้ดอ้างอิง:** `d:\hack\BDI\Test_for_finalist\First_try\EXP01_cosine_baseline\run_exp01.py` — copy มาแล้วแก้ path + ref library

**⏱ เวลาที่คาดการณ์:** 1-2 ชั่วโมง
**GPU:** ไม่ใช้

---

# ═══════════════════════════════════════════
# EXP-A: NMF/ICA Decomposition + HMDB Matching
# ═══════════════════════════════════════════

## หลักการ
1. ใช้ NMF (Non-negative Matrix Factorization) แยก Binning Table ออกเป็น Components
2. ต้องประมาณจำนวน Components (K) อัตโนมัติ — ห้ามกำหนดล่วงหน้าว่า K=38
3. Match แต่ละ Component กับ HMDB Reference ด้วย Cosine Similarity + Hungarian Algorithm

## ไฟล์: `EXPA_nmf_decomposition/run_expA.py`

### Pseudocode ละเอียด

```python
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

def estimate_num_components(data_matrix, k_range=range(5, 80, 5)):
    """ประมาณจำนวน components ที่เหมาะสม"""
    errors = []
    for k in k_range:
        nmf = NMF(n_components=k, init='nndsvda', random_state=42, max_iter=200)
        W = nmf.fit_transform(data_matrix)
        H = nmf.components_
        reconstruction_error = nmf.reconstruction_err_
        errors.append((k, reconstruction_error))
        print(f"K={k}, Error={reconstruction_error:.4f}")
    
    # หาจุด Elbow: จุดที่ error ลดลงน้อยลงมาก
    # ใช้วิธีหา max curvature หรือ % change < threshold
    # ...
    return best_k

def run_expA():
    # 1. Load data
    # 2. Estimate K (จำนวน components)
    best_k = estimate_num_components(samples_intensity)
    
    # 3. Run NMF with best K
    nmf = NMF(n_components=best_k, init='nndsvda', random_state=42, max_iter=500)
    W = nmf.fit_transform(samples_intensity)  # (10000, K)
    H = nmf.components_                        # (K, 10001)
    
    # 4. Build reference spectra from HMDB library
    # (same Lorentzian method)
    
    # 5. Match NMF components to HMDB references
    sim = cosine_similarity(H, ref_matrix)  # (K, N_metabolites)
    
    # NOTE: K ≠ N_metabolites, so can't use Hungarian directly
    # Instead: สำหรับแต่ละ component, หา reference ที่ sim สูงสุด
    # ถ้า sim > threshold → map component นั้นให้สาร
    # หลาย components อาจ map ไปที่สารเดียวกันได้ (รวม weights)
    
    component_to_metabolite = {}
    for comp_idx in range(best_k):
        best_ref_idx = np.argmax(sim[comp_idx])
        best_sim_score = sim[comp_idx, best_ref_idx]
        if best_sim_score > 0.5:  # minimum similarity threshold
            meta_name = metabolites[best_ref_idx]
            component_to_metabolite[comp_idx] = (meta_name, best_sim_score)
    
    # 6. Build prediction matrix
    # สำหรับแต่ละ sample: ถ้า W[sample, comp] > threshold 
    # และ comp ถูก map ไปที่สาร → ถือว่ามีสารนั้น
    
    # 7. Evaluate against ground truth (grid search threshold on W)
```

### Variants ที่ต้องลอง
1. **Standard NMF** — `NMF(init='nndsvda')`
2. **Sparse NMF** — `NMF(alpha_W=0.1, l1_ratio=1.0)` (sparseness on W)
3. **ICA** — `from sklearn.decomposition import FastICA` (alternative decomposition)
4. เก็บผลทั้ง 3 variants แล้วเปรียบเทียบ

### ข้อสำคัญ
- `samples_intensity` ต้อง transpose ให้เป็น **(10000, 10001)** ก่อนใส่ NMF
- NMF ต้องการค่า ≥ 0 ทั้งหมด → clip ก่อน
- NMF กับ 10,000 samples × 10,001 features จะใช้ RAM เยอะ (~800 MB ขึ้นไป) → ถ้า RAM ไม่พอให้ลด samples หรือใช้ MiniBatchNMF

**⏱ เวลาที่คาดการณ์:** 2-3 ชั่วโมง (NMF ช้ากับข้อมูลใหญ่)
**GPU:** ไม่ใช้ (sklearn NMF ใช้ CPU)

---

# ═══════════════════════════════════════════
# EXP-C: DTW + Hungarian Peak Matching
# ═══════════════════════════════════════════

## หลักการ
1. หาพีคในสเปกตรัมตัวอย่าง (เหมือน EXP-E)
2. ใช้ Dynamic Time Warping (DTW) จับคู่กับ reference spectrum ของแต่ละสาร
3. ใช้ Hungarian Algorithm จับคู่พีค 1-to-1 (tolerance ±0.03 ppm)
4. คำนวณ Hybrid Score = weighted combination

## ไฟล์: `EXPC_dtw_matching/run_expC.py`

### Pseudocode ละเอียด

```python
from dtaidistance import dtw  # pip install dtaidistance
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

def compute_dtw_score(sample_spectrum, ref_spectrum, window=150):
    """DTW with Sakoe-Chiba constraint"""
    # ใช้ window constraint เพื่อจำกัด path (เร็วขึ้น)
    distance = dtw.distance(sample_spectrum.astype(np.double), 
                            ref_spectrum.astype(np.double),
                            window=window)
    # Normalize by length
    return distance / len(sample_spectrum)

def compute_peak_match_score(sample_peaks_ppm, ref_peaks_ppm, tolerance=0.03):
    """Hungarian matching of peaks"""
    if len(sample_peaks_ppm) == 0 or len(ref_peaks_ppm) == 0:
        return 0.0
    
    # Build cost matrix: |sample_peak_i - ref_peak_j|
    cost = np.abs(sample_peaks_ppm[:, None] - ref_peaks_ppm[None, :])
    
    # Run Hungarian
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Count matches within tolerance
    matched = sum(1 for r, c in zip(row_ind, col_ind) if cost[r, c] <= tolerance)
    return matched / len(ref_peaks_ppm)

def compute_hybrid_score(dtw_score, peak_score, w_dtw=0.4, w_peak=0.6):
    """Combine DTW and peak matching scores"""
    # DTW: lower is better → convert to similarity
    dtw_sim = 1.0 / (1.0 + dtw_score)
    return w_dtw * dtw_sim + w_peak * peak_score

def run_expC():
    # 1. Load data
    # 2. Build reference spectra + extract reference peak positions
    # 3. For each sample:
    #    a. Find peaks
    #    b. For each reference metabolite:
    #       - DTW score
    #       - Peak match score
    #       - Hybrid score
    # 4. Threshold on hybrid score → prediction
    # 5. Evaluate
```

### ⚠️ Performance Warning
- DTW เป็น O(N²) → กับสเปกตรัม 10,001 จุด จะ **ช้ามาก**
- **วิธีแก้:** ไม่ DTW ทั้งเส้น แต่ DTW เฉพาะ **บริเวณที่มีพีค** (crop region ±0.5 ppm รอบ peak cluster)
- หรือ downsample สเปกตรัมก่อน (เช่น ทุก 10 จุด → 1,001 จุด) สำหรับ DTW
- ถ้ายังช้า ให้ใช้แค่ peak matching score (skip DTW)

**⏱ เวลาที่คาดการณ์:** 3-4 ชั่วโมง (ขึ้นกับ optimization)
**GPU:** ไม่ใช้

---

# ═══════════════════════════════════════════
# EXP-D: Autoencoder Embedding + FAISS Vector Search
# ═══════════════════════════════════════════

## หลักการ
1. เทรน 1D Convolutional Autoencoder (self-supervised: reconstruct input = input) กับ Mock Data
2. ใช้ Encoder ส่วนเดียว สกัด Latent Vector (Embedding) ของทุก Sample + ทุก Reference
3. สร้าง FAISS Index จาก Reference Embeddings
4. ค้นหา nearest neighbors สำหรับแต่ละ Sample

## ไฟล์: `EXPD_autoencoder_faiss/run_expD.py`

### Architecture

```python
import torch
import torch.nn as nn

class NMRAutoencoder(nn.Module):
    def __init__(self, input_size=10001, latent_dim=256):
        super().__init__()
        
        # Encoder: 10001 → 256
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),   # → 5001
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # → 2501
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # → 1251
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),# → 626
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                                 # → 256x1
            nn.Flatten(),                                            # → 256
        )
        
        # Decoder: 256 → 10001
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 626 * 256),
            nn.ReLU(),
            nn.Unflatten(1, (256, 626)),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1), # → 1252
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # → 2504
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # → 5008
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),    # → 10016
            nn.ReLU(),
        )
        # Note: output อาจไม่ตรง 10001 จุด → ใช้ interpolation หรือ crop
    
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z)
```

### Training Loop

```python
def train_autoencoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # GPU-specific batch size
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size = 16 if vram_gb < 6 else 64 if vram_gb < 16 else 256
    else:
        batch_size = 8
    
    model = NMRAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # DataLoader: samples shape (10000, 10001) → Dataset
    dataset = NMRDataset(samples_intensity)  # Custom Dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=2, pin_memory=True)
    
    EPOCHS = 50
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            batch = batch.unsqueeze(1).float().to(device)  # (B, 1, 10001)
            
            output = model(batch)
            # Crop/pad output to match input size
            output = output[:, :, :10001]
            
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.6f}")
    
    return model
```

### FAISS Search

```python
import faiss  # pip install faiss-cpu

def build_faiss_index(model, ref_spectra, device):
    """สร้าง FAISS index จาก reference spectra"""
    model.eval()
    with torch.no_grad():
        ref_tensor = torch.tensor(ref_spectra).unsqueeze(1).float().to(device)
        ref_embeddings = model.encode(ref_tensor).cpu().numpy()
    
    # Normalize embeddings
    faiss.normalize_L2(ref_embeddings)
    
    # Build index
    index = faiss.IndexFlatIP(ref_embeddings.shape[1])  # Inner Product = Cosine Sim
    index.add(ref_embeddings)
    
    return index, ref_embeddings

def search_metabolites(model, samples, index, metabolite_names, device, top_k=10):
    """ค้นหาสารที่ใกล้เคียงที่สุดสำหรับแต่ละ sample"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(0, len(samples), 32):
            batch = torch.tensor(samples[i:i+32]).unsqueeze(1).float().to(device)
            embeddings = model.encode(batch).cpu().numpy()
            faiss.normalize_L2(embeddings)
            
            scores, indices = index.search(embeddings, top_k)
            
            for j in range(len(embeddings)):
                sample_results = []
                for k in range(top_k):
                    sample_results.append({
                        'metabolite': metabolite_names[indices[j, k]],
                        'score': float(scores[j, k])
                    })
                results.append(sample_results)
    
    return results
```

### ข้อสำคัญ
- **RTX 2050 (4GB):** batch_size = 16, ต้องระวัง OOM
- **H100 (80GB):** batch_size = 256-512
- **Self-supervised:** ไม่ใช้ Label ตอนเทรน → เทรนด้วย reconstruction loss เท่านั้น
- **Decoder architecture** อาจต้องปรับ output size ให้ตรง 10,001 จุดพอดี (ใช้ `F.interpolate` หรือ padding)

**⏱ เวลาที่คาดการณ์:** 4-6 ชั่วโมง (รวมเทรน ~50 epochs)
**GPU:** ✅ ใช้ RTX 2050

---

# ═══════════════════════════════════════════
# สรุปผลลัพธ์ (Results Comparison)
# ═══════════════════════════════════════════

## ไฟล์: `results_comparison.md`

หลังจากรันทุก Experiment เสร็จ ให้สร้างไฟล์สรุปผลเปรียบเทียบ:

```markdown
# สรุปผลการทดลอง Second Try

## ตารางเปรียบเทียบ

| Experiment | Model | Precision | Recall | F1-Score | Inference Time/Sample | Total Time | หมายเหตุ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| EXP-E | Peak Picking | X.XXXX | X.XXXX | X.XXXX | X.XXXXs | XXXs | ... |
| EXP-B | Cosine + HMDB | X.XXXX | X.XXXX | X.XXXX | X.XXXXs | XXXs | ... |
| EXP-A | NMF/ICA | X.XXXX | X.XXXX | X.XXXX | X.XXXXs | XXXs | ... |
| EXP-C | DTW + Hungarian | X.XXXX | X.XXXX | X.XXXX | X.XXXXs | XXXs | ... |
| EXP-D | Autoencoder + FAISS | X.XXXX | X.XXXX | X.XXXX | X.XXXXs | XXXs | ... |

## วิเคราะห์
<เขียนวิเคราะห์เปรียบเทียบ>

## คำแนะนำสำหรับวันจริง
<เลือก Pipeline ที่ดีที่สุด>
```

---

# ═══════════════════════════════════════════
# Checklist สำหรับ Agent
# ═══════════════════════════════════════════

- [ ] **Phase 0A:** สร้าง HMDB Reference Library
  - [ ] สร้าง `data/hmdb_reference/download_hmdb.py`
  - [ ] รัน download
  - [ ] สร้าง `data/hmdb_reference/build_reference_spectra.py`
  - [ ] รัน build → ได้ `hmdb_reference_library.csv`
  - [ ] ตรวจสอบว่ามี ≥ 38 สาร + format ถูกต้อง
- [ ] **Phase 0B:** สร้าง Mock Data 10,000 Samples
  - [x] สร้าง `data/generate_mock_data.py`
  - [x] รัน generate → ได้ `mock_nmr_data_10k.csv` + `mock_ground_truth_10k.csv`
  - [x] ตรวจสอบ shape: (10001, 10001) สำหรับ data, (10000, N+1) สำหรับ GT
- [x] **EXP-E:** Peak Picking + Rule-Based
  - [x] สร้าง `EXPE_peak_picking/run_expE.py`
  - [x] รัน → บันทึกผลใน `results/expE_report.txt`
- [x] **EXP-B:** Cosine Similarity + HMDB
  - [x] สร้าง `EXPB_cosine_hmdb/run_expB.py`
  - [x] รัน → บันทึกผลใน `results/expB_report.txt`
- [x] **EXP-A:** NMF/ICA Decomposition
  - [x] สร้าง `EXPA_nmf_decomposition/run_expA.py`
  - [x] รัน → บันทึกผลใน `results/expA_report.txt`
- [x] **EXP-C:** DTW + Hungarian
  - [x] สร้าง `EXPC_dtw_matching/run_expC.py`
  - [x] รัน → บันทึกผลใน `results/expC_report.txt`
- [x] **EXP-D:** Autoencoder + FAISS
  - [x] ติดตั้ง `faiss-cpu` (หรือ `faiss-gpu`)
  - [x] สร้าง `EXPD_autoencoder_faiss/run_expD.py`
  - [x] รัน → บันทึกผลใน `results/expD_report.txt`
- [ ] **สรุปผล:** สร้าง `results_comparison.md`
