# NMR Spectroscopy Hybrid Physics-Aware AI Pipeline

โปรเจกต์นี้สาธิตและตรวจสอบระบบการประมวลผลสเปกตรัม NMR แบบไฮบริด (Flow Validation) ตามรายละเอียดในคู่มือ `NMR_DEEP_HANDBOOK.md` โดยรันสำเร็จ 100% บน Python Environment

---

## 🏛️ โครงสร้างไพป์ไลน์ 4 ขั้นตอน (Pipeline Stages)
1. **Stage 1**: **SequenceAwareEncoder** นำแบบจำลอง 1D-ResNet ที่ผ่านการ Pre-train กับข้อมูล ECG (PhysioNet) มาทำ Transfer Learning เพื่อสกัดลักษณะเด่นของสัญญาณ 1 มิติ และแปลงลงสู่ Latent Vector ขนาด 512 มิติ
2. **Stage 2**: **LatentSpaceODESolver** ใช้สมการเชิงอนุพันธ์ต่อเนื่อง (Neural ODE) เพื่อบีบปรับแนวแกนและชดเชยค่าความเบี่ยงเบนยอดคลื่น (Chemical Shift Drift)
3. **Stage 3**: **SpectrumDecoder + LocalizedPatchEBM** กู้คืนสัญญาณกลับพิกัด ppm และตรวจจับพีคหลอกสิ่งปนเปื้อน (Ghost Peak Detection) ด้วย Energy-Based Model แบบแยกโซนเคมี
4. **Stage 4**: **Constrained DTW & Bipartite Peak Assignment** วิเคราะห์คะแนนจับคู่ชนิดสารสัมพัทธ์ในรูปแบบคะแนนความเชื่อมั่นผสมแบบไฮบริด (Hybrid Score)

---

## 📊 ข้อมูลที่ใช้งาน (Datasets Used)

ไพป์ไลน์นี้พัฒนาขึ้นโดยอิงจากโครงสร้างข้อมูลสเปกตรัมชีวภาพและแบบจำลองปรีเทรนดังนี้:

1. **ชุดข้อมูล NMRQNet Simulated Mixture Dataset**:
   - **ที่มา**: สร้างขึ้นผ่านเฟรมเวิร์ก **NMRQNet** (LiuzLab) เพื่อจำลองโครงสร้างสารเมแทบอไลต์ผสมที่มีพิกัดแกน ppm สมบูรณ์
   - **`NMRQNet/simulated_mixtures_highres.csv`**: ข้อมูลสเปกตรัมนำเข้า (Features) ประกอบด้วย 100 ตัวอย่าง โดยแต่ละตัวอย่างมีความละเอียดข้อมูลถึง **40,000 มิติ** (แกน ppm ตั้งแต่ $-1$ ถึง $11\text{ ppm}$)
   - **`NMRQNet/ground_truth_concentrations.csv`**: ตารางปริมาณความเข้มข้นจริง (Labels) ของสารเมแทบอไลต์ 9 ชนิดที่เป็นสารบ่งชี้โรคทางคลินิก (Clinical Biomarkers)
2. **น้ำหนักปรีเทรน 1D-ResNet**:
   - **`resnet1d/trained_model/model.pth`**: โมเดลโครงข่าย ResNet แบบ 1 มิติ ที่ผ่านการฝึกฝนบนข้อมูลสัญญาณคลื่นหัวใจ/สรีรวิทยาขนาดใหญ่จากคลังข้อมูล **PhysioNet Challenge 2017** นำมาใช้ชดเชยการขาดแคลนตัวอย่าง (Sim2Real Transfer Learning)
3. **คลังพิกัดสารบริสุทธิ์อ้างอิง**:
   - **`nmr_flow_test/reference_peaks.json`**: บรรจุพิกัดตำแหน่งพีคทางฟิสิกส์เคมี (เช่น ค่า ppm, linewidth, multiplicity) ของสารเมแทบอไลต์เป้าหมายทั้ง 9 ชนิดดึงมาจาก HMDB

---

## ⚙️ ขั้นตอนการเตรียมเครื่องและการติดตั้ง (Installation & Setup)

ทำตามขั้นตอนด้านล่างนี้เพื่อสร้าง Virtual Environment และติดตั้งไลบรารีทั้งหมด:

### 1. โคลน Repository และจัดเตรียมโมเดลโมดูล
```bash
# ตรวจสอบโครงสร้างไฟล์หลัก
# ไฮไลต์หลักจะอยู่ในโฟลเดอร์ nmr_flow_test/ และ resnet1d/
```

### 2. สร้างสภาพแวดล้อมจำลอง (Python Virtual Environment)
แนะนำให้สร้าง Virtual Environment แยกเพื่อป้องกันความขัดแย้งของแพ็กเกจ:
```bash
# สร้างโฟลเดอร์ .venv สำหรับ Virtual Environment
python -m venv .venv

# เปิดใช้งาน (Activate) Virtual Environment
# สำหรับ Windows (PowerShell):
.venv\Scripts\Activate.ps1
# สำหรับ Windows (CMD):
.venv\Scripts\activate.bat
# สำหรับ macOS/Linux:
source .venv/bin/activate
```

### 3. ติดตั้ง Dependencies ทั้งหมด
รันการติดตั้งผ่านไฟล์ `requirements.txt` ที่เตรียมไว้ (ไฟล์นี้ตั้งค่าดึง PyTorch CPU-only เพื่อความรวดเร็วและประหยัดพื้นที่เก็บข้อมูล):
```bash
pip install -r requirements.txt
```

---

## 📈 วิธีรันโปรแกรมตรวจสอบผล (Execution & Verification)

### รันผ่าน Jupyter Notebook (แนะนำ)
1. เปิดโปรแกรม VS Code หรือ Jupyter Lab
2. เลือกเปิดไฟล์ `nmr_flow_test/nmr_pipeline_test.ipynb`
3. ตรงปุ่มเลือก **Kernel** (มุมขวาบน) ให้เลือก Python Interpreter จากโฟลเดอร์ Virtual Environment ที่เราเพิ่งเปิดใช้งาน (`.venv`)
4. กด **Run All** หรือคลิกรันทีละ Cell เพื่อดูผลลัพธ์กราฟและการวิเคราะห์
5. ผลลัพธ์สรุปรายงานทางการแพทย์จะถูกส่งออกอัตโนมัติมาที่ไฟล์ `nmr_flow_test/clinical_report.json`

### รันด้วยโค้ดตรวจสอบผ่าน Terminal
หากต้องการทดสอบความถูกต้องอย่างรวดเร็วโดยไม่ใช้ UI จูปิเตอร์ สามารถนำโค้ดในสคริปต์ตรวจสอบมาทดลองรันได้ดังนี้:
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

---

## 📁 รายละเอียดโครงสร้างโฟลเดอร์
```
d:\hack\BDI\
├── .gitignore                   # กำหนดไฟล์ที่ไม่ต้องนำขึ้น Git (venv, temp files, caches)
├── README.md                    # คู่มือแนะนำการรันโปรเจกต์นี้
├── requirements.txt             # ระบุเวอร์ชันและแหล่งดาวน์โหลดไลบรารีท่อรัน
├── nmr_flow_test/
│   ├── nmr_pipeline_test.ipynb  # Jupyter Notebook ที่รัน Pipeline ทั้งหมด
│   ├── reference_peaks.json     # ฐานข้อมูลพีคอ้างอิงสารเมแทบอไลต์ 9 ชนิด
│   └── NMR_PIPELINE_REPORT.md   # รายงานผลสรุปการรันและค่าคะแนน Hybrid Score
├── resnet1d/
│   ├── trained_model/
│   │   └── model.pth            # ไฟล์น้ำหนักปรีเทรน 1D-ResNet (123 MB)
│   └── net1d.py                 # โค้ดสถาปัตยกรรม 1D-ResNet
└── NMRQNet/
    ├── simulated_mixtures_highres.csv     # ข้อมูลนำเข้า NMR Spectrum
    └── ground_truth_concentrations.csv   # ข้อมูลความเข้มข้นสารอ้างอิงจริง
```
