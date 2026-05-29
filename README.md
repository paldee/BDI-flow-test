# NMR Spectroscopy Hybrid Physics-Aware AI Pipeline

โปรเจกต์นี้สาธิตและตรวจสอบระบบการประมวลผลสเปกตรัม NMR แบบไฮบริด (Flow Validation) ตามรายละเอียดในคู่มือ `NMR_DEEP_HANDBOOK.md` โดยรันสำเร็จ 100% บน Python Environment

---

## 🏛️ โครงสร้างไพป์ไลน์ 4 ขั้นตอน (Pipeline Stages)
1. **Stage 1**: **SequenceAwareEncoder** นำแบบจำลอง 1D-ResNet ที่ผ่านการ Pre-train กับข้อมูล ECG (PhysioNet) มาทำ Transfer Learning เพื่อสกัดลักษณะเด่นของสัญญาณ 1 มิติ และแปลงลงสู่ Latent Vector ขนาด 512 มิติ
2. **Stage 2**: **LatentSpaceODESolver** ใช้สมการเชิงอนุพันธ์ต่อเนื่อง (Neural ODE) เพื่อบีบปรับแนวแกนและชดเชยค่าความเบี่ยงเบนยอดคลื่น (Chemical Shift Drift)
3. **Stage 3**: **SpectrumDecoder + LocalizedPatchEBM** กู้คืนสัญญาณกลับพิกัด ppm และตรวจจับพีคหลอกสิ่งปนเปื้อน (Ghost Peak Detection) ด้วย Energy-Based Model แบบแยกโซนเคมี
4. **Stage 4**: **Constrained DTW & Bipartite Peak Assignment** วิเคราะห์คะแนนจับคู่ชนิดสารสัมพัทธ์ในรูปแบบคะแนนความเชื่อมั่นผสมแบบไฮบริด (Hybrid Score)

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
└── resnet1d/
    ├── trained_model/
    │   └── model.pth            # ไฟล์น้ำหนักปรีเทรน 1D-ResNet (123 MB)
    └── net1d.py                 # โค้ดสถาปัตยกรรม 1D-ResNet
```
