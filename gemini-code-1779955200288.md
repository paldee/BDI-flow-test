# Architectural Review & Issue Tracker: 3-Stage Hybrid AI for NMR
**Status:** Requires Architectural Redesign
**Target:** Automated AI Pipeline for NMR Spectroscopy (Phenome Track)

เอกสารฉบับนี้สรุปข้อจำกัดและช่องโหว่ทางวิศวกรรม (Engineering Flaws) ของสถาปัตยกรรม 3-Stage Hybrid AI ปัจจุบัน (Encoder -> Neural ODE -> EBM) เพื่อให้ AI Agent หรือทีมพัฒนาทำการปรับปรุงแก้ไขโครงสร้างก่อนนำไปใช้งานจริง

---

## 🛑 1. Critical Flaw (จุดบอดร้ายแรงที่ต้องแก้ด่วนที่สุด)

### 1.1 การตรวจสอบกฎฟิสิกส์ใน Latent Space (Missing Decoder Layer)
* **ปัญหา:** ปัจจุบันระบบส่งเวกเตอร์แฝงขนาด 128 มิติ (Latent Space) จาก Stage 2 (ODE) ไปให้ Stage 3 (EBM) เพื่อตรวจสอบกฎฟิสิกส์เคมี เช่น "Spin-spin coupling" และ "Peak Distances"
* **ทำไมถึงทำงานไม่ได้จริง:** กฎฟิสิกส์เคมีของ NMR อ้างอิงกับพิกัดบนแกน `ppm` และรูปร่างความสูงของกราฟ (20,000 ฟีเจอร์) เวกเตอร์ 128 มิติเป็นเพียงตัวเลขสกัดนามธรรม (Abstract features) ที่สูญเสียรูปทรงของแกน `ppm` ไปแล้ว EBM จึงไม่สามารถอ่านระยะห่างของพีคเพื่อจับ Ghost Peak ได้
* **ผลกระทบต่อโจทย์ Hackathon:** ทำให้ไม่สามารถทำ Feature Selection เพื่อบอกแพทย์ได้ว่า สารบ่งชี้โรคอยู่พิกัด ppm ที่เท่าไหร่
* **Action Required:** > ต้องเพิ่ม **"Decoder Layer"** คั่นระหว่าง Stage 2 และ Stage 3 เพื่อแปลงข้อมูล (Reconstruct) จาก 128 มิติ กลับไปเป็นสเปกตรัม 20,000 มิติบนแกน `ppm` ก่อนส่งให้ EBM ประเมินพลังงาน

---

## ⚠️ 2. Structural & Performance Bottlenecks (ข้อจำกัดเชิงโครงสร้าง)

### 2.1 การใช้ Dense Neural Network ทำลาย Spatial Relationship
* **ปัญหา:** Stage 1 (NMR Feature Encoder) เลือกใช้ Dense Layer (Linear/MLP) ในการรับ Input 20,000 จุด
* **ทำไมถึงเป็นข้อจำกัด:** ข้อมูล NMR มีลักษณะคล้ายสัญญาณอนุกรมเวลา (Time-series / Sequence) ที่จุดติดกันมีความหมายเกี่ยวเนื่องกันเป็นรูปร่างยอดพีค (Peak Shape) โมเดล Dense จะมองแต่ละจุดแยกขาดจากกัน (Flatten) ทำให้โมเดลเรียนรู้ลายเซ็นของสารได้ยากและกินทรัพยากรพารามิเตอร์มหาศาล
* **Action Required:**
  > เปลี่ยนสถาปัตยกรรมใน Stage 1 เป็น **1D-CNN (Convolutional Neural Network)** หรือกลุ่ม **Time-Series Transformer** เพื่อสกัดฟีเจอร์รูปร่างของคลื่นสัญญาณแทน

### 2.2 อัตราการบีบอัดสูงเกินไป (Extreme Compression & Information Loss)
* **ปัญหา:** การบีบข้อมูลจาก 20,000 มิติ เหลือเพียง 128 มิติ (ลดลง 156 เท่า) ภายในขั้นตอนเดียว
* **ทำไมถึงเป็นข้อจำกัด:** สารประกอบที่เป็น Biomarker บางชนิดอาจมีความเข้มข้นต่ำมาก (Low abundance) การบีบอัดที่รุนแรงเกินไปจะทำให้โมเดลมองว่าพีคเล็กๆ เหล่านั้นเป็น Noise และตัดทิ้งไป ทำให้สูญเสียข้อมูลสำคัญทางการแพทย์
* **Action Required:**
  > ปรับโครงสร้าง Encoder ให้ลดมิติแบบค่อยเป็นค่อยไป (Hierarchical Downsampling) และพิจารณาขยายขนาด Latent Space (เช่น 256 หรือ 512 มิติ) 

### 2.3 ภาระการประมวลผลของ Neural ODE (High Computational Cost)
* **ปัญหา:** Stage 2 จำเป็นต้องรัน Euler Integration ถึง 4 ลูปย่อยในพื้นที่แฝงสำหรับทุกๆ Sample
* **ทำไมถึงเป็นข้อจำกัด:** หากนำไปใช้จริงในโรงพยาบาลที่มีปริมาณข้อมูลสูง จะทำให้เกิด Latency และใช้เวลา Inference นานกว่าโมเดลปกติหลายสิบเท่า
* **Action Required:**
  > ต้องเพิ่มเทคนิคเพื่อเร่งความเร็ว ODE Solver (เช่น การใช้ adjoint method แบบปรับแต่ง หรือกำหนด Tolerance threshold ที่ยืดหยุ่นขึ้น)

---

## 🔍 3. Missing Logic (ลอจิกที่ขาดหายไปใน Pipeline ปลายน้ำ)

### 3.1 อัลกอริทึมการจับคู่ฐานข้อมูลไม่ชัดเจน (Undefined Matching Algorithm)
* **ปัญหา:** ในโมดูล Automated Knowledge Discovery ระบุว่าจะนำกราฟไปเทียบกับฐานข้อมูล HMDB แต่ไม่ได้กำหนดกลไกทางคณิตศาสตร์ในการประเมิน
* **ทำไมถึงเป็นข้อจำกัด:** กรรมการจะตั้งคำถามว่า AI รู้ได้อย่างไรว่า "เหมือน" (ใช้ระยะทาง Euclidean, Cosine Similarity, Dynamic Time Warping (DTW) หรือ Classifier อีกตัว?)
* **Action Required:**
  > ระบุอัลกอริทึมที่จะใช้เทียบ "ลายเซ็น" สเปกตรัมที่ผ่านการ Clean แล้ว กับคลังข้อมูล HMDB ให้ชัดเจน (แนะนำให้ใช้ DTW หรือ Optimization Matching)

---

## 🛠️ Instructions for the Agent (คำสั่งสำหรับ AI ผู้แก้ไขโครงสร้าง)
1. **Redesign Stage 1:** Replace Dense blocks with 1D-CNN or PatchTST pre-trained architectures.
2. **Inject Decoder:** Introduce a robust 1D-CNN Decoder after the Neural ODE block to reconstruct the 20k features.
3. **Refactor Stage 3 (EBM):** Ensure the EBM verifies physics on the reconstructed 20k ppm space, not the latent space.
4. **Specify Matching Logic:** Define the exact algorithm used for querying and confidence scoring against the HMDB mock API.