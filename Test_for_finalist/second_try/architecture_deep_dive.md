# 🧬 Deep Dive Architecture: Hybrid FAISS + NNLS (EXP-G)
> **เป้าหมายของเอกสารนี้:** เพื่อใช้อธิบายสถาปัตยกรรมและกระบวนการทำงานเชิงลึก (Technical Pipeline) ให้กับ AI Agent ตัวอื่นๆ หรือนักพัฒนาในทีมเข้าใจตรรกะเบื้องหลังโมเดลที่สามารถทำคะแนน F1-Score ได้ถึง **82.06%** บนโจทย์ NMR Metabolite Annotation

---

## 1. ฐานคิดทางฟิสิกส์ (Core Physics Assumption)
สถาปัตยกรรมนี้ถูกออกแบบมาแบบ **Physics-Informed** โดยไม่ได้ใช้ Black-box Deep Learning แต่ใช้สมการคณิตศาสตร์ที่ล้อตามกฎฟิสิกส์ของการเกิดสัญญาณ NMR:
* **สมมติฐาน:** สัญญาณ NMR ของสารผสม (Mixture) คือ **"ผลรวมเชิงเส้น (Linear Superposition)"** ของสเปกตรัมสารประกอบเดี่ยว (Pure reference spectra)
* **สมการหลัก:** $X \approx \Phi W + \epsilon$
  * $X$ = สัญญาณตัวอย่างจริง (10,000 มิติ)
  * $\Phi$ = เมทริกซ์อ้างอิงของสารที่ต้องสงสัย (Reference Spectra)
  * $W$ = น้ำหนักความเข้มข้นของสาร (Concentration) ซึ่ง **ต้องมีค่ามากกว่าหรือเท่ากับศูนย์เสมอ ($W \ge 0$)**
  * $\epsilon$ = Gaussian Noise และคลื่นรบกวน

---

## 2. โครงสร้าง Pipeline 3 ขั้นตอนหลัก (The 3-Stage Pipeline)

### 🔵 Stage 1: FAISS Multi-Window Search (High-Recall Candidate Filtering)
การนำสารทั้ง 1,328 ชนิดไปแก้สมการพร้อมกันจะทำให้เกิดปัญหา Overdetermined (ตัวแปรเยอะเกินไปจนแก้สมการไม่ได้) ด่านแรกจึงทำหน้าที่ **"ร่อนตะแกรงหยาบ"** เพื่อตัดสารที่เป็นไปไม่ได้ทิ้ง

1. **Sliding Window:** ซอยสเปกตรัม (0-10 ppm) ออกเป็นหน้าต่างย่อยๆ ขนาด `0.2 ppm` (เลื่อนทีละ `0.1 ppm` แบบ Overlap)
2. **Dimensionality Reduction:** ใช้ `TruncatedSVD` บีบอัดความซับซ้อนของแต่ละหน้าต่าง จากหลายร้อยจุดให้เหลือแค่ **64 มิติ (Latent Vector)** เพื่อดึงเฉพาะรูปร่างคลื่นหลัก
3. **Vector Similarity Search (FAISS):** นำ Latent Vector ไปค้นหาความคล้ายคลึงแบบ Inner Product (Cosine Similarity) เทียบกับฐานข้อมูลสาร 1,328 ชนิด ดึงตัวที่คล้ายที่สุด `Top-5` ของแต่ละหน้าต่างออกมา
4. **Voting System:** สารใดก็ตามที่โผล่มาติด Top-5 ในหน้าต่างใดหน้าต่างหนึ่ง จะได้คะแนนโหวต หากสารไหนได้โหวต $\ge 1$ จะถูกคัดเลือกเข้าสู่รอบถัดไป (Candidates)
> **💡 ผลลัพธ์ Stage 1:** ดึงผู้ต้องสงสัยจาก 1,328 ชนิด เหลือเพียงประมาณ 30-50 ชนิดต่อ Sample (ช่วยลดภาระการคำนวณใน Stage 2 ได้มหาศาล)

### 🔴 Stage 2: Non-negative Least Squares (Physics-Aware Deconvolution)
ด่านที่สองคือการนำ "ผู้ต้องสงสัย (Candidates)" จากด่านแรก มาแก้สมการเพื่อหา "ปริมาณ (Weight)" ของสารแต่ละตัว

1. **Sub-Matrix Extraction:** ดึงเส้นกราฟมาตรฐาน (Reference Spectra) เฉพาะของสารที่เป็น Candidate ออกมาสร้างเป็นเมทริกซ์ $\Phi_{sub}$
2. **NNLS Optimization:** ใช้ฟังก์ชัน `scipy.optimize.nnls` แก้สมการ $W_{optimal} = \text{argmin}_{W \ge 0} || \Phi_{sub} W - X ||^2_2$
3. **ทำไมต้อง NNLS?:** เพราะในโลกความเป็นจริง ความเข้มข้นของสารเคมีติดลบไม่ได้ NNLS จะบังคับไม่ให้น้ำหนักติดลบ ซึ่งสอดคล้องกับหลักเคมีฟิสิกส์เป๊ะ และใช้เวลาคำนวณรวดเร็วมาก
> **💡 ผลลัพธ์ Stage 2:** เราจะได้ค่าน้ำหนักต่อเนื่อง (Continuous weight) ของสาร Candidate แต่ละตัว หากสารไหนไม่มีอยู่จริง NNLS มักจะกดน้ำหนักให้เป็น 0 โดยอัตโนมัติ

### 🟢 Stage 3: Adaptive Thresholding (Noise Pruning)
แม้ NNLS จะเก่ง แต่เนื่องจากข้อมูลตัวอย่างมี Gaussian Noise ปะปนอยู่ NNLS มักจะ "พยายามอธิบาย Noise" โดยการโยนค่าน้ำหนักเล็กๆ น้อยๆ ไปให้สารที่ไม่มีอยู่จริง (เกิด False Positives) ด่านนี้จึงใช้ **กฎเหล็ก 2 ข้อ** ในการกรองสารหลอกทิ้ง:

1. **Fixed Threshold (`> 0.25`):** น้ำหนักของสารต้องมีค่าเกิน 0.25 ขึ้นไป จึงจะถือว่าเป็นสารที่มีนัยสำคัญ
2. **Adaptive Ratio (`> 0.10`):** น้ำหนักของสารตัวนั้น ต้องมีค่า **ไม่น้อยกว่า 10% ของน้ำหนักสารที่เยอะที่สุดในหลอดเดียวกัน** (เช่น ถ้าในหลอดมีสาร A น้ำหนัก 10.0 สาร B จะต้องมีน้ำหนักอย่างน้อย 1.0 ถึงจะรอด)
> **💡 ผลลัพธ์ Stage 3:** ทลาย False Positives ลงอย่างราบคาบ ดันค่า Precision ให้พุ่งจาก 54% กลายเป็น 85.74% ได้สำเร็จ!

---

## 3. สรุปประสิทธิภาพเชิงตัวเลข (Performance Metrics)
* **Dataset:** 10,000 Mock Samples (1328 Classes, 3-7 Metabolites per sample, Gaussian noise added)
* **Macro F1-Score:** **0.8206** (82.06%)
* **Precision:** 0.8574
* **Recall:** 0.7994
* **Inference Speed:** ~193 ms / Sample (ทำงานรวดเร็วด้วย Multiprocessing 4-cores และ Memory-mapped Caching)
