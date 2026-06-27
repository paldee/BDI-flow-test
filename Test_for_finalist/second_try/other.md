# 💡 สรุปแนวคิดการยกระดับโมเดล 1D NMR Metabolite Annotation

เอกสารนี้รวบรวมแนวคิดเชิงสถาปัตยกรรม 4 รูปแบบที่ออกแบบมาเพื่อทะลุขีดจำกัดคะแนน Macro F1-Score 52.13% ในชุดข้อมูล 1D NMR ที่เผชิญปัญหา Extreme Overlapping และ Sparsity สูง

---

## 1. Mathematical Optimization (Sparse Dictionary Learning)
การเปลี่ยนมุมมองจากการฝึกโมเดลให้ "ทำนาย" คลาส เป็นการ "แก้สมการหาค่าความเข้มข้น" (Signal Deconvolution)

* **แนวคิดหลัก:** สเปกตรัมที่ผสมกันคือผลรวมเชิงเส้น (Linear Combination) ของสเปกตรัมเดี่ยว เราสามารถสร้าง Dictionary Matrix ($\mathbf{\Phi}$) ที่เก็บรูปแบบ Peak ของทั้ง 1,328 สารไว้ และหาค่าน้ำหนัก ($\mathbf{w}$) ที่ทำให้เกิด Error น้อยที่สุด โดยบังคับให้คำตอบส่วนใหญ่เป็น 0 (Sparsity) ผ่านการทำ Non-negative Least Squares (NNLS) ร่วมกับ L1 Regularization (Lasso)
* **สมการอ้างอิง:**
  $$\min_{\mathbf{w} \ge 0} \|\mathbf{X} - \mathbf{\Phi}\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_1$$
* **การประยุกต์ใช้:** ขยายขนาด Dictionary โดยเพิ่มสเปกตรัมที่จำลองการเลื่อน (Shift Drifts) เข้าไปเผื่อไว้ หรือใช้การแก้สมการแบบวนลูป (Iterative Optimization) ควบคู่กับการทำ Alignment

---

## 2. Hybrid Routing (2-Stage Pipeline)
การสร้างระบบ Orchestration ที่ทำงานเป็นทอด ๆ เพื่อลด Latency และเพิ่มความแม่นยำ

* **Stage 1 (High Recall Filter):** ใช้โมเดลที่ทำงานเร็วและเบา เช่น Multi-Window FAISS ค้นหาและคัดกรอง "ผู้ต้องสงสัย" (Candidates) ให้เหลือเพียง 30-50 สารที่มีแนวโน้มเป็นไปได้มากที่สุด
* **Stage 2 (High Precision Reranker):** ส่งกลุ่ม Candidate ไปให้โมเดลที่ซับซ้อนกว่าเพื่อตัดสินใจขั้นสุดท้าย เช่น การใช้ Constrained Dynamic Time Warping (DTW) เพื่อวัดระยะห่าง Peak แบบยืดหยุ่น หรือใช้ Tree-based Model (XGBoost/LightGBM) เพื่อให้คะแนนความน่าจะเป็น

---

## 3. Graph Representation & Matching
การเปลี่ยนรูปแบบ Data Structure เพื่อลดข้อผิดพลาดจากการเลื่อนของแกน X (Chemical Shift Drifts)

* **แนวคิดหลัก:** แปลงสัญญาณ 1D NMR ให้กลายเป็นกราฟโครงสร้าง (Topological Graph)
  * **Node:** จุดยอด Peak ต่าง ๆ
  * **Edge:** ระยะห่างเชิงสัมพัทธ์ระหว่าง Peak
* **การประยุกต์ใช้:** ใช้อัลกอริทึม Graph Matching หรือ Graph Neural Networks (GNN) เพื่อเปรียบเทียบรูปแบบกราฟของสารผสมกับกราฟต้นแบบ เนื่องจากระยะห่างสัมพัทธ์ระหว่าง Peak ของสารตัวเดียวกันจะคงที่เสมอ วิธีนี้จึงทนทานต่อ Shift Drifts เป็นอย่างมาก

---

## 4. Sim2Real Data Generation & Deep Learning Ensembles
การแก้ปัญหา Data Starvation สำหรับสถาปัตยกรรม Deep Learning

* **แนวคิดหลัก:** ใช้สมการคณิตศาสตร์ (Lorentzian generator) สังเคราะห์ข้อมูล Mock Data ขึ้นมาเองจำนวนมหาศาล (100k+ ตัวอย่าง) โดยใส่ Noise ปรับสัดส่วนความเข้มข้น และจำลอง Shift Drifts ในรูปแบบต่าง ๆ เพื่อใช้เป็นข้อมูลฝึกสอน
* **การประยุกต์ใช้:** นำข้อมูลสังเคราะห์ไป Pre-train โมเดล Multi-Scale 1D-CNN (การใช้ Kernel หลายขนาดเพื่อจับลักษณะ Peak ทั้งแบบแคบและกว้าง) เพื่อให้โมเดลรู้จัก Feature พื้นฐาน ก่อนนำมา Fine-tune กับข้อมูลทดสอบจริง