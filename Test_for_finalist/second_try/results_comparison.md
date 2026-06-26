# 🏆 Finalist Test Results Comparison

| Experiment | Method | F1-Score (Macro) | Precision | Recall | Time per Sample (ms) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-E** | Peak Picking + Rule-Based | 0.0226 | 0.0114 | 0.9990 | 104.79 | Baseline: High false positive rate due to peak overlap. |
| **EXP-B** | Cosine Similarity | 0.1318 | 0.0799 | 0.9559 | 0.18 | Very fast (Matrix Mult), but fails on mixtures and shift drifts. |
| **EXP-A** | NMF Decomposition | 0.0032 | 0.0017 | 0.0372 | 3.00 | Global NMF (K=50) fails because it can only predict at most 50 metabolites out of 1328, destroying macro F1. |
| **EXP-C** | DTW + Cosine Pre-filter | **0.3248** | 0.3943 | 0.3723 | 13.68 | Best so far! 2-stage: cosine pre-filter (top-30) + shift-tolerant DTW. |
| **EXP-D** | Multi-Window FAISS | **0.5213** | 0.5620 | 0.4930 | 0.20 | Tuned (Window=0.2ppm, TopK=3, Vote=3). **Massive jump to 52% F1!** Unsupervised winner. |
| **EXP-F** | 1D-CNN + Transformer | 0.0905 | 0.0696 | 0.1735 | 0.96 | DL struggled with 1328 classes on only 10k samples (underfitting). Needs 100k+ samples to beat FAISS. |
