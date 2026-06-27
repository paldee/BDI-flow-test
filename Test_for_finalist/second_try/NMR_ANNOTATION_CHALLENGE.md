# 🔬 1D NMR Metabolite Annotation Challenge (BDI Hackathon)

Welcome to the 1D NMR Metabolite Annotation Challenge. This document is designed for AI coding agents to understand the datasets, the problem formulation, current benchmark baselines, and suggestions to push performance beyond the **52.13% Macro F1-Score** bottleneck.

---

## 🎯 Problem Statement
Given a raw 1D $^1\text{H}$ NMR spectrum mixture (represented as 10,001 intensity values along a chemical shift axis from 0.0 to 10.0 ppm), the goal is to **identify which of the 1,328 possible reference metabolites are present in the mixture**. 

This is formulated as an **extreme multi-label classification** or **signal deconvolution** task.

### Key Challenges:
1. **Extreme Overlapping Peaks:** NMR spectra of mixtures are superpositions of individual metabolite spectra. Since multiple compounds share similar chemical environments, their peaks overlap extensively.
2. **Sparsity:** While there are 1,328 candidate metabolites, each sample contains only a small subset (averaging 15 metabolites, modeled via a Poisson distribution clipped between 3 and 50). The positive label rate is extremely low (~1.1%).
3. **Chemical Shift Drifts:** Real-world experimental conditions (pH, temperature, ionic strength) cause peaks to shift slightly along the ppm axis. A residual shift of $\sigma = 0.0005$ ppm has been simulated.
4. **Noise:** Low-level Gaussian noise ($\sigma = 0.01$) is injected.

---

## 📁 Dataset & File Structures

All data files are located in the workspace under the `data/` directory:
* Data file: [mock_nmr_data_10k.csv](file:///d:/hack/BDI/Test_for_finalist/second_try/data/mock_nmr_data_10k.csv) (~661 MB)
* Labels file: [mock_ground_truth_10k.csv](file:///d:/hack/BDI/Test_for_finalist/second_try/data/mock_ground_truth_10k.csv) (~53 MB)
* Reference library: [hmdb_reference_library.csv](file:///d:/hack/BDI/Test_for_finalist/second_try/data/hmdb_reference/hmdb_reference_library.csv) (~126 MB)

### 1. Mixture Spectra: `mock_nmr_data_10k.csv`
* **Format:** CSV with shape `(10001, 10001)` (10,001 rows for ppm values, 1 column for `ppm`, and 10,000 columns for samples).
* **Columns:**
  - `ppm`: Chemical shift axis values starting from `0.0` to `10.0` at steps of `0.001` ppm (length 10,001).
  - `Sample_00001` to `Sample_10000`: Float values representing spectral intensity at each ppm coordinate.

### 2. Concentration Ground Truth: `mock_ground_truth_10k.csv`
* **Format:** CSV with shape `(10000, 1329)` (10,000 rows, 1 column for `Sample_ID`, and 1,328 columns for candidate metabolites).
* **Columns:**
  - `Sample_ID`: String identifier mapping to the column headers of the spectra dataset.
  - 1,328 columns (e.g., `1-Methylhistidine`, `Cysteine`, etc.): Float values representing the ground-truth concentration of each metabolite.
  - **Classification Target:** A metabolite is **present** (Class Label = 1) if its concentration is $> 0.0$, and **absent** (Class Label = 0) if concentration is $0.0$.

### 3. Pure Reference Library: `hmdb_reference/hmdb_reference_library.csv`
Contains the individual peak signatures of the 1,328 reference metabolites.
* **Columns:**
  - `Name`: Name of the metabolite.
  - `cluster`: Coarse chemical shift bin.
  - `loc`: The exact center of the peak (ppm).
  - `height`: The relative height (intensity) of the peak.
  - `width`: The width (FWHM) of the peak.
* **Pure Spectrum Reconstruction:**
  To reconstruct the pure spectrum of a metabolite $m$, you can model each peak as a Lorentzian function and sum them:
  $$\text{spectrum}_m(x) = \sum_{\text{peaks } p} \frac{\text{height}_p}{1 + \left(\frac{x - \text{loc}_p}{\text{width}_p / 900.0}\right)^2}$$
  *(Note: Peak width in the library is divided by 450.0, corresponding to a FWHM of `width_p / 450.0`. In python vectorized spectra generation, this is typically scaled based on resolution).*

---

## 💻 Loading Data in Python

Since the datasets are large, loading them efficiently is crucial to avoid RAM overflows:

```python
import pandas as pd
import numpy as np

# 1. Load spectra (transpose to get shape: [samples, 10001])
print("Loading spectra data...")
df_spectra = pd.read_csv("data/mock_nmr_data_10k.csv")
ppm_axis = df_spectra["ppm"].values
# Transpose so that rows are samples and columns are ppm bins
X = df_spectra.drop(columns=["ppm"]).values.T  # Shape: (10000, 10001)

# 2. Load ground-truth concentrations and convert to binary labels
print("Loading ground truth...")
df_gt = pd.read_csv("data/mock_ground_truth_10k.csv")
metabolite_names = list(df_gt.columns[1:])  # Skip 'Sample_ID'
concentrations = df_gt.drop(columns=["Sample_ID"]).values
y = (concentrations > 0.0).astype(int)  # Shape: (10000, 1328)

print(f"Features (X) Shape: {X.shape}")
print(f"Labels (y) Shape: {y.shape}")
```

---

## 🏆 Current Baselines & Benchmarks

The metric for evaluation is the **Macro F1-Score** across all 1,328 classes. Here are the results of previous experiments (tested on a hold-out test split of the 10,000 samples):

| Experiment | Method | Macro F1-Score | Precision | Recall | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-E** | Peak Picking + Rule-Based | 0.0226 | 0.0114 | **0.9990** | High false positive rate due to massive peak overlaps in mixtures. |
| **EXP-B** | Cosine Similarity | 0.1318 | 0.0799 | 0.9559 | Compares raw mixture to pure reference templates; struggles with mixtures and shift drifts. |
| **EXP-A** | NMF Decomposition (K=50) | 0.0032 | 0.0017 | 0.0372 | Fails because a static matrix decomposition cannot resolve 1,328 potential components with a small $K$. |
| **EXP-C** | DTW + Cosine Pre-filter | 0.3248 | 0.3943 | 0.3723 | 2-stage: Cosine similarity filters top-30 candidates, then Dynamic Time Warping (DTW) matches peaks. |
| **EXP-D** | **Multi-Window FAISS (Best)** | **0.5213** | **0.5620** | 0.4930 | Unsupervised candidate. Uses sliding windows (0.2 ppm), queries a FAISS index of pure signatures, and votes. |
| **EXP-F** | 1D-CNN + Transformer | 0.0905 | 0.0696 | 0.1735 | Supervised deep learning. Suffers from severe **data starvation** due to training a 1,328-label classifier on only 10,000 samples. |

---

## 💡 Directions to Beat 52% F1-Score

To improve the annotation score, here are several promising directions you should explore:

### 1. Hybrid FAISS + Advanced Reranking
* **Optimized Voting:** Instead of simple majority voting in `EXP-D`, implement a weighted voting scheme based on FAISS distance, peak heights, or peak matching ratios.
* **Constrained Dynamic Time Warping (DTW):** Run the Multi-Window FAISS to retrieve top candidate metabolites, then apply a localized DTW to align peak shifts and calculate exact match qualities.

### 2. Deep Learning with Data Augmentation & Pretraining
* **Sim2Real Data Pretraining:** The current Deep Learning model (`EXP-F`) failed due to training size limitations. Since we have the exact Lorentzian generator (`generate_mock_data.py`), you can generate **100k+ mock samples** offline to pretrain the network before fine-tuning.
* **Multi-Scale 1D-CNN / ResNet:** Implement a multi-scale kernel size network to capture both narrow peaks and wider multiplet features.

### 3. Sparse Dictionary Learning & Optimization
* **Non-negative Least Squares (NNLS) with L1 Regularization (Lasso):**
  Since the mixture spectrum is a linear combination of pure spectra, you can solve:
  $$\min_{\mathbf{w} \ge 0} \|\mathbf{X} - \mathbf{\Phi}\mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_1$$
  where $\mathbf{\Phi}$ is the $10001 \times 1328$ dictionary of reconstructed pure spectra.
  - *Tip:* Since there are shift drifts, you can expand the dictionary $\mathbf{\Phi}$ by adding slightly shifted versions of the pure spectra, or solve this iteratively with alignment.

### 4. Graph Neural Networks (GNN) on Peak Graphs
* Represent each spectrum as a set of detected peaks (nodes) with relative distances (edges). Perform graph matching between the mixture peak graph and the reference library templates.

---

**Your goal is to choose one or more of these strategies, implement them, and evaluate them on the 10k dataset to see if they can break the 52.13% F1-score barrier! Good luck!**
