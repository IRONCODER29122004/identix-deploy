# Deepfake Detection Study From Faculty Paper (Line-Traceable)

Source analyzed: `Required/phd_mam_research_paper_extracted.txt`
Paper title in source: *Deepfake Detection Using Meso4Net and Capsule Networks Through Facial Feature and Pattern Analysis*

## 1) What the paper proposes

The paper proposes a hybrid model called **FFA-MPDV** that combines:
- **Meso4Net** for lightweight mesoscopic feature extraction.
- **Capsule Networks** for preserving spatial hierarchies and subtle feature relationships.
- **FPN (Feature Pyramid Network)** for multi-scale fusion of low/high level features.
- **Spatial attention** for focusing on manipulated facial regions.

### Line evidence
- Model name and contribution claims: lines 20-27.
- FPN + spatial domain emphasis: lines 114-117 and 169-171.
- Hybrid architecture section starts: line 681 onward.
- Meso4 + Capsule + FPN + spatial attention architecture statement: lines 834-846.

## 2) Dataset and class setup used in the paper

The methodology section states a Kaggle-collected dataset for binary classification with two classes:
- **Class 0:** fake/generated.
- **Class 1:** real/authentic.

### Reported counts in the paper text
- Total images: **190,335**
- Fake: **95,134**
- Real: **95,201**
- Train: 70,001 fake + 70,001 real
- Test: 5,492 fake + 5,413 real
- Validation: 19,641 fake + 19,787 real

### Line evidence
- Dataset totals and split counts: lines 686-694.
- DFDC subset statement: lines 696-705.
- Why DFDC over others: lines 706-716.
- Class labels: lines 735-737.

## 3) Preprocessing and augmentation recipe

The extracted methodology indicates:
- Remove duplicates/corrupted files.
- Normalize pixel values to [0,1] using rescale 1/255.
- Resize to 256x256 RGB.
- Augmentation with horizontal flip, zoom, and shear/shift styles.

### Line evidence
- Data cleaning note: lines 746-747.
- Normalization and 1/255: lines 750-756.
- Resize to 256x256x3: lines 771-774.
- Augmentation details: lines 790-826.

## 4) Architecture details captured from the text

### Meso4 branch
- Input: 256x256x3.
- Four conv-style stages with pooling and dropout regularization.
- Lightweight parameterization intended for efficient deepfake detection.

### Capsule branch
- Capsule routing by agreement with softmax coupling coefficients.
- Intended to preserve hierarchical spatial relations and subtle manipulations.

### FPN + Spatial attention
- Multi-scale feature fusion through pyramid-like merging.
- Spatial attention uses avg/max pooled maps + conv + sigmoid to focus regions.

### Line evidence
- Meso4 details and input size: lines 1069-1134.
- Capsule routing and coefficients: lines 998-1024.
- FPN multi-scale extraction: lines 896-919.
- Spatial attention math and operation: lines 922-953.

## 5) Training setup and metrics found in the text

### Hyperparameter details explicitly mentioned
- Adam optimizer.
- Learning rate references: 0.001 (text discussion) and 0.0001 (optimizer bullet).
- Batch size tested: 16 and 32.
- Threshold 0.5 for binary decision.

### Evaluation discussion in paper text
- Accuracy, precision, recall, F1, ROC-AUC, PR curve, confusion matrix.
- Reported top performance claim for FFA-MPDV on DFDC: around
  - Accuracy: 97.30%
  - Precision: 96.45%
  - Recall: 96.00%

### Line evidence
- Adam and LR explanation: lines 1150-1163.
- Hyperparameter bullets including batch sizes: lines 1191-1211.
- Threshold rule (0.5): lines 1242-1253.
- Results section beginning: lines 1305 onward.
- Comparative claims and best performance wording: lines 1404-1420.

## 6) Practical implementation notes for Kaggle

Because the extracted text contains a few internal inconsistencies (for example MSE mention for binary classification while also discussing thresholded binary prediction), the implementation should:
- Keep architecture alignment with the paper (Meso4 + Capsule + FPN + spatial attention).
- Use robust modern binary objective by default (`BCEWithLogitsLoss`) for stable convergence.
- Optionally expose MSE as an experimental toggle for strict reproduction attempts.
- Preserve binary thresholding at 0.5 for final class prediction.

## 7) What is implemented in the accompanying notebook

Notebook created: `Required/deepfake_detection_kaggle_ffa_mpdv.ipynb`

It includes:
- Kaggle path auto-discovery and dataset scanning for fake/real folders.
- 256x256 preprocessing and paper-aligned augmentation.
- FFA-MPDV-style architecture in PyTorch:
  - Meso4 branch
  - Capsule-style branch with dynamic routing
  - FPN-like multi-scale fusion
  - Spatial attention
  - Binary head
- Training + validation loop with metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC).
- Confusion matrix + ROC/PR plotting.
- Export to `.pth` bundle for Kaggle artifacts.

## 8) Limitations and reproducibility cautions

- Exact table reconstruction from PDF extraction may lose formatting due OCR/text conversion artifacts.
- Some claims in text appear narrative/summary rather than fully specified reproducible protocol.
- Cross-dataset generalization claims are discussed, but this notebook is focused on a Kaggle single-dataset training workflow first.

## 9) Next-step experiments recommended

1. Run strict ablation variants:
   - Meso4 only
   - Meso4 + Capsule
   - Meso4 + Capsule + FPN
   - Full FFA-MPDV
2. Compare BCE vs MSE training objective while keeping architecture fixed.
3. Add cross-dataset validation (FaceForensics++ / Celeb-DF if available in environment).
4. Add mixed precision + gradient accumulation for larger effective batch size on Kaggle GPUs.
