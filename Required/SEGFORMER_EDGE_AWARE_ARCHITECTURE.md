# SegFormer-Lite Edge-Aware Segmentation Model
## Detailed Architecture & Training Specifications

**Date Created:** February 27, 2026  
**Project:** Capstone 4-1 - Enhanced Facial Landmark Segmentation  
**Model Version:** SegFormer-MiT-B0 with Edge-Aware Dual-Head Architecture  

---

## 1. Executive Summary

This document specifies a new lightweight segmentation model designed to complement the existing BiSeNet facial landmark detector. The model uses a SegFormer-Lite architecture with dual heads (region + edge) to improve boundary accuracy while maintaining CPU compatibility for training on resource-constrained hardware.

**Key Constraints:**
- CPU-only training (i5-1235U, 16GB RAM)
- Input resolution: 384×384 (balance between accuracy and speed)
- Batch size: 4–8 with gradient accumulation
- Training duration: 50–80 epochs
- No modifications to existing BiSeNet or landmark_app.py code

---

## 2. Architecture Overview

### 2.1 Backbone: Efficient Transformer (MiT-B0)
- **Model:** SegFormer-B0 / MiT-B0 (Mitsubaish Transformer)
- **Parameters:** ~3.7M (lightweight for CPU inference)
- **Feature Extraction:** 4 stages with hierarchical multi-scale features
- **Patch Embedding:** Progressive patch embedding from 4×4 to 16×16
- **Receptive Field:** Designed for efficient global context aggregation

**Stage Architecture:**
```
Input (384×384)
    ↓
Stage 1: Patch Embed (H/4, W/4) → DepthWise Conv + FeedForward
    ↓ (96 channels)
Stage 2: Patch Embed (H/8, W/8) → Multi-Head Self-Attention
    ↓ (192 channels)
Stage 3: Patch Embed (H/16, W/16) → Cross-Attention
    ↓ (384 channels)
Stage 4: Patch Embed (H/32, W/32) → Long-Range Context
    ↓ (768 channels)
```

### 2.2 Dual-Head Decoder Architecture

#### **Head 1: Region Segmentation (11-class)**
- **Input Features:** Multi-scale outputs from MiT-B0 (4 stages)
- **Decoder Blocks:** Progressive upsampling with feature fusion
  - All-MLP decoder (simple + efficient for CPU)
  - Channel reduction: 256 → 128 → 64
- **Output:** 384×384 × 11 (logits for 11 facial regions)

**Classes:**
1. Background
2. Skin
3. Left Eyebrow
4. Right Eyebrow
5. Left Eye
6. Right Eye
7. Nose
8. Lips (outer)
9. Teeth
10. Hair
11. Inner Mouth

#### **Head 2: Edge Detection (Binary)**
- **Input Features:** High-resolution features from Stage 1 + 2
- **Purpose:** Detect facial region boundaries with high precision
- **Decoder:** Lightweight CNN (3 conv layers + transposed convolutions)
- **Output:** 384×384 × 1 (probability of edge pixel)

**Edge Generation Strategy:**
- Derived from 11-class masks using morphological gradient
- Distance-based boundaries: pixels at region transitions
- Positive class: edge pixels; Negative: non-edge pixels

### 2.3 Feature Fusion & Edge-Aware Refinement

**Fusion Block:**
```
region_logits (384×384×11)
    ↓
Softmax → region_confidence
    ↓
edge_predictions (384×384×1)
    ↓
Sigmoid → edge_confidence
    ↓
[Edge-Aware Refinement]
    ↓
refined_region_logits (384×384×11)
    ↓
Upsampled to original image size (via bilinear in inference)
```

**Refinement Logic:**
```python
# Pseudo-code
boundary_mask = (edge_confidence > 0.5).float()
refined_logits = region_logits * (1 - boundary_mask * 0.2)
# Reduce confidence at predicted edges → encourage decision boundary alignment
```

---

## 3. Loss Functions

### 3.1 Region Segmentation Loss
**Function:** Cross-Entropy + Dice Loss (weighted)

```
L_region = α * CrossEntropyLoss(region_logits, region_labels) 
         + β * DiceLoss(region_logits, region_labels)
```

**Parameters:**
- α = 0.6 (CE weight, handles class imbalance via label smoothing)
- β = 0.4 (Dice weight, improves boundary alignment)
- **Class Weights:** Inversely proportional to class frequency (hair, background weighted less)
- **Label Smoothing:** ε = 0.05 (soft targets for regularization)

### 3.2 Edge Detection Loss
**Function:** Binary Cross-Entropy + Dice Loss (weighted)

```
L_edge = γ * BCEWithLogitsLoss(edge_logits, edge_labels) 
       + δ * DiceLoss(edge_predictions, edge_labels)
```

**Parameters:**
- γ = 0.5 (BCE weight, standard for binary classification)
- δ = 0.5 (Dice weight, balances recall/precision on edges)
- **Positive Weight:** 2.0 (edge pixels less frequent, down-weight false negatives)

### 3.3 Consistency Loss (Boundary Alignment)
**Function:** L2 between region boundaries and edge predictions

```
L_consistency = λ * ||region_boundaries - edge_predictions||²
              where region_boundaries = Sobel(region_confidence)
```

**Parameters:**
- λ = 0.1 (consistency weight, ensures coherence)

### 3.4 Total Loss
```
L_total = L_region + L_edge + L_consistency
```

---

## 4. Training Configuration

### 4.1 Optimizer & Scheduler
- **Optimizer:** AdamW
  - Learning Rate: 3e-4
  - Beta-1: 0.9, Beta-2: 0.999
  - Weight Decay: 1e-2 (L2 regularization)
  
- **Scheduler:** Cosine Annealing with Warmup
  - Warmup Epochs: 5
  - Cosine Decay: 50–80 total epochs
  - Minimum LR: 1e-5 (prevents underfitting at end)

### 4.2 Batch Configuration (CPU-Optimized)
- **Batch Size (per GPU):** 4
- **Gradient Accumulation Steps:** 4
- **Effective Batch Size:** 16 (equivalent)
- **Epochs:** 50–80 (80 if validation improves, stop at 50 if plateau)

**Rationale:** On CPU, small batches (4) prevent memory overflow. Accumulation simulates larger batches without causing OOM.

### 4.3 Data Augmentation

**Augmentation Pipeline (Albumentations):**
```python
transforms = [
    # Geometric
    Resize(384, 384),
    HorizontalFlip(p=0.5),
    RandomResizedCrop(384, 384, scale=(0.8, 1.0), p=0.3),
    Rotate(limit=15, p=0.3),
    GaussDistortion(p=0.1),
    
    # Photometric
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),
    RandomGamma(gamma_limit=(80, 120), p=0.2),
    
    # Noise & Blur
    GaussNoise(var_limit=(10.0, 50.0), p=0.1),
    MotionBlur(blur_limit=(3, 7), p=0.1),
    
    # Normalization
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet
]
```

**Train/Val Split:** 80% / 20% of existing CelebAMask-HQ

---

## 5. Dataset Preparation

### 5.1 Data Structure
```
Required/data/datasets/
├── train/
│   ├── images/          (RGB images, 512×512 or higher)
│   ├── labels/          (11-class segmentation masks)
│   └── landmarks/       (face landmark annotations, optional)
├── val/
│   ├── images/
│   ├── labels/
│   └── landmarks/
└── test/
    ├── images/
    ├── labels/
    └── landmarks/
```

### 5.2 Edge Map Generation

**Algorithm:**
1. Load 11-class mask (integer values 0–10)
2. Apply morphological gradient:
   ```
   edge_map = dilate(mask) - erode(mask)
   ```
3. Binary conversion: `edge_map = (edge_map > 0).astype(uint8)`
4. Optional smoothing: Gaussian blur (σ=1.0) for soft edges

**Output:** Binary edge maps (0/1) co-registered with images

### 5.3 Data Loader Features
- **Lazy Loading:** Load images on-the-fly (RAM efficient)
- **Cache Strategy:** Optionally cache preprocessed masks in memory if RAM available
- **Workers:** 0 (CPU training, parallel data loading often slower than sequential)
- **Prefetch:** Load next batch while GPU/CPU processes current batch

---

## 6. Training Metrics & Evaluation

### 6.1 Per-Epoch Metrics (Logged)

**Region Head:**
- Mean Intersection-over-Union (mIoU): Average IoU across 11 classes
- Per-Class IoU: Individual class performance (highlight hair, lips separately)
- Dice Score: Harmonic mean of precision/recall

**Edge Head:**
- F1-Score: Balance between edge recall and precision
- Boundary Accuracy: % of predicted edges within 2 pixels of GT boundaries

### 6.2 Validation Strategy
- **Frequency:** Every 5 epochs (checkpoint on best mIoU)
- **Early Stopping:** If validation mIoU doesn't improve for 15 epochs → stop
- **Best Model:** Saved checkpoint with highest mean(mIoU + F1/10)

### 6.3 Post-Training Evaluation (Test Set)

**Metrics:**
- mIoU, Per-Class IoU, Dice Coefficient
- Inference Speed (ms/image on CPU)
- Boundary F-score (edge accuracy)
- Comparison vs BiSeNet (using test/ split)

---

## 7. Implementation Structure

### 7.1 New Files to Create
1. **segformer_edge_aware.py** (Model definition)
   - SegFormer-B0 backbone wrapper
   - Region decoder (11-class)
   - Edge decoder (binary)
   - Fusion block
   
2. **edge_map_generator.py** (Utility)
   - Load 11-class masks → generate binary edge maps
   - Morphological preprocessing

3. **data_loader_edge.py** (Dataset)
   - Custom PyTorch Dataset class
   - Augmentation pipeline
   - Batch collation

4. **train_segformer_notebook.ipynb** (Training)
   - Model initialization
   - Loss function definitions
   - Training loop with validation
   - Checkpointing & early stopping
   - Metrics visualization

5. **evaluate_segformer.py** (Evaluation)
   - Load trained model
   - Inference on test set
   - Compute mIoU, F-score, speed benchmarks
   - Comparison vs BiSeNet (optional)

### 7.2 Dependencies
```
torch==2.0+
transformers>=4.30.0  (for SegFormer backbone)
albumentations>=1.3.0
scikit-image
scikit-learn
opencv-python>=4.7.0
matplotlib
seaborn
tqdm
```

---

## 8. Expected Performance

### 8.1 Training Dynamics

**Epoch Progression:**
| Epoch | mIoU (Region) | F1 (Edge) | Loss | Inference (ms) |
|-------|---------------|-----------|------|----------------|
| 1–10  | ~0.25–0.45    | ~0.30–0.50 | 2.5–1.5 | 500–450 |
| 11–30 | ~0.45–0.60    | ~0.50–0.70 | 1.5–0.8 | 450–400 |
| 31–50 | ~0.60–0.72    | ~0.70–0.80 | 0.8–0.5 | 400–380 |
| 51–80 | ~0.72–0.78    | ~0.80–0.85 | 0.5–0.35 | 380–370 |

**Rationale:**
- Early epochs show rapid improvement (random → structured features)
- Plateau after ~50 epochs (CPU training slower to converge)
- Edge head learns faster (simpler binary task)
- Region head refines boundary sharpness in later epochs

### 8.2 Inference Speed (CPU i5-1235U)
- **Batch Size 1:** ~400–500 ms/image (384×384)
- **Batch Size 4:** ~100–120 ms/image (pipelined)
- **Comparison vs BiSeNet:** 2–3× slower (larger input + dual heads)

**Optimization Options:**
- Quantization (FP16 or INT8) → 50% speedup, minimal accuracy loss
- Model distillation (smaller teacher) → future work

### 8.3 Target Accuracy
- **mIoU:** ≥0.72 (improvement over BiSeNet ~0.65–0.68)
- **Edge F1:** ≥0.80 (boundary-focused improvement)
- **Per-Class:** Especially improve hair, lips, eyebrow IoU

---

## 9. Integration with Existing System

### 9.1 No Changes to BiSeNet
- Existing landmark_app.py remains untouched
- BiSeNet inference pipeline unchanged
- New model runs as **separate parallel module** (optional)

### 9.2 Deployment Path (Future)
```python
# In landmark_app.py (pseudo-code, not yet implemented)
from segformer_edge_aware import SegFormerEdgeAware

# Option 1: Use BiSeNet (existing)
segmentation = bisenet_model(image)

# Option 2: Use SegFormer (new, if requested)
segmentation = segformer_model(image)

# Option 3: Ensemble (average predictions)
segmentation = 0.5 * bisenet_model(image) + 0.5 * segformer_model(image[resized])
```

---

## 10. Hyperparameter Justification

| Hyperparameter | Value | Rationale |
|---|---|---|
| Input Size | 384×384 | Balance: BiSeNet uses 256; doubled for better boundary (cost: 2.25× slower) |
| Backbone | MiT-B0 | Lightweight (3.7M params), efficient for CPU, proven on facial segmentation |
| LR | 3e-4 | Standard for Transformer fine-tuning; AdamW default |
| Batch Size | 4 + accumulation | CPU memory limit; accumulation simulates larger effective batch |
| Warmup | 5 epochs | Stabilize initial learning before cosine decay |
| λ (consistency) | 0.1 | Weak constraint; primary losses dominate |
| Edge positive weight | 2.0 | Edge pixels ~20% of data; weight compensates for class imbalance |

---

## 11. Troubleshooting & Monitoring

### 11.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM during training | Batch too large | Reduce to batch_size=2, accumulation=8 |
| Loss NaN after epoch 5 | Learning rate too high | Reduce LR to 1e-4 |
| mIoU plateaus at 0.35 | Poor data loading | Verify edge maps generated correctly; check class weights |
| Inference slow (>1000ms) | CPU bottleneck | Use FP16, reduce input to 320×320, or switch to GPU |

### 11.2 Monitoring Dashboards (in Notebook)
- Real-time loss curves (train/val) with matplotlib
- Per-class IoU bar charts (epoch-wise)
- Edge F1 evolution
- Learning rate schedule visualization
- Sample predictions (overlay) every 10 epochs

---

## 12. References & Credits

- **SegFormer Paper:** Xie et al., 2021 (Efficient Semantic Segmentation with Transformer)
- **BiSeNet Baseline:** BiSeNet v2 (used in landmark_app.py)
- **CelebAMask-HQ Dataset:** Lee et al., 2020
- **Albumentations:** Image augmentation library

---

## Appendix: File Locations

**New Files Created:**
```
Required/
├── segformer_edge_aware.py              (Model definition)
├── edge_map_generator.py                (Edge map preprocessing)
├── data_loader_edge.py                  (Custom dataset)
├── train_segformer_notebook.ipynb       (Training notebook)
├── evaluate_segformer.py                (Evaluation script)
├── SEGFORMER_EDGE_AWARE_ARCHITECTURE.md (This document)
```

**Dataset Used (Existing):**
```
Required/data/datasets/
├── train/
├── val/
└── test/
```

**No Changes To:**
```
Required/
├── landmark_app.py          (UNCHANGED)
├── deepfake_detector.py     (UNCHANGED)
├── hybrid_detector.py       (UNCHANGED)
├── models/
│   ├── best_model.pth       (UNCHANGED)
│   └── ...                  (UNCHANGED)
```

---

**Document Version:** 1.0  
**Last Updated:** February 27, 2026  
**Status:** Ready for Implementation
