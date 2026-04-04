# Deepfake Detector v2 - Quick Reference Card

## TLDR - Get Started in 30 Seconds

```bash
# Test the model is working
python test_v2_quick.py

# Process crops from your video
python test_pipeline_v2.py data/pipelines_crops my_results
```

---

## API Quick Reference

### Python: Single Image
```python
from deepfake_detector_v2 import load_v2_model

detector = load_v2_model('models/deepfake_detector_v2_ffa_mpdv.pth')
result = detector.predict('image.jpg')
print(f"{result['label_name']}: {result['confidence']:.0%}")
```

### Python: Multiple Images
```python
results = detector.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
for r in results:
    print(f"{r['label_name']}")
```

### Python: Full Pipeline
```python
from test_pipeline_v2 import run_v2_inference_on_crops

summary = run_v2_inference_on_crops('data/pipelines_crops', output_dir='results')
print(f"FAKE: {summary['fake_percentage']:.1f}%")
```

### Command Line
```bash
python test_pipeline_v2.py <crop_dir> <output_dir>
```

---

## Output Meanings

| Field | Range | Meaning |
|-------|-------|---------|
| `label` | 0 or 1 | 0=REAL, 1=FAKE |
| `proba` | 0.0-1.0 | Probability of being FAKE |
| `confidence` | 0.0-1.0 | Distance from 0.5 (higher = more certain) |
| `logit` | -∞ to +∞ | Raw model output |

**Example**: `{'label': 1, 'proba': 0.95, 'confidence': 0.90}` means **VERY LIKELY FAKE**

---

## Files Overview

| File | Purpose | Size |
|------|---------|------|
| `deepfake_detector_v2.py` | Model wrapper & inference | 380 lines |
| `test_pipeline_v2.py` | Batch processing pipeline | 400 lines |
| `test_v2_quick.py` | Validation tests | 300 lines |
| `models/deepfake_detector_v2_ffa_mpdv.pth` | Trained weights | 3.2 MB |

---

## Common Tasks

### Task 1: Test if v2 works
```bash
cd Required
python test_v2_quick.py
# Expected: 3/3 tests passed ✅
```

### Task 2: Process crops from a video
```bash
python test_pipeline_v2.py data/pipelines_crops results/my_analysis
# Outputs:
#   - results/my_analysis/overlays/        (images with predictions)
#   - results/my_analysis/results.json     (detailed results)
#   - results/my_analysis/REPORT.txt       (summary)
```

### Task 3: Compare v1 vs v2
```python
# v1: Landmark-based
from deepfake_detector import DeepfakeDetector
v1 = DeepfakeDetector()
v1_score = v1.analyze_texture_naturalness(frames, bboxes)  # Returns score 0-100

# v2: Neural network
from deepfake_detector_v2 import load_v2_model
v2 = load_v2_model('models/deepfake_detector_v2_ffa_mpdv.pth')
v2_result = v2.predict(image)  # Returns {'proba': 0.0-1.0, ...}
```

### Task 4: Get model info
```python
from deepfake_detector_v2 import load_v2_model
detector = load_v2_model('models/deepfake_detector_v2_ffa_mpdv.pth')
info = detector.get_model_info()
print(info['final_metrics'])
# Shows: ROC-AUC 0.921, F1 0.821, Precision 0.886, etc.
```

---

## Performance Summary

- **Speed**: ~50-100ms per image (CPU)
- **Accuracy**: 92.1% ROC-AUC on validation set
- **Memory**: ~1.2 GB GPU (batch size 32)
- **Threshold**: 0.5 (adjustable)

---

## What's Different from v1?

| Aspect | v1 | v2 |
|--------|----|----|
| **Type** | Heuristic | Deep Learning |
| **Features** | Landmarks, texture | Multi-scale CNN features |
| **Speed** | Real-time | ~100ms/image |
| **Accuracy** | ~75% | ~92% ROC-AUC |
| **GPU Needed** | No | Optional |

---

## Troubleshooting

**Q: Model not loading?**  
A: Check file exists: `models/deepfake_detector_v2_ffa_mpdv.pth` ✅

**Q: Out of memory?**  
A: Use `device='cpu'` or smaller batches

**Q: Low confidence scores?**  
A: Normal. Check image is valid face crop

**Q: Why different from v1?**  
A: Different architectures. Run both for ensemble.

---

## Important Paths

```
models/
  └─ deepfake_detector_v2_ffa_mpdv.pth    ← Model file here
  
Required/
  ├─ deepfake_detector_v2.py              ← Inference wrapper
  ├─ test_pipeline_v2.py                  ← Batch processor
  ├─ test_v2_quick.py                     ← Quick test
  ├─ deepfake_detector.py                 ← v1 (unchanged)
  └─ pipeline_runner.py                   ← Main pipeline (unchanged)

data/
  ├─ pipelines_crops/                     ← Input crops
  └─ v2_detection_results/                ← Output directory
```

---

## One-Liner Commands

```bash
# Quick validation
python test_v2_quick.py

# Process all crops
python test_pipeline_v2.py data/pipelines_crops results

# View results
cat results/results.json | python -m json.tool

# Count fakes
python -c "import json; r=json.load(open('results/results.json')); print(f\"Fake: {r['fake_percentage']:.1f}%\")"
```

---

## Training Info (Reference)

- **Epochs**: 50
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: MSE
- **Batch size**: 16
- **Dataset**: DFDC + FaceForensics++ mixed
- **Hardware**: GPU training (Kaggle)
- **Final ROC-AUC**: 0.921 (92.1%)

---

## Status: ✅ READY TO USE

All tests passing. No additional setup needed. Just import and predict!

```python
from deepfake_detector_v2 import load_v2_model
detector = load_v2_model('models/deepfake_detector_v2_ffa_mpdv.pth')
result = detector.predict('image.jpg')
print(f"Result: {result['label_name']}")
```

**That's it!** 🎉
