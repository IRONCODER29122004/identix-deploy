# Version Compatibility Guide
## SegFormer Edge-Aware Training Notebook

**Last Updated:** February 27, 2026  
**Status:** Updated with comprehensive version checks and fallbacks

---

## ✅ Changes Made to Improve Version Compatibility

### 1. **Conservative Version Constraints**
Updated package installation with specific version ranges to prevent breaking changes:

```python
'torch>=2.0.0,<2.2.0'              # Avoid torch 2.2+ breaking changes
'transformers>=4.30.0,<4.40.0'     # Compatible range with torch 2.0+
'albumentations>=1.3.0,<1.5.0'     # Stable version range
'opencv-python>=4.7.0,<4.10.0'     # Prevent latest breaking changes
'numpy>=1.23.0,<2.0.0'             # NumPy 2.0 may have compatibility issues
```

**Why:** Loose version constraints (`>=2.0.0`) can install incompatible versions

### 2. **Automatic Version Checking**
Added comprehensive version detection and compatibility warnings:

- Prints all installed package versions at startup
- Detects problematic version combinations
- Warns about known issues (torch 2.2+, NumPy 2.0, etc.)
- Checks for required components (Transformers, Albumentations)

### 3. **Model Loading with Multi-Level Fallbacks**
Implemented 3-tier backup system for backbone loading:

**Tier 1:** SegformerForSemanticSegmentation (preferred, from transformers)  
↓ (if fails)  
**Tier 2:** SegformerModel with custom config  
↓ (if fails)  
**Tier 3:** Simple CNN backbone (all-Python, no external deps)

This ensures training can proceed even if transformers hub is unavailable.

### 4. **Matplotlib Style Compatibility**
Fixed matplotlib style system for version differences:

```python
# Tries in order:
'seaborn-v0_8-darkgrid'  # matplotlib 3.6+
'seaborn-darkgrid'       # matplotlib 3.5 and below
'ggplot'                 # fallback
```

### 5. **Enhanced Error Messages**
All error states now provide:
- Clear error type and message
- Likely cause explanation
- Suggested remediation steps

---

## 🔧 Common Version Issues & Solutions

### Issue: `AttributeError: module 'transformers' has no attribute 'SegformerForSemanticSegmentation'`

**Cause:** Transformers version mismatch  
**Solution:**
```bash
pip install transformers==4.35.0 --force-reinstall
```

### Issue: `RuntimeError: CUDA out of memory` (on CPU)

**Cause:** NumPy 2.0 or PyTorch 2.2+ memory allocation issue  
**Solution:**
```bash
pip install numpy==1.24.3 torch==2.0.1 --force-reinstall
```

### Issue: `ValueError: matplotlib style 'seaborn-v0_8-darkgrid' not found`

**Cause:** Old matplotlib version  
**Solution:** Auto-handled by notebook (uses fallback styles)

### Issue: `ModuleNotFoundError: No module named 'albumentations.pytorch'`

**Cause:** Albumentations installed without PyTorch integration  
**Solution:**
```bash
pip install albumentations[pytorch]==1.3.14
```

### Issue: `urllib3.exceptions.ConnectionError` during model download

**Cause:** Network issue or HuggingFace hub timeout  
**Solution:**
```python
# Cache the model first
from transformers import AutoModel
model = AutoModel.from_pretrained('nvidia/segformer-b0-finetuned-cityscapes-1024-1024')
# Save locally
model.save_pretrained('./segformer_cache')
```

---

## 📋 Recommended Installation Steps

### Step 1: Clean Previous Installations
```bash
pip uninstall torch transformers albumentations -y
```

### Step 2: Install with Specific Versions
```bash
pip install \
  torch==2.0.1 \
  transformers==4.35.0 \
  albumentations==1.3.14 \
  numpy==1.24.3 \
  opencv-python==4.8.1 \
  scikit-image==0.21.0 \
  scikit-learn==1.3.2 \
  matplotlib==3.7.2 \
  seaborn==0.13.0 \
  scipy==1.11.3 \
  Pillow==10.0.0 \
  tqdm==4.66.1
```

### Step 3: Verify Installation
Run the notebook's first two cells:
1. Package installation cell
2. Version checking cell

Expected output:
```
ENVIRONMENT & VERSION CHECK
============================
Python: 3.11.x ...
PyTorch: 2.0.1
Transformers: 4.35.0
NumPy: 1.24.3
...
✓ All compatibility checks passed!
```

---

## 🎯 Version Compatibility Matrix

| Component | Recommended | Min | Max | Notes |
|-----------|-------------|-----|-----|-------|
| Python | 3.10-3.11 | 3.8 | 3.12 | 3.12+ untested |
| PyTorch | 2.0.1 | 2.0.0 | 2.1.2 | Avoid 2.2+ (breaking changes) |
| Transformers | 4.35.0 | 4.30.0 | 4.39.0 | Older versions may lack features |
| NumPy | 1.24.x | 1.23.0 | 1.26.x | Avoid 2.0+ (alpha version) |
| Albumentations | 1.3.14 | 1.3.0 | 1.4.x | 1.5+ may have issues |
| OpenCV | 4.8.1 | 4.7.0 | 4.9.x | Avoid 4.10+ if issues |
| Matplotlib | 3.7.2 | 3.6.0 | 3.8.x | 3.5 uses older style names |
| Seaborn | 0.13.0 | 0.12.0 | 0.13.x | No critical version issues |

---

## 🚨 What to Do If Issues Persist

### Option 1: Use Docker (Most Reliable)
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
RUN pip install transformers==4.35.0 albumentations==1.3.14
```

### Option 2: Create Fresh Virtual Environment
```bash
python -m venv segformer_env
source segformer_env/bin/activate  # Linux/Mac
# or
segformer_env\Scripts\activate  # Windows

pip install -r requirements_pinned.txt
```

### Option 3: Check System Python Installation
```bash
python3.11 --version
python3.11 -m pip --version
```

### Option 4: Force Reinstall All Dependencies
```bash
pip cache purge
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements_pinned.txt --force-reinstall
```

---

## 📝 Key Changes Summary

### Installation Cell (#3)
- ✅ Added 12 packages with explicit version ranges
- ✅ Better error handling and reporting
- ✅ Shows installation progress per package
- ✅ Displays installation summary at end

### Imports Cell (#4)  
- ✅ Added automatic version detection
- ✅ Added compatibility checking function
- ✅ Warns about known issues
- ✅ Tests critical imports (Transformers, Albumentations)

### Model Initialization Cell (#11)
- ✅ 3-tier fallback system for backbone loading
- ✅ Better error messages for debugging
- ✅ Handles variable feature extraction methods
- ✅ Graceful degradation to CNN backbone if needed

### Visualization Cell (#20)
- ✅ Matplotlib style compatibility with version fallbacks
- ✅ Tests style availability before use
- ✅ Falls back to ggplot/default if needed

---

## 🔍 How to Monitor Version Compatibility

### During Training
The notebook will print:
```
ENVIRONMENT & VERSION CHECK
============================
Python: 3.11.x + ...
PyTorch: x.x.x
✓ All compatibility checks passed!
```

Or warnings like:
```
⚠ COMPATIBILITY WARNINGS:
  - PyTorch 2.2+ detected - some transformers may have issues
  - Seaborn v0_8 style not available - will use default matplotlib style
```

### In Case of Errors
Look for error messages starting with `✗`:
```
✗ MODEL INITIALIZATION FAILED
  Error: ImportError: cannot import name 'SegformerForSemanticSegmentation'
  
  This likely indicates a version compatibility issue.
  Check the version output above and consider:
  1. Installing specific PyTorch: pip install torch==2.0.1
  2. Installing specific transformers: pip install transformers==4.35.0
```

---

## ✅ Verification Checklist

- [ ] Run installation cell (cell 3)
- [ ] Check output for "✓ All packages installed successfully!"
- [ ] Run imports cell (cell 4)
- [ ] Check output for "✓ All compatibility checks passed!"
- [ ] Run model initialization cell (cell 11)
- [ ] Check output for "✓ Model initialized successfully"
- [ ] If any ✗ errors appear, follow the suggested fixes
- [ ] Proceed with training (cell 12)

---

## 📞 Support Resources

| Issue | Resource |
|-------|----------|
| PyTorch version issues | https://pytorch.org/get-started/locally/ |
| Transformers compatibility | https://huggingface.co/docs/transformers/installation |
| CUDA/GPU issues | Not applicable (CPU-only training) |
| Albumentations issues | https://github.com/albumentations-team/albumentations |

---

**Status:** ✅ Comprehensive version handling implemented  
**Next Step:** Run notebook cells starting from installation cell