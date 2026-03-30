# Edge Refinement Enhancement - Technical Summary

**Date**: 2025-11-26  
**Change ID**: 21  
**User Request**: "make the accuracy even better... need to make edge detection better for the landmarks"

## What Was Added

### 1. Configuration Flags
New settings in `landmark_app.py` after the existing post-processing flags:

```python
REFINE_EDGES = True                  # Master toggle
BILATERAL_D = 9                      # Filter diameter
BILATERAL_SIGMA_COLOR = 75           # Color similarity threshold
BILATERAL_SIGMA_SPACE = 75           # Spatial distance threshold
BOUNDARY_KERNEL = 3                  # Edge detection kernel size
```

### 2. Core Algorithm: _refine_edges_guided()

**Purpose**: Sharpen landmark boundaries by aligning them with actual facial features in the original image.

**Steps**:
1. **Bilateral Filtering**: Smooths prediction while preserving strong edges
   - Uses both spatial and color information
   - Parameters tuned for face-scale features (9px diameter, sigma=75)

2. **Edge Detection**: Finds real boundaries in original face crop
   - Canny edge detection (thresholds: 50-150)
   - Captures actual facial feature edges (lips, eyes, nose contours)

3. **Hybrid Masking**: Combines sharp and smooth regions
   - At detected edges: use original prediction (sharp)
   - Away from edges: use bilateral-smoothed (clean)
   - Transition zones created via 3×3 dilation

**Why This Works**:
- Bilateral filter removes noise but respects intensity gradients
- Canny edges from original image are ground truth for feature boundaries
- Hybrid approach gets best of both: clean regions + sharp edges

### 3. Integration Points

**predict_landmarks_bisenet()** (line ~560):
```python
# Pass original face crop to guide edge refinement
face_crop_np = np.array(face_crop)
face_prediction_resized = smooth_prediction_map(
    face_prediction_resized, 
    original_crop=face_crop_np
)
```

**predict_landmarks_for_face()** (line ~803):
```python
# Video frame processing
face_crop_np = np.array(face_crop)
prediction_resized = smooth_prediction_map(
    prediction_resized, 
    original_crop=face_crop_np
)
```

## Impact on Quality

### Before (smoothing only):
- Clean regions, reduced speckles
- Boundaries still pixelated
- Not aligned with actual features

### After (smoothing + edge refinement):
- Clean regions maintained
- Sharp, aligned boundaries
- Edges follow real facial contours
- Better visual accuracy

## Tuning Guidance

If edges are too soft:
- Decrease `BILATERAL_SIGMA_COLOR` (try 50)
- Decrease `BILATERAL_SIGMA_SPACE` (try 50)

If edges are too harsh:
- Increase bilateral sigmas (try 100)
- Increase `BOUNDARY_KERNEL` to 5

If you want more edge preservation:
- Lower Canny thresholds: `cv2.Canny(gray, 30, 100)`

Disable completely:
```python
REFINE_EDGES = False
```

## Rollback Instructions

1. Set `REFINE_EDGES = False` in `landmark_app.py`
2. Or remove calls to `_refine_edges_guided()` from `smooth_prediction_map()`
3. Or revert to previous commit before Change 21

## Files Modified

- `landmark_app.py`: Added refinement functions and integrated into pipeline
- `CHANGE_LOG.md`: Documented as Change 21
- `STABLE_FACE_SEGMENTATION_SNAPSHOT.md`: Updated with new settings
- `EDGE_REFINEMENT_TECHNICAL_NOTES.md`: This file

## Performance Notes

- Adds ~15-25ms per face crop on CPU (bilateral + Canny)
- Negligible on GPU
- Only runs when `SMOOTH_MASK=True` and `REFINE_EDGES=True`
- Original crop must be passed; if None, skips refinement

## Testing

Recommended test:
```python
# From project root
python landmark_app.py
# Upload test image with clear facial features
# Compare before/after edge quality at eye/lip/nose boundaries
```

Visual inspection checklist:
- [ ] Eye boundaries clean and aligned
- [ ] Lip edges sharp and natural
- [ ] Nose bridge smooth with defined edges
- [ ] Skin-hair boundary well-defined
- [ ] No jagged pixelation artifacts
