# Stable Face Segmentation Snapshot (2025-11-26)

This snapshot documents the exact logic and parameters that produced the "very good" output so you can always revert to it quickly.

## Core Settings (Updated with Edge Refinement)
- Model loading order: prefer `best_model_512.pth`, fallback to `best_model.pth` (current active: `best_model.pth`).
- Transform (must match training):
  ```python
  transform = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  ```
- Face detection: Haar Cascade, permissive
  ```python
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
  ```
- Face-focused inference:
  - Detect largest face → crop with 40% padding (INCREASED to capture full hair)
  - Inference at 256×256 → resize prediction back to crop size → paste into full-size mask
  - Visualization: class-colored `label_to_color`
- Post-processing (cleaner mask): enabled by default
  ```python
  SMOOTH_MASK = True
  # Removes tiny blobs and fills small skin holes without touching other classes
  ```
- Edge refinement (NEW - for sharper boundaries):
  ```python
  REFINE_EDGES = True
  BILATERAL_D = 9
  BILATERAL_SIGMA_COLOR = 50         # Lowered for sharper edges
  BILATERAL_SIGMA_SPACE = 50         # Lowered for sharper edges
  BOUNDARY_KERNEL = 5                # Increased for wider edge zones
  # Canny thresholds: 40-120 (balanced for clean edge detection)
  # Gaussian pre-smoothing (3×3) before Canny for cleaner edges
  # Uses bilateral filtering + Canny edge detection for boundary alignment
  ```
- Ear classification fix:
  ```python
  # Geometric detection: left/right 15% zones
  # Reclassifies isolated skin blobs in ear zones to background
  # Prevents ears from being marked as facial skin
  ```
 - Elliptical ear suppression (NEW optional):
   ```python
   EAR_SUPPRESSION = True
   ELLIPSE_SCALE_X = 1.00
   ELLIPSE_SCALE_Y = 0.92   # Slight vertical shrink to avoid neck
   MIN_HEAD_CONTOUR_AREA_RATIO = 0.02
   # Removes skin outside fitted head ellipse (ears/side protrusions)
   ```

## One-Minute Restore Checklist
1. Ensure `best_model.pth` exists in the project root (or add `best_model_512.pth` to use it).
2. In `landmark_app.py`, confirm the following sections:
   - Transform block equals the snippet above.
   - `model_paths = ['best_model_512.pth', 'best_model.pth']` (order matters).
   - `detect_faces(...)` uses `scaleFactor=1.2, minNeighbors=2, minSize=(30,30)`.
   - `SMOOTH_MASK = True` near the post-processing helpers.
3. Restart server.

## Quick Server Commands (PowerShell)
```powershell
# From the project root
python landmark_app.py
```

## Revert Keywords (for team handoff)
Say: "Restore stable 256 ImageNet face-focused settings with smoothing" to re-apply the above config.

## Notes
- Using 256×256 + ImageNet normalization matches how `best_model.pth` was trained, which is why this produces the better output.
- Smoothing is conservative: it removes speckles and fills tiny skin holes but avoids overwriting eyes, lips, etc.
