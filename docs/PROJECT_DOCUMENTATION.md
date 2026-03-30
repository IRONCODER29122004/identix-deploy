# Project Documentation

## 1. Overview
This project implements a facial landmark and region segmentation system using a customized BiSeNet model (ResNet-based context path) served through a Flask web application. It supports:
- Image (photo) face-focused segmentation with robust face-box candidate selection.
- Video segmentation with multi-person filtering, landmark quality scoring, best-per-landmark frame selection, and best overall frame extraction.
- Optional deepfake heuristic analysis (sequence-based consistency checks).
- Interactive recoloring of segmentation masks and visual overlays in the browser.

Primary goal: clean, face-only segmentation (skin, eyebrows, eyes, nose, lips, inner mouth, hair) with reliable selection even under poor lighting, rotation, or cluttered backgrounds.

## 2. Key Features
- BiSeNet (11-class) facial part segmentation (skin, brows, eyes, nose, upper lip, inner mouth, lower lip, hair).
- Smart face detection + candidate scoring (multi-box evaluation, gating, padding fallback).
- Video false-positive suppression (segmentation-driven validation of Haar detections).
- Landmark quality scoring (coverage, compactness, consistency, fragmentation penalty).
- Edge refinement (optional) using bilateral filtering + morphological boundary gradients.
- EXIF orientation correction for mobile portrait images.
- Best overall frame logic for video (average of landmark quality metrics) mirroring photo segmentation UX.
- Recolor endpoint for custom class color maps.
- Historical logging of changes (CHANGE_LOG.md) with reversible diffs.

## 3. Architecture Overview
- `landmark_app.py`: Main Flask application (routes, pipelines, detection, segmentation, video logic, deepfake analysis wrapper).
- `model.py` + `resnet.py`: Alternate BiSeNet variant (training reference). Production runtime currently constructs a BiSeNet inline in `landmark_app.py` (ResNet-50 backbone version).
- `templates/*.html`: Frontend pages (image analysis, video analysis, etc.).
- `best_model.pth`, `best_model_512.pth`: Model weight checkpoints (prefers 512 fine-tuned if available; falls back gracefully).
- `CHANGE_LOG.md`: Chronological log of code/logic adjustments.

## 4. Models
### 4.1 BiSeNet Variant (Runtime)
Implemented in `landmark_app.py`:
- Context Path: ResNet-50 (layers conv1 → layer4) feeding attention refinement modules at 1/16 and 1/32.
- Spatial Path: Lightweight 3-stage conv chain producing 1/8 spatial feature.
- Feature Fusion Module: Channel-wise attention combining spatial + context features.
- Outputs: Main + two auxiliary heads for training (aux disabled at inference).

### 4.2 Alternate BiSeNet (model.py)
- Uses Resnet18 backbone (custom `Resnet18` class).
- SpatialPath replaced by reusing intermediate ResNet feature (`feat_res8`).
- FeatureFusionModule merges feature maps; three output heads (main/aux16/aux32).
- Provided for experimentation/training; not directly used in live server.

## 5. Segmentation Classes (Indices)
```
0: Background
1: Skin
2: Left Eyebrow
3: Right Eyebrow
4: Left Eye
5: Right Eye
6: Nose
7: Upper Lip
8: Inner Mouth
9: Lower Lip
10: Hair
```

## 6. Image (Photo) Processing Pipeline
1. Receive upload (`/predict` route).
2. Load image → `ImageOps.exif_transpose` for orientation.
3. Face detection (Haar, permissive). If faces found → evaluate candidates:
   - Filter by area ratio, aspect ratio.
   - Score by coverage (capped), feature presence bonus, center penalty.
   - Gate improbable masks (non-background ratio, minimal facial feature ratio).
   - Retry top candidates with enlarged padding if initial segmentation fails gating.
   - Fallback: largest candidate if all else fails.
4. Crop face region with dynamic padding (default 40%).
5. Resize to 256×256 (or 512 variant if using alternative weights) + ImageNet normalization.
6. Run BiSeNet; get prediction mask.
7. Resize mask back to crop size; integrate into full-size canvas.
8. Optional smoothing & edge refinement (`SMOOTH_MASK`, `REFINE_EDGES`).
9. Ear heuristic applied (moderate—Change 24 baseline) for side zone corrections.
10. Generate colored mask, overlay, visualization, raw grayscale mask.
11. Return JSON with base64 assets + stats.

## 7. Video Processing Pipeline (Post Change 31)
1. Uniform frame sampling (`extract_frames`).
2. Haar detection per frame (permissive) → each detection validated by `validate_face_detection`:
   - Quick segmentation of cropped region.
   - Compute feature ratio (eyes/nose/mouth classes) and skin ratio.
   - Accept only detections with feature_ratio ≥ 0.5% AND skin 5–50%.
   - Compute quality proxy: feature_ratio * 100 + skin component.
3. Track faces via IoU matching (`get_face_id`).
4. Filter tracked faces:
   - Appearances ≥ 3.
   - Average quality ≥ 1.0.
5. Main character = max(screen_time × avg_quality).
6. For each main-character appearance frame:
   - Full landmark segmentation & scoring per landmark (coverage + shape metrics).
   - Aggregate total landmark score → average frame quality.
7. Best per landmark frame chosen by max quality.
8. Best overall frame chosen by max average landmark quality (mirrors image experience).
9. Other faces shown only if screen time ≥10% of main’s.
10. Serialize `best_frames` + `best_overall` with base64 images.

## 8. Landmark Quality Scoring (Video & Frame Analytics)
`calculate_landmark_quality` combines:
- Coverage vs expected baseline (class-specific normalization).
- Shape compactness (4πA / P²).
- Consistency (largest contour coverage ratio).
- Fragmentation penalty (# of contours beyond first).
Result clamped 0–100.

Overall frame quality = average of 10 landmark scores.

## 9. Face Candidate Scoring (Image Mode)
Metrics:
- Coverage (capped at 40%).
- Feature bonus if any of eyes/nose/mouth classes appear.
- Center penalty biases away from extreme corners.
- Early accept threshold (score ≥ 0.06) for efficiency.
Gating rejects masks with unrealistic non-background or missing facial structures.

## 10. Edge Refinement (Optional)
Parameters:
- Bilateral filter (`BILATERAL_D`, `BILATERAL_SIGMA_COLOR`, `BILATERAL_SIGMA_SPACE`).
- Morphological gradient (`BOUNDARY_KERNEL`).
Purpose: Sharpen transitions at lips/eyes/hair boundaries without distorting class regions.

## 11. Ear Classification Heuristic (Post Revert: Change 28)
Simplified approach:
- Side zone = outer 20% width.
- Vertical band 15–85% height.
- Remove small skin-like blobs meeting area + aspect constraints to avoid ears mis-tagged as facial skin.
Advanced ellipse/dynamic margin methods removed due to over-aggression.

## 12. API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Photo segmentation (face-focused) |
| `/recolor` | POST | Apply custom color map to existing mask |
| `/predict_video` | POST | Video multi-person landmark segmentation |
| `/detect_deepfake` | POST | Run deepfake heuristic analysis |
| `/video-analysis` | GET | Video analysis UI |
| `/facial-landmarks` | GET | Image analysis UI |
| `/health` | GET | Health check (model/device) |
| `/login` / `/register` | POST | Simple demo auth |
| `/history` | GET | User upload history |

## 13. Response Data (Selected Fields)
### `/predict`
```
{
  success: true,
  original, prediction, visualization, overlay, mask,
  image_size: (W,H),
  face_detected: bool,
  num_landmarks: 11,
  landmark_counts: {class_index: pixel_count},
  landmark_names: {...}
}
```
### `/predict_video`
```
{
  success: true,
  main_character: {
    screen_time,
    screen_time_percent,
    best_overall: { frame_number, quality_score, original, prediction, overlay },
    landmarks: {
      "1": { landmark_name, frame_number, quality_score, original, prediction, overlay, highlight },
      ...
    }
  },
  other_faces: [ { face_id, screen_time, screen_time_percent, frame_number, original, overlay } ],
  processed_frames,
  total_frames,
  total_people
}
```

## 14. Configuration & Tunable Parameters
| Name | Purpose | Typical Range |
|------|---------|---------------|
| `SMOOTH_MASK` | Enable mild mask cleaning | bool |
| `SMALL_COMPONENT_AREA_RATIO` | Remove tiny blobs | 3e-4 – 8e-4 |
| `REFINE_EDGES` | Edge sharpening | bool |
| `BILATERAL_*` | Edge-preserving smoothing | Adjust for sharpness vs smooth |
| `BOUNDARY_KERNEL` | Edge thickness in refinement | 3–7 |
| Video: `MIN_SCREEN_TIME` | Min frames to keep face | 2–5 |
| Video: `MIN_AVG_QUALITY` | Quality threshold | 0.8–2.0 |
| Video: `MIN_OTHER_SCREEN_TIME_RATIO` | Secondary faces threshold | 0.05–0.15 |
| Image candidate area ratio | Filter tiny detections | 0.001–0.002 |
| Image aspect gating | Face aspect plausibility | 0.6–1.6 |

## 15. File Inventory (Key)
- `landmark_app.py`: Endpoints + runtime segmentation logic.
- `model.py`: Alternate BiSeNet (ResNet18) for experimentation.
- `resnet.py`: Backbone feature extraction (ResNet18 variant).
- `CHANGE_LOG.md`: Granular modifications record.
- `templates/video_analysis.html`: Frontend video UI with best overall frame card.
- Model weight files: `best_model.pth`, `best_model_512.pth`.

## 16. Change Log Summary (Recent)
- Change 28: Revert advanced ear suppression to stable heuristic.
- Change 29: EXIF orientation fix (portrait images upright).
- Change 30: Robust face candidate selection (multi-box scoring, gating, fallback).
- Change 31: Video false-positive filtering + best overall frame + quality-driven selection.
See `CHANGE_LOG.md` for full reversible specifics.

## 17. Performance Considerations
- Haar detection is permissive; filtering relies on fast segmentation validation—ensure GPU/optimized torch backend for throughput.
- Frame sampling uniform; increase `max_frames` for higher temporal fidelity (trade-off CPU/GPU cost).
- Bilateral filtering incurs cost; disable `REFINE_EDGES` for speed-critical batch runs.

## 18. Troubleshooting
| Symptom | Likely Cause | Mitigation |
|---------|--------------|-----------|
| All video faces removed | Threshold too strict | Lower `MIN_AVG_QUALITY` or `MIN_SCREEN_TIME` |
| Too many other faces | Lower validation strictness accidentally | Raise `MIN_AVG_QUALITY`; increase feature_ratio requirement |
| Mask misaligned | Wrong candidate chosen | Verify gating thresholds; adjust center penalty |
| Portrait rotated | EXIF not applied | Ensure `ImageOps.exif_transpose` present |
| Lips/hair fuzzy | Edge refinement off | Enable `REFINE_EDGES` or tune bilateral params |

## 19. Security / Privacy Notes
- Demo-level auth; not production hardened.
- No persistence beyond in-memory user history.
- Large uploads allowed (500MB) — consider rate limiting in production.

## 20. Possible Extensions
- GPU batched frame segmentation for video speed-up.
- Integration with tracking (e.g., SORT/Deep SORT) for more stable IDs.
- Confidence-based dynamic frame skipping.
- Export segmentation as vector paths (SVG) for downstream design workflows.
- Extend deepfake module with temporal consistency ML model.

## 21. Glossary
- Coverage: Pixel proportion vs expected anatomical class size.
- Compactness: Shape circularity metric (4πA/P²).
- Fragmentation: Penalization for multiple disjoint contours.
- Gating: Early rejection of implausible candidate face crops.

## 22. Deployment Notes
- Designed for local dev (`debug=True`). For production: set `SECRET_KEY`, disable debug, add WSGI (gunicorn) and reverse proxy.
- Ensure consistent model file path; fallback warns if model absent.

## 23. License / Attribution
- ResNet & torchvision components under their respective open licenses.
- BiSeNet architectural inspiration from original paper; this variant adapted for face-part segmentation.

## 24. Quick Start
```bash
# Install deps (example)
pip install flask torch torchvision opencv-python pillow numpy

# Run server
python landmark_app.py
# Visit
http://localhost:5000/facial-landmarks
http://localhost:5000/video-analysis
```

## 25. Minimal Integration Example (Programmatic)
```python
from PIL import Image
import requests

img = Image.open('face.jpg')
# POST to /predict using requests
# (multipart form: key 'image')
```
(Use browser UI for full visualization.)

## 26. Recommended Parameter Adjustments
| Scenario | Adjustment |
|----------|-----------|
| Backlit face | Increase padding retry or relax feature_ratio gating slightly |
| Many false video faces | Raise `MIN_AVG_QUALITY` to ≥1.5 |
| Too few other faces | Lower `MIN_OTHER_SCREEN_TIME_RATIO` to 0.05 |
| Performance constraints | Disable `REFINE_EDGES`; lower `max_frames` |
| Shallow masks | Double-check normalization (ImageNet mean/std) |

## 27. Validation Checklist
- EXIF orientation intact? (Upload portrait test image)
- Best overall frame appears? (Video with multiple frames)
- False positives suppressed? (Background objects not listed)
- Landmark colors consistent? (Recolor endpoint functioning)
- No KeyError for `quality_score`? (Video main_character landmarks)

---
**Status**: Documentation reflects state after Change 31 (2025-11-26). Update this file if new pipeline variants or thresholds are introduced.
