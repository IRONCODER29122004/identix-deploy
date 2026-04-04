# IDENTIX Project Documentation (Consolidated)

Date: 2026-02-26
Main entrypoint: Required/landmark_app.py

## 1) Project Overview
IDENTIX is a Flask-based facial analysis system that provides:
- Facial landmark segmentation (BiSeNet, 11 classes)
- Video analysis with multi-person tracking
- Deepfake detection (rule-based + ML ensemble)
- Live webcam deepfake detection routes
- Optional MongoDB-backed authentication
- Chrome extension for live face swapping (educational use)

## 2) Entrypoints and How To Run
- Main web app: Required/landmark_app.py
  - Run: python landmark_app.py
  - UI: http://127.0.0.1:5000
- Hybrid deepfake API: Required/flask_hybrid_api.py
  - Run: python flask_hybrid_api.py
- Realtime detector demo: Required/test_realtime.py
  - Run: python test_realtime.py --mode webcam

## 3) File Inventory (Main Project)

Top-level Required/ files:
- Required/.env
- Required/best_model.pth
- Required/create_xception_model.py
- Required/deepfake_detector.py
- Required/download_models.py
- Required/faceswap_api_routes.py
- Required/flask_hybrid_api.py
- Required/gradio_interface.py
- Required/hybrid_detector.py
- Required/landmark_app.py
- Required/live_detection_routes.py
- Required/mediapipe_landmark_detector.py
- Required/ml_deepfake_detector.py
- Required/model.py
- Required/mongodb_utils.py
- Required/pipeline_runner.py
- Required/realtime_detector.py
- Required/requirements_hybrid.txt
- Required/requirements_realtime.txt
- Required/resnet.py
- Required/run_server.py
- Required/setup_hybrid.py
- Required/start_server.py
- Required/test_detector_improvements.py
- Required/test_integration.py
- Required/test_realtime.py
- Required/test_segmentation_diagnosis.py
- Required/test_server.py
- Required/test_xception_input.py
- Required/video_20241219_102330.mp4

Chrome extension (Required/chrome-extension/faceswap-live):
- Required/chrome-extension/faceswap-live/background.js
- Required/chrome-extension/faceswap-live/manifest.json
- Required/chrome-extension/faceswap-live/content/content.js
- Required/chrome-extension/faceswap-live/content/face-swapper.js
- Required/chrome-extension/faceswap-live/content/styles.css
- Required/chrome-extension/faceswap-live/content/video-processor.js
- Required/chrome-extension/faceswap-live/popup/popup.css
- Required/chrome-extension/faceswap-live/popup/popup.html
- Required/chrome-extension/faceswap-live/popup/popup.js

Templates (Required/templates):
- Required/templates/about.html
- Required/templates/blog.html
- Required/templates/careers.html
- Required/templates/contact.html
- Required/templates/cookie-policy.html
- Required/templates/gdpr.html
- Required/templates/image_analysis.html
- Required/templates/index.html
- Required/templates/landmark_index.html
- Required/templates/live_faceswap.html
- Required/templates/privacy-policy.html
- Required/templates/profile.html
- Required/templates/results.html
- Required/templates/settings.html
- Required/templates/terms-of-service.html
- Required/templates/video_analysis.html

Models (Required/models):
- Required/models/79999_iter.pth
- Required/models/best_model.keras
- Required/models/best_model.pth
- Required/models/deeplab_model.keras
- Required/models/deeplab_model_stage1.keras
- Required/models/unet_model.keras
- Required/models/unet_smoke.keras
- Required/models/vit_full.keras
- Required/models/vit_model.keras
- Required/models/vit_smoke.keras
- Required/models/xception_ff.pth

Deployment snapshot (Required/deploy/identix-deploy):
- Required/deploy/identix-deploy/.dockerignore
- Required/deploy/identix-deploy/app.py
- Required/deploy/identix-deploy/best_model.pth
- Required/deploy/identix-deploy/deepfake_detector.py
- Required/deploy/identix-deploy/Dockerfile
- Required/deploy/identix-deploy/hybrid_detector.py
- Required/deploy/identix-deploy/live_detection_routes.py
- Required/deploy/identix-deploy/ml_deepfake_detector.py
- Required/deploy/identix-deploy/mongodb_utils.py
- Required/deploy/identix-deploy/realtime_detector.py
- Required/deploy/identix-deploy/render.yaml
- Required/deploy/identix-deploy/requirements.txt
- Required/deploy/identix-deploy/models/xception_ff.pth
- Required/deploy/identix-deploy/static/favicon.svg
- Required/deploy/identix-deploy/static/chrome-extension/faceswap-live/background.js
- Required/deploy/identix-deploy/static/chrome-extension/faceswap-live/manifest.json
- Required/deploy/identix-deploy/static/chrome-extension/faceswap-live/content/content.js
- Required/deploy/identix-deploy/static/chrome-extension/faceswap-live/content/face-swapper.js
- Required/deploy/identix-deploy/static/chrome-extension/faceswap-live/content/styles.css
- Required/deploy/identix-deploy/static/chrome-extension/faceswap-live/content/video-processor.js
- Required/deploy/identix-deploy/static/chrome-extension/faceswap-live/popup/popup.css
- Required/deploy/identix-deploy/static/chrome-extension/faceswap-live/popup/popup.html
- Required/deploy/identix-deploy/static/chrome-extension/faceswap-live/popup/popup.js
- Required/deploy/identix-deploy/templates/about.html
- Required/deploy/identix-deploy/templates/blog.html
- Required/deploy/identix-deploy/templates/careers.html
- Required/deploy/identix-deploy/templates/contact.html
- Required/deploy/identix-deploy/templates/cookie-policy.html
- Required/deploy/identix-deploy/templates/deepfake_detection.html
- Required/deploy/identix-deploy/templates/gdpr.html
- Required/deploy/identix-deploy/templates/image_analysis.html
- Required/deploy/identix-deploy/templates/index.html
- Required/deploy/identix-deploy/templates/landmark_index.html
- Required/deploy/identix-deploy/templates/privacy-policy.html
- Required/deploy/identix-deploy/templates/profile.html
- Required/deploy/identix-deploy/templates/results.html
- Required/deploy/identix-deploy/templates/settings.html
- Required/deploy/identix-deploy/templates/terms-of-service.html
- Required/deploy/identix-deploy/templates/video_analysis.html

Scripts (Required/scripts):
- Required/scripts/check_checkpoint.py
- Required/scripts/clear_users.py
- Required/scripts/debug_prediction.py
- Required/scripts/inspect_checkpoint_shapes.py
- Required/scripts/list_users.py
- Required/scripts/setup_mongodb.py

Tests (Required/tests):
- Required/tests/test_all_combinations.py
- Required/tests/TEST_landmark_simple.py
- Required/tests/test_mediapipe_accuracy.py
- Required/tests/test_ping.py
- Required/tests/test_sample2.py

Dataset location (preserved):
- Required/data/datasets/train/{images,labels,landmarks}
- Required/data/datasets/test/{images,labels,landmarks}
- Required/data/datasets/val/{images,labels,landmarks}

## 4) Model Usage Check (Deploy Snapshot vs Main)
Compared the following files between Required/ and Required/deploy/identix-deploy:
- landmark_app.py, live_detection_routes.py, deepfake_detector.py, ml_deepfake_detector.py
- hybrid_detector.py, mongodb_utils.py

Result:
- Differences exist in hybrid_detector.py and mongodb_utils.py.
- Therefore the deploy snapshot folder is retained.

Key differences:
- hybrid_detector.py: deploy version adds PIL-to-OpenCV conversion and uses different ensemble weights.
- mongodb_utils.py: main version allows graceful fallback when MongoDB is unavailable.

## 5) Contributions and Components

Core model and inference:
- BiSeNet (ResNet-50) segmentation model with ImageNet normalization and 11-class output.
- Face-focused crop pipeline and full-image fallback.

Video processing:
- Multi-person detection and tracking.
- Quality scoring and best-frame selection.

Deepfake detection:
- Rule-based temporal and artifact analysis.
- ML-based Xception detector integration.
- Hybrid ensemble voting.

Live detection:
- API routes for real-time webcam analysis.
- Gradio demo interface.

Deployment:
- Port auto-detection for local, Render, and Hugging Face.
- Dockerfile and render.yaml in deploy snapshot.

Chrome extension:
- Face-swapper pipeline and UI components (educational use).

## 6) Consolidated README Content (Summarized)

Main IDENTIX README (deployment package):
- Flask backend, PyTorch BiSeNet, MongoDB Atlas auth, OpenCV + PIL.
- Endpoints: /predict, /predict_video, /detect_deepfake, /health.
- Env vars: MONGODB_URI, SECRET_KEY, FLASK_ENV, PORT.

Hugging Face deployment README:
- Use port 7860 on HF, 5000 locally; auto-detection in app.
- Link GitHub repo and set secrets for MongoDB.

Realtime detection README:
- realtime_detector.py with temporal window, frame skipping, and overlay.
- gradio_interface.py for live UI.

Hybrid ensemble README:
- Rule-based + Xception ML voting with confidence aggregation.
- Setup: setup_hybrid.py and download_models.py.

Chrome extension README:
- faceswap-live extension structure with popup and content scripts.
- Educational use with watermarking guidance.

Report submission READMEs:
- Modules: face segmentation, video segmentation, web app, deepfake detection, training.
- BiSeNet architecture, 256x256 input, 11 classes.

## 7) Consolidated Summary Documents (Summarized)

Synchronization summary:
- Feature gate threshold aligned to 0.003.
- Face detection, transforms, and gating parameters matched.

Hugging Face fix summary:
- Port auto-detection and platform detection added.
- Documentation updated for HF deployment.

UI restructure summary:
- Dedicated routes for image, video, deepfake pages.
- Single template with default_mode switch.

Improvements summary:
- Advanced preprocessing, stricter face detection, and normalized quality scores.

MediaPipe summary:
- Optional MediaPipe-based enhancement described in legacy summary (not active now).

Complete project summary:
- Detection + generation ecosystem with extension and web integration.

## 8) Notes
- Dataset files preserved under Required/data/datasets.
- Consolidated documentation replaces multiple README and summary files.
- Deploy snapshot retained due to model-usage differences.
