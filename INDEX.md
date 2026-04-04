# IDENTIX Complete Project Index

**Last Updated**: March 25, 2026  
**Status**: ✅ Reorganization Complete & v9 Kaggle FFA-MPDV Pipeline Added ✨

---

## Quick Start (5 Minutes)

```bash
cd Required
python landmark_app.py
# Open http://localhost:5000 in browser
```

---

## Main Documentation Files (Read These First)

1. **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Start here! Comprehensive guide
2. **[REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md)** - What was reorganized
3. **[Required/README_PROJECT_STRUCTURE.md](Required/README_PROJECT_STRUCTURE.md)** - Detailed structure
4. **[Required/README_V2_MODEL.md](Required/README_V2_MODEL.md)** - ✨ **NEW** Deepfake Detector v2 Quick Start
5. **[Waste/README_WASTE.md](Waste/README_WASTE.md)** - What's archived and why

---

## Directory Tree

```
Code_try_1/
│
├── [DOCUMENTATION]
│   ├── PROJECT_OVERVIEW.md               ← START HERE!
│   ├── REORGANIZATION_SUMMARY.md
│   └── This file (INDEX.md)
│
├── Required/                             ← MAIN PROJECT
│   ├── [8 Core Python Files]
│   │   ├── landmark_app.py              ✅ Main facial landmark detection
│   │   ├── app.py                       ✅ Video segmentation pipeline
│   │   ├── deepfake_detector.py         ✅ Deepfake analysis
│   │   ├── pipeline_runner.py
│   │   ├── mongodb_utils.py
│   │   ├── model.py
│   │   ├── resnet.py
│   │   └── mediapipe_landmark_detector.py
│   │
│   ├── .env                              🔐 Configuration (SECRET)
│   ├── README_PROJECT_STRUCTURE.md
│   │
│   ├── models/                           [10 pre-trained models]
│   │   ├── best_model.pth               ⭐ Main BiSeNet model
│   │   ├── best_model_512.pth
│   │   ├── unet_model.keras
│   │   ├── unet_smoke.keras
│   │   ├── deeplab_model.keras
│   │   ├── deeplab_model_stage1.keras
│   │   ├── vit_model.keras
│   │   ├── vit_full.keras
│   │   ├── vit_smoke.keras
│   │   └── 79999_iter.pth
│   │
│   ├── data/                             [Runtime: auto-created]
│   │   ├── uploads/                     User-uploaded files
│   │   ├── pipelines_frames/            Extracted video frames
│   │   └── pipelines_crops/             Cropped face regions
│   │
│   ├── templates/                        [16 HTML UI files]
│   │   ├── index.html                   Landing page
│   │   ├── landmark_index.html          Landmark detection UI
│   │   ├── deepfake_detection.html      Deepfake analysis UI
│   │   ├── image_analysis.html
│   │   ├── video_analysis.html
│   │   ├── results.html
│   │   ├── profile.html
│   │   ├── settings.html
│   │   ├── about.html
│   │   ├── contact.html
│   │   ├── privacy-policy.html
│   │   ├── terms-of-service.html
│   │   ├── gdpr.html
│   │   ├── cookie-policy.html
│   │   ├── blog.html
│   │   └── careers.html
│   │
│   ├── static/                           [Static assets]
│   │   └── js/
│   │       └── deepfake_frontend.js
│   │
│   ├── docs/                             [23 documentation files]
│   │   ├── README.md
│   │   ├── DEEPFAKE_README.md
│   │   ├── LANDMARK_README.md
│   │   ├── PROJECT_DOCUMENTATION.md
│   │   ├── QUICK_REFERENCE.md
│   │   ├── DEPLOYMENT.md
│   │   ├── CODE_REVIEW.md
│   │   ├── MEDIAPIPE_UPGRADE_GUIDE.md
│   │   └── [15+ more documentation files]
│   │
│   ├── scripts/                          [4 utility scripts]
│   │   ├── setup_mongodb.py
│   │   ├── clear_users.py
│   │   ├── list_users.py
│   │   └── debug_prediction.py
│   │
│   ├── tests/                            [5 test scripts]
│   │   ├── TEST_landmark_simple.py
│   │   ├── test_mediapipe_accuracy.py
│   │   ├── test_all_combinations.py
│   │   ├── test_ping.py
│   │   └── test_sample2.py
│   │
│   └── deploy/                           [Deployment package]
│       └── identix-deploy/              GitHub/Render ready
│           ├── app.py                   Deployment app
│           ├── requirements.txt
│           ├── render.yaml
│           ├── deepfake_detector.py
│           ├── mongodb_utils.py
│           ├── best_model.pth
│           ├── templates/
│           └── DEPLOYMENT.md
│
└── Waste/                                ← ARCHIVED FILES
    ├── README_WASTE.md
    │
    ├── archives/                        [Backup files]
    │   └── landmark_app_OLD_BACKUP.py
    │
    ├── assets/                          [Test samples]
    │   ├── combo_test_*.png             Model comparison images
    │   ├── output_mediapipe_*.jpg       MediaPipe outputs
    │   ├── test_output_*.png            Test results
    │   ├── training_history.png
    │   └── Sample2.mp4
    │
    ├── notebooks/                       [10+ Jupyter notebooks]
    │   ├── collab_notebook.ipynb
    │   ├── face_segmentation_demo.ipynb
    │   ├── main.ipynb, main2.ipynb, main3.ipynb, main4.ipynb
    │   ├── model_test.ipynb
    │   └── [6+ more notebooks]
    │
    ├── Facial_Landmark_Project/        [Old structure copy]
    │   ├── app/
    │   ├── web_app/
    │   ├── models/
    │   └── docs/
    │
    ├── Report_Submission/              [Project submission]
    │   ├── 1_Face_Segmentation/
    │   ├── 2_Video_Segmentation/
    │   ├── 3_Website/
    │   ├── 4_Deepfake_Detection/
    │   └── 5_Deep_Learning/
    │
    ├── Report_Submission.zip           [Compressed archive]
    │
    ├── cloudinary_backend/             [Legacy cloud integration]
    │
    ├── train/, test/, val/             [Training data splits]
    │   └── images/, labels/, landmarks/
    │
    └── __pycache__/                    [Python cache]
```

---

## What Each Main App Does

### 1. **landmark_app.py** ⭐ RECOMMENDED
**Purpose**: Facial landmark detection and segmentation
- Upload image/video
- Detect faces
- Segment 11 facial landmark classes
- Display results with visualization
- Save segmentation masks

**Run**:
```bash
python landmark_app.py
```

**Access**: http://localhost:5000/landmark_index.html

**Features**:
- Real-time face detection
- Multi-face support
- Webcam support
- Download segmentation masks
- Upload history tracking

---

### 2. **app.py**
**Purpose**: Video segmentation pipeline
- Extract frames from videos
- Detect landmarks in each frame
- Crop best face regions
- Run inference on crops
- Generate comparisons

**Run**:
```bash
python app.py
```

**Access**: http://localhost:5000

**Features**:
- Batch video processing
- Frame extraction control
- Multi-model inference
- Crop management

---

### 3. **deepfake_detector.py**
**Purpose**: Deepfake detection and analysis
- Analyze video for manipulation
- Extract features
- Score authenticity
- Generate report

**Integrated into**: Both apps above

**Features**:
- Temporal consistency check
- Boundary artifact detection
- Blink pattern analysis
- Landmark stability metrics

---

## Technology Stack
### 3b. **deepfake_detector_v2.py** ✨ **NEW**
**Purpose**: Deepfake detection v2 (Neural Network FFA-MPDV)
- Deep learning based detection using trained FFA-MPDV model
- Meso4 backbone + FPN + Capsule routing + Spatial attention
- Multi-scale feature fusion (92.1% ROC-AUC on validation)
- Trained via Kaggle notebook (50 epochs, paper-baseline)

**Usage**: Independent or alongside v1 for comparison

**Features**:
- Single or batch image prediction
- Confidence scores and logit outputs
- Automatic preprocessing (256x256, paper normalization)
- GPU/CPU support (50-100ms per image CPU, 5-10ms GPU)
- High-level wrapper class with clean API

**Performance**:
- ROC-AUC: 92.1% | Precision: 88.6% | Recall: 76.5% | F1: 82.1%

**Quick Start**:
```bash
python test_v2_quick.py  # Verify installation (3/3 tests)
```

**Documentation**: 
- Quick start: [README_V2_MODEL.md](Required/README_V2_MODEL.md)
- Full guide: [DEEPFAKE_DETECTOR_V2_GUIDE.md](Required/docs/DEEPFAKE_DETECTOR_V2_GUIDE.md)
- Reference card: [V2_QUICK_REFERENCE.md](Required/docs/V2_QUICK_REFERENCE.md)
- Checklist: [V2_COMPLETION_CHECKLIST.md](Required/docs/V2_COMPLETION_CHECKLIST.md)

**Status**: ✅ Production Ready (All tests passing 3/3)

---

### 3c. **deepfake_detector_v9_kaggle_ffa_mpdv.py** ✨ **NEW**
**Purpose**: Deepfake detection v9 (Kaggle FFA-MPDV paper-baseline checkpoint)
- Separate integration for Kaggle-exported checkpoint
- Professor-faithful architecture path (Meso4Professor + FPN + Spatial Attention + Capsules)
- Lightweight TTA at inference for stability
- Integrated into `landmark_app.py` as model key `ffa_mpdv_kaggle_v9`

**Assets**:
- Checkpoint: `Required/models/ffa_mpdv_v9_kaggle_paper_baseline.pth`
- Notebook source snapshot: `Required/notebooks/kaggle_versions/kaggle_v9_source.ipynb`

**Separate Pipeline**:
- `Required/pipelines/v9_kaggle_ffa_mpdv/README.md`
- `Required/pipelines/v9_kaggle_ffa_mpdv/tune_v9_thresholds.py`
- `Required/pipelines/v9_kaggle_ffa_mpdv/tune_v9_quick.py`
- `Required/pipelines/v9_kaggle_ffa_mpdv/compare_v9_against_existing.py`

**Notebook-Observed Performance**:
- ROC AUC: ~0.95
- PR AP: ~0.96

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Flask | 2.3.3 |
| Model Framework | PyTorch | 2.5.1 |
| Vision | TorchVision | 0.20.1 |
| Computer Vision | OpenCV | 4.8.1 |
| Database | MongoDB | (Atlas) |
| Deep Learning | TensorFlow/Keras | 2.x |
| Frontend | HTML/CSS/JavaScript | - |

---

## Key Files Quick Reference

| Need | File |
|------|------|
| Start app | `Required/landmark_app.py` |
| Deepfake detection v1 | `Required/deepfake_detector.py` |
| Deepfake detection v2 ✨ | `Required/deepfake_detector_v2.py` |
| v2 Quick test ✨ | `Required/test_v2_quick.py` |
| v2 Batch pipeline ✨ | `Required/test_pipeline_v2.py` |
| Video pipeline | `Required/app.py` |
| Configuration | `Required/.env` |
| Models | `Required/models/*.pth` or `*.keras` |
| UI/Templates | `Required/templates/*.html` |
| Database | `Required/mongodb_utils.py` |
| Tests | `Required/tests/*.py` |
| Deployment | `Required/deploy/identix-deploy/` |
| Documentation | `Required/docs/` |

---

## Common Tasks

### Task: Run Facial Landmark Detection
```bash
cd Required
python landmark_app.py
# Visit http://localhost:5000/landmark_index.html
```

### Task: Analyze Video for Deepfakes (v1 - Landmark-based)
```bash
# Use landmark_app.py
# Go to http://localhost:5000/deepfake_detection.html
# Upload video and click "Analyze"
```

### Task: Test Deepfake Detector v2 ✨
```bash
cd Required
# Verify v2 installation
python test_v2_quick.py              # Should show 3/3 tests passing ✅

# Process video crops with v2
python test_pipeline_v2.py data/pipelines_crops results_v2
# Generates: overlays/, results.json, REPORT.txt

# Python API
from deepfake_detector_v2 import load_v2_model
detector = load_v2_model('models/deepfake_detector_v2_ffa_mpdv.pth')
result = detector.predict('image.jpg')
print(f"{result['label_name']}: {result['confidence']:.0%}")
```

### Task: Process Video Pipeline
```bash
cd Required
python app.py
# Visit http://localhost:5000
# Upload video
```

### Task: Deploy to Cloud
```bash
cd Required/deploy/identix-deploy/
# Follow DEPLOYMENT.md
# Push to GitHub
# Deploy to Render
```

### Task: Review Old Code
```bash
cd Waste/Facial_Landmark_Project/
# Or Waste/Report_Submission/
```

### Task: Run Tests
```bash
cd Required/tests/
python TEST_landmark_simple.py
python test_mediapipe_accuracy.py
```

---

## API Endpoints Summary

```
GET  /                          Landing page
GET  /landmark_index.html       Landmark detection UI
POST /api/predict_image        Detect landmarks in image
POST /api/predict_video        Process video frames
POST /detect_deepfake          Analyze video for deepfakes
GET  /api/history              Get user history
POST /upload                   Upload to pipeline
GET  /results/<filename>       Get pipeline results
GET  /status/<filename>        Check processing status
GET  /health                   Health check
```

---

## Environment Setup

### 1. Install Dependencies
```bash
pip install -r Required/deploy/identix-deploy/requirements.txt
```

### 2. Create .env
```bash
cp Required/.env Required/.env.example
# Then edit Required/.env
```

### 3. Configure Environment
```
MONGODB_URI=your_mongodb_uri_here
SECRET_KEY=your_secret_key_here
FLASK_ENV=development
PORT=5000
```

### 4. Run Application
```bash
cd Required
python landmark_app.py
```

---

## Performance Metrics

| Operation | Time |
|-----------|------|
| Load models | 2-3 seconds |
| Predict image | 50-200ms (GPU/CPU) |
| Extract video frames | 1 FPS |
| Deepfake analysis | 1-2s/frame |
| Web response | <500ms |

---

## Documentation Index

**For beginners**:
1. PROJECT_OVERVIEW.md
2. Required/README_PROJECT_STRUCTURE.md
3. Required/docs/QUICK_REFERENCE.md

**For developers**:
**For v2 Deepfake Detector ✨**:
1. Required/README_V2_MODEL.md (Quick start - 5 min)
2. Required/docs/V2_QUICK_REFERENCE.md (Cheat sheet)
3. Required/docs/DEEPFAKE_DETECTOR_V2_GUIDE.md (Full guide)
4. Required/docs/V2_COMPLETION_CHECKLIST.md (Verification)

**For developers**:
1. Required/docs/PROJECT_DOCUMENTATION.md
2. Required/deploy/identix-deploy/CODE_REVIEW.md
3. Required/docs/LANDMARK_README.md

**For deployment**:
1. Required/deploy/identix-deploy/DEPLOYMENT.md
2. Required/docs/MEDIAPIPE_UPGRADE_GUIDE.md

**For troubleshooting**:
1. PROJECT_OVERVIEW.md (Troubleshooting section)
2. Required/docs/QUICK_REFERENCE.md

---

## Help & Support

**For Setup Issues**:
- See: PROJECT_OVERVIEW.md → Troubleshooting

**For API Questions**:
- See: Required/docs/DEEPFAKE_README.md
- See: Required/docs/LANDMARK_README.md

**For Deployment**:
- See: Required/deploy/identix-deploy/DEPLOYMENT.md

**For Architecture**:
- See: Required/docs/PROJECT_DOCUMENTATION.md

**For History**:
- See: Waste/README_WASTE.md

---

## Project Status

✅ **Complete & Working**
- Facial landmark detection (tested)
- Deepfake detection (integrated)
- Video processing pipeline
- Database integration
- Web UI with all features
- Full documentation
- Deployment package

🚀 **Ready for**
- Production deployment
- Active development
- Team collaboration
- Client demos

---

## Next Steps

1. **Read** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
2. **Run** `cd Required && python landmark_app.py`
3. **Visit** http://localhost:5000
4. **Upload** test image or video
5. **Explore** the features

---

## File Organization Statistics

| Metric | Count |
|--------|-------|
| Python Core Files | 8 |
| Pre-trained Models | 10 |
| HTML Templates | 16 |
| Documentation Files | 23+ |
| Test Scripts | 5 |
| Total Files Organized | 150+ |

---

**Last Updated**: January 31, 2026  
**Status**: ✅ Complete & Verified  
**Ready for**: Development & Deployment

