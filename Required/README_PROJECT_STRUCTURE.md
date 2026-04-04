# IDENTIX Project Structure (Reorganized)

## Overview
This is the main production-ready project directory for IDENTIX. All files have been organized for optimal workflow and clarity.

## Directory Structure

```
Required/
├── app.py                          # Flask app for video segmentation pipeline
├── landmark_app.py                 # Facial landmark detection (BiSeNet-based)
├── deepfake_detector.py            # Deepfake detection analysis module
├── mongodb_utils.py                # MongoDB connection utilities
├── pipeline_runner.py              # Video processing pipeline orchestrator
├── mediapipe_landmark_detector.py  # MediaPipe integration (stub)
├── model.py                        # Segmentation model definitions
├── resnet.py                       # ResNet architecture components
├── .env                            # Environment configuration (SECRET)
│
├── models/                         # Pre-trained models
│   ├── best_model.pth             # BiSeNet landmark detector (trained)
│   ├── best_model_512.pth         # BiSeNet 512x512 fine-tuned version
│   ├── 79999_iter.pth             # Alternative checkpoint
│   ├── unet_model.keras           # UNet segmentation model
│   ├── unet_smoke.keras           # UNet lightweight version
│   ├── deeplab_model.keras        # DeepLab v3+ model
│   ├── deeplab_model_stage1.keras # DeepLab stage 1 checkpoint
│   ├── vit_model.keras            # Vision Transformer model
│   ├── vit_full.keras             # ViT full version
│   └── vit_smoke.keras            # ViT lightweight version
│
├── data/                           # Runtime data directories
│   ├── uploads/                   # User-uploaded images/videos
│   ├── pipelines_frames/          # Extracted video frames
│   └── pipelines_crops/           # Cropped face regions
│
├── templates/                      # HTML templates (Flask)
│   ├── index.html                 # Landing page
│   ├── landmark_index.html        # Landmark detection page
│   ├── deepfake_detection.html    # Deepfake analysis page
│   ├── image_analysis.html        # Image segmentation page
│   ├── video_analysis.html        # Video segmentation page
│   ├── profile.html               # User profile page
│   ├── results.html               # Analysis results display
│   ├── settings.html              # User settings
│   ├── about.html                 # About page
│   ├── contact.html               # Contact form
│   ├── privacy-policy.html        # Privacy documentation
│   ├── terms-of-service.html      # Terms of service
│   ├── gdpr.html                  # GDPR compliance
│   ├── cookie-policy.html         # Cookie policy
│   ├── blog.html                  # Blog/documentation
│   └── careers.html               # Careers page
│
├── static/                         # Static assets
│   └── js/
│       └── deepfake_frontend.js   # Frontend deepfake detection UI
│
├── docs/                           # Documentation
│   ├── README.md                  # Main project readme
│   ├── DEPLOYMENT.md              # Deployment guide
│   ├── CODE_REVIEW.md             # Code audit & review
│   ├── DEEPFAKE_README.md         # Deepfake detection docs
│   ├── LANDMARK_README.md         # Landmark detection docs
│   ├── PROJECT_DOCUMENTATION.md   # Comprehensive docs
│   ├── MEDIAPIPE_UPGRADE_GUIDE.md # MediaPipe migration guide
│   ├── QUICK_REFERENCE.md         # Quick start guide
│   └── [Other documentation]      # Additional technical docs
│
├── scripts/                        # Utility scripts
│   ├── clear_users.py             # MongoDB user cleanup
│   ├── list_users.py              # List database users
│   ├── setup_mongodb.py           # MongoDB initialization
│   └── debug_prediction.py        # Debug predictions
│
├── tests/                          # Test scripts
│   ├── test_landmark_simple.py    # Basic landmark detection test
│   ├── test_mediapipe_accuracy.py # MediaPipe validation
│   ├── test_all_combinations.py   # Comprehensive testing
│   └── test_sample2.py            # Sample video testing
│
├── deploy/                         # Deployment package
│   └── identix-deploy/            # GitHub/HuggingFace deployment
│       ├── app.py                 # Deployment-optimized app
│       ├── requirements.txt       # Python dependencies
│       ├── render.yaml            # Render platform config
│       ├── deepfake_detector.py   # Deepfake module
│       ├── mongodb_utils.py       # DB utilities
│       ├── best_model.pth         # Deployed model
│       ├── templates/             # Deployment templates
│       └── [Deployment files]     # Other deployment assets
│
└── README_PROJECT_STRUCTURE.md     # This file
```

## What Each File Does

### Core Application
- **app.py** - Flask server for video segmentation pipeline. Handles uploads, frame extraction, and inference.
- **landmark_app.py** - Main facial landmark detection app using BiSeNet model. Multi-face detection & analysis.
- **deepfake_detector.py** - Advanced deepfake detection using temporal, boundary, and physiological analysis.
- **pipeline_runner.py** - Orchestrates video processing: extraction → landmark detection → segmentation.

### Models & Architecture
- **model.py** - Defines segmentation model architectures.
- **resnet.py** - ResNet backbone components used in BiSeNet.

### Database & Utilities
- **mongodb_utils.py** - MongoDB connection pooling and database management.

### Configuration
- **.env** - Environment variables (MongoDB URI, secret keys, etc.) - NEVER commit this file.

## Key Subdirectories

### models/
Contains pre-trained weights for:
- **BiSeNet** (best_model.pth) - Primary facial landmark detector
- **UNet** - Semantic segmentation alternative
- **DeepLab** - Advanced segmentation model
- **ViT** - Vision Transformer-based approach

### data/
Runtime directories (auto-created):
- uploads/ - User uploads
- pipelines_frames/ - Video frame extraction
- pipelines_crops/ - Processed face crops

### templates/
All Flask HTML templates organized by feature:
- Landmark detection UI
- Deepfake analysis UI
- Results display
- User management pages

### docs/
Complete technical documentation:
- Deployment guides
- API documentation
- Architecture explanations
- Migration guides

### scripts/
Utility scripts for maintenance:
- Database management
- Model debugging
- User administration

### tests/
Test suites for validation:
- Model accuracy tests
- Integration tests
- Sample processing

## Path Configuration

The project uses relative paths configured as:
```python
UPLOAD_FOLDER = 'data/uploads'
PIPELINES_FRAMES_DIR = 'data/pipelines_frames'
PIPELINES_CROPS_DIR = 'data/pipelines_crops'
model_paths = ['models/best_model_512.pth', 'models/best_model.pth']
```

All file references are relative to the Required/ directory.

## Getting Started

### 1. Setup Environment
```bash
pip install -r deploy/identix-deploy/requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings:
# - MONGODB_URI
# - SECRET_KEY
# - FLASK_ENV
```

### 3. Run Main App
```bash
# Facial landmark detection (recommended)
python landmark_app.py

# Or video pipeline
python app.py
```

### 4. Run via Docker (Deployment)
See deploy/identix-deploy/ for Render/Docker configuration.

## Features Implemented

### Facial Landmark Detection
- ✅ BiSeNet-based 11-class landmark segmentation
- ✅ Multi-face detection with bounding boxes
- ✅ Real-time webcam processing
- ✅ Video frame analysis

### Deepfake Detection
- ✅ Temporal consistency analysis
- ✅ Boundary artifact detection
- ✅ Blink pattern analysis
- ✅ Landmark stability metrics
- ✅ Confidence scoring (0-100%)

### Video Segmentation
- ✅ Frame extraction from videos
- ✅ Parallel face detection
- ✅ Batch segmentation
- ✅ Results visualization

### User Management
- ✅ Authentication & registration
- ✅ Upload history tracking
- ✅ MongoDB integration
- ✅ Session management

## Dependencies

Main packages (see deploy/identix-deploy/requirements.txt):
- Flask 2.3.3
- PyTorch 2.5.1
- TorchVision 0.20.1
- OpenCV 4.8.1
- PyMongo 4.6.0
- TensorFlow/Keras (for alternative models)

## Deployment

The deploy/ folder contains the GitHub/HuggingFace-ready package:
- Production-optimized code
- Docker configuration
- Environment setup
- All dependencies listed

See deploy/identix-deploy/DEPLOYMENT.md for full deployment instructions.

## Important Notes

1. **Models** - All .pth and .keras files are pre-trained and ready to use.
2. **.env** - Never commit your .env file. Use .env.example as template.
3. **Data** - The data/ directory is auto-created at runtime. Don't commit uploads.
4. **Development** - Use landmark_app.py for active development. Best detection quality.
5. **Production** - Use deploy/identix-deploy/ for deployment to cloud platforms.

## Documentation

For detailed information:
- **Getting Started**: See docs/QUICK_REFERENCE.md
- **API Docs**: See docs/DEEPFAKE_README.md & docs/LANDMARK_README.md
- **Deployment**: See deploy/identix-deploy/DEPLOYMENT.md
- **Architecture**: See docs/PROJECT_DOCUMENTATION.md

## Troubleshooting

**Model not loading?** Check that models/ directory contains the .pth/.keras files.
**MongoDB error?** Set MONGODB_URI in .env or leave empty to run without DB.
**Port already in use?** Change port in code or stop other processes on port 5000.
**Import errors?** Install requirements: `pip install -r deploy/identix-deploy/requirements.txt`

---

**Last Updated**: January 2026
**Status**: Production Ready
