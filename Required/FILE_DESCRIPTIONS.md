# 📁 FILE DESCRIPTIONS & ORGANIZATION GUIDE

**Project**: IDENTIX - AI-Powered Facial Landmark & Deepfake Detection System
**Last Updated**: January 31, 2026
**Purpose**: Complete reference guide explaining every file in the Required/ folder

---

## 🗂️ FOLDER STRUCTURE OVERVIEW

```
Required/
├── CORE APPLICATION FILES (Python)
├── CONFIGURATION FILES
├── MODELS & WEIGHTS
├── DATA DIRECTORIES
├── TEMPLATES (HTML/UI)
├── STATIC ASSETS (CSS, JS)
├── DOCUMENTATION
├── SCRIPTS (Utilities)
├── TESTS
└── DEPLOYMENT PACKAGE
```

---

## 📄 CORE APPLICATION FILES (Python)

### 1. **landmark_app.py** ⭐ PRIMARY APPLICATION
- **Category**: Main Flask Application
- **Purpose**: Main facial landmark detection application with web interface
- **Size**: ~1,940 lines
- **Key Components**:
  - Flask server setup and route handlers
  - BiSeNet model architecture (ResNet50 backbone)
  - Face detection and landmark segmentation
  - Deepfake detection integration
  - Video analysis pipeline
  - WebSocket/real-time features
- **Dependencies**: torch, torchvision, flask, opencv-cv2, mediapipe, numpy, PIL
- **Key Classes**:
  - `BiSeNet` - Segmentation model with context and spatial paths
  - `ContextPath` - ResNet-based context extraction
  - `SpatialPath` - Lightweight spatial feature extraction
  - `FeatureFusionModule` - Attention-based feature fusion
  - `ARM` (Attention Refinement Module)
- **Key Routes**:
  - `/` - Landing page
  - `/api/predict_image` - Image landmark detection
  - `/api/predict_video` - Video landmark detection
  - `/detect_deepfake` - Deepfake analysis
  - `/api/history` - User history
- **Status**: ✅ ACTIVE & VERIFIED (Jan 31, 2026)
- **Version**: 1.0.0
- **Run Command**: `python landmark_app.py`
- **Output**: Web interface on http://localhost:5000

---

### 2. **app.py** 🎬 VIDEO SEGMENTATION PIPELINE
- **Category**: Flask Application (Secondary)
- **Purpose**: Video segmentation pipeline orchestrator
- **Size**: ~195 lines
- **Key Components**:
  - Flask video upload handler
  - Pipeline orchestration
  - Background job management
  - Status tracking JSON
  - Inference runner
  - File serving
- **Dependencies**: flask, pipeline_runner, werkzeug, pathlib, json, threading
- **Key Routes**:
  - `/upload` - Video upload endpoint
  - `/results/<filename>` - Results retrieval
  - `/status/<filename>` - Pipeline status
  - `/inference/<filename>` - Run inference
  - `/cleanup` - Clean uploaded files
- **Key Functions**:
  - `run_pipeline_background()` - Async pipeline execution
  - `serve_file()` - Safely serve files
- **Configuration Constants**:
  - `UPLOAD_FOLDER = 'data/uploads'`
  - `PIPELINES_FRAMES_DIR = 'data/pipelines_frames'`
  - `PIPELINES_CROPS_DIR = 'data/pipelines_crops'`
  - `STATUS_FILE = 'data/pipeline_status.json'`
- **Status**: ✅ ACTIVE (Updated Jan 2026)
- **Run Command**: `python app.py` (on different port than landmark_app)
- **Note**: Runs on port 5001 or different port to avoid conflicts

---

### 3. **deepfake_detector.py** 🔍 DEEPFAKE ANALYSIS ENGINE
- **Category**: Analysis Module
- **Purpose**: Multi-method deepfake detection and analysis
- **Size**: ~328 lines
- **Key Components**:
  - Temporal consistency analyzer
  - Boundary artifact detector
  - Blink pattern analyzer
  - Landmark stability calculator
  - Confidence scoring
- **Dependencies**: numpy, opencv-cv2, scipy, torch, collections, matplotlib
- **Key Class**: `DeepfakeDetector`
- **Key Methods**:
  - `calculate_landmark_distances()` - Geometric face metrics
  - `analyze_temporal_consistency()` - Frame-to-frame movement
  - `detect_boundary_artifacts()` - Gradient-based artifact detection
  - `analyze_blink_pattern()` - Eye blink authenticity
  - `calculate_landmark_stability()` - Jitter measurement
  - `analyze_video()` - Complete video analysis
- **Output**: JSON report with verdict (Authentic/Suspicious/Deepfake)
- **Confidence Scoring**: 0-100% (higher = more likely authentic)
- **Status**: ✅ INTEGRATED & WORKING
- **Used By**: landmark_app.py route `/detect_deepfake`
- **Accuracy**: ~85% on test deepfakes (reference baseline)

---

### 4. **pipeline_runner.py** 🎞️ VIDEO PROCESSING ORCHESTRATOR
- **Category**: Pipeline Module
- **Purpose**: Orchestrate video processing: frame extraction → landmark detection → segmentation
- **Size**: ~264 lines
- **Key Components**:
  - Video frame extraction
  - Multi-frame landmark detection (MediaPipe)
  - Face region cropping
  - Batch inference runner
  - Model candidate selection
- **Dependencies**: cv2, mediapipe, tensorflow, json, tqdm, os, pathlib
- **Key Functions**:
  - `extract_frames()` - Extract frames at specified FPS
  - `detect_landmarks_for_frames()` - MediaPipe 468-landmark extraction
  - `pick_best_face_region_and_save()` - Crop optimization
  - `find_model_candidate()` - Smart model file selection
  - `run_inference_on_crops()` - Batch inference execution
  - `run_pipeline()` - Main orchestration function
- **Supported Models**:
  - UNet: `unet_model.keras`, `unet_smoke.keras`
  - DeepLab: `deeplab_model.keras`, `deeplab_model_stage1.keras`
  - ViT: `vit_model.keras`, `vit_full.keras`, `vit_smoke.keras`
- **Status**: ✅ ACTIVE (Path-updated Jan 2026)
- **Used By**: app.py for video pipeline

---

### 5. **mongodb_utils.py** 💾 DATABASE CONNECTION MANAGER
- **Category**: Database Module
- **Purpose**: MongoDB connection management and utilities
- **Size**: ~104 lines
- **Key Components**:
  - Singleton MongoClient pattern
  - Connection pooling and validation
  - Error handling for database operations
  - Client initialization
- **Dependencies**: pymongo, python-dotenv
- **Key Functions**:
  - `get_client()` - Get or create MongoClient singleton
  - `get_db()` - Get facial_landmarks_db database reference
  - `validate_connection()` - Test connection with 5s timeout
- **Configuration**:
  - Database: `facial_landmarks_db`
  - Collections: `auths` (users), `uploads` (history)
  - Timeout: 5 seconds (serverSelectionTimeoutMS)
  - Environment Variable: `MONGODB_URI`
- **Error Handling**:
  - ConfigurationError
  - OperationFailure
  - ServerSelectionTimeoutError
- **Status**: ✅ OPTIONAL (Can run without DB)
- **Note**: Currently running with MONGODB_URI="" (disabled)

---

### 6. **model.py** 🧠 SEGMENTATION MODEL ARCHITECTURES
- **Category**: Model Definition Module
- **Purpose**: BiSeNet and supporting architecture definitions
- **Size**: ~[varies] lines
- **Key Components**:
  - BiSeNet class definition
  - ContextPath with ResNet backbone
  - SpatialPath architecture
  - FeatureFusionModule
  - Auxiliary heads for training
- **Key Classes**:
  - `BiSeNet` - Main segmentation model
  - `ARM` (Attention Refinement Module)
  - `FFM` (Feature Fusion Module)
- **Output**: 11-class facial landmark segmentation
- **Status**: ✅ REFERENCE (Model inline in landmark_app.py)
- **Note**: landmark_app.py includes inline BiSeNet; this file is for reference

---

### 7. **resnet.py** 🏗️ RESNET ARCHITECTURE COMPONENTS
- **Category**: Model Architecture Module
- **Purpose**: ResNet backbone definitions (ResNet18, ResNet50)
- **Size**: ~[varies] lines
- **Key Components**:
  - ResNet50 backbone (primary)
  - ResNet18 backbone (lightweight)
  - Residual blocks
  - Bottleneck layers
  - Feature extraction stages
- **Used By**: BiSeNet ContextPath in landmark_app.py
- **Status**: ✅ REFERENCE (Implemented in torch.models)
- **Note**: Primarily used via torchvision; included for custom modifications

---

### 8. **mediapipe_landmark_detector.py** 📍 MEDIAPIPE INTEGRATION (STUB)
- **Category**: Optional Module
- **Purpose**: MediaPipe Face Mesh integration for 468-landmark detection
- **Size**: ~[varies] lines
- **Status**: ⚠️ STUB/OPTIONAL
- **Note**: Currently disabled in primary workflow; used in pipeline_runner.py
- **Key Features**:
  - 468 3D face landmarks
  - Face tracking across frames
  - Confidence scoring per landmark
- **Used By**: pipeline_runner.py for multi-frame processing

---

## ⚙️ CONFIGURATION & ENVIRONMENT FILES

### 9. **.env** 🔐 ENVIRONMENT VARIABLES (SECRET)
- **Purpose**: Store sensitive configuration
- **Status**: ⚠️ DO NOT COMMIT TO GIT
- **Key Variables**:
  ```
  MONGODB_URI="mongodb+srv://..." (optional)
  SECRET_KEY="[generated secret]"
  FLASK_ENV="development" or "production"
  PORT="5000"
  DEBUG="True" or "False"
  ```
- **How to Create**:
  1. Copy from .env.example
  2. Generate SECRET_KEY: `python -c "import secrets; print(secrets.token_hex(32))"`
  3. Set MONGODB_URI if using database
  4. Save and add to .gitignore

---

## 📦 MODELS & WEIGHTS

### **models/** folder - Pre-trained Model Weights

#### **Best Models (Recommended)**
| File | Model | Purpose | Size | Status |
|------|-------|---------|------|--------|
| `best_model.pth` | BiSeNet | Primary landmark detector | ~25MB | ✅ Active |
| `best_model_512.pth` | BiSeNet | High-res fine-tuned (512×512) | ~25MB | ✅ Active |

#### **Alternative Models**
| File | Model | Purpose | Status |
|------|-------|---------|--------|
| `79999_iter.pth` | BiSeNet | Checkpoint variant | ✅ Available |
| `unet_model.keras` | UNet | Segmentation alternative | ✅ Available |
| `unet_smoke.keras` | UNet | Lightweight UNet | ✅ Available |
| `deeplab_model.keras` | DeepLab v3+ | DeepLab alternative | ✅ Available |
| `deeplab_model_stage1.keras` | DeepLab | Stage-1 checkpoint | ✅ Available |
| `vit_model.keras` | Vision Transformer | ViT alternative | ✅ Available |
| `vit_full.keras` | Vision Transformer | Full ViT model | ✅ Available |
| `vit_smoke.keras` | Vision Transformer | Lightweight ViT | ✅ Available |

**Loading Models**:
```python
# PyTorch (BiSeNet)
model = torch.load('models/best_model.pth', weights_only=True)

# TensorFlow (Alternatives)
model = tf.keras.models.load_model('models/unet_model.keras')
```

---

## 📂 DATA DIRECTORIES

### **data/** folder - Runtime Data (Auto-created)

| Directory | Purpose | Auto-Created |
|-----------|---------|--------------|
| `data/uploads/` | User uploaded images/videos | ✅ Yes |
| `data/pipelines_frames/` | Extracted video frames | ✅ Yes |
| `data/pipelines_crops/` | Cropped face regions | ✅ Yes |

**Notes**:
- These folders are auto-created by the app
- Safe to delete; app recreates them
- .gitignore should include data/ to prevent pushing user uploads

---

## 🎨 TEMPLATES (HTML/UI)

### **templates/** folder - Flask Template Files

#### **Main Application Templates**

| File | Purpose | Route | Status |
|------|---------|-------|--------|
| `index.html` | Landing page | `/` | ✅ Active |
| `landmark_index.html` | Landmark detection UI | `/facial-landmarks` | ✅ Active |
| `deepfake_detection.html` | Deepfake analyzer UI | `/deepfake` | ✅ Active |

#### **Analysis Templates**

| File | Purpose | Route | Status |
|------|---------|-------|--------|
| `image_analysis.html` | Image segmentation UI | `/image-analysis` | ✅ Active |
| `video_analysis.html` | Video segmentation UI | `/video-analysis` | ✅ Active |
| `results.html` | Results display | `/results` | ✅ Active |

#### **User Management Templates**

| File | Purpose | Route | Status |
|------|---------|-------|--------|
| `profile.html` | User profile | `/profile` | ✅ Active |
| `settings.html` | User settings | `/settings` | ✅ Active |

#### **Legal & Support Templates**

| File | Purpose | Status |
|------|---------|--------|
| `about.html` | About page | ✅ Active |
| `contact.html` | Contact form | ✅ Active |
| `privacy-policy.html` | Privacy policy | ✅ Active |
| `terms-of-service.html` | Terms & conditions | ✅ Active |
| `gdpr.html` | GDPR compliance | ✅ Active |
| `cookie-policy.html` | Cookie policy | ✅ Active |
| `blog.html` | Blog/documentation | ✅ Active |
| `careers.html` | Careers page | ✅ Active |

---

## 🎨 STATIC ASSETS

### **static/** folder - CSS, JavaScript, Images

```
static/
├── css/
│   ├── style.css              # Main stylesheet
│   ├── responsive.css         # Mobile responsive styles
│   └── animations.css         # CSS animations
│
├── js/
│   ├── deepfake_frontend.js  # Deepfake UI interactions
│   ├── landmark.js            # Landmark UI interactions
│   ├── video_handler.js       # Video upload handling
│   └── api_client.js          # API communication
│
└── images/
    ├── logo.png               # Identix logo
    ├── icons/                 # UI icons
    └── examples/              # Example images
```

---

## 📚 DOCUMENTATION

### **docs/** folder - Comprehensive Documentation (40+ files)

**Quick Reference Documents**:
- `QUICK_REFERENCE.md` - Fast start guide
- `IDENTIX_GUIDE.md` - Platform overview
- `PROJECT_DOCUMENTATION.md` - Technical details

**Feature Documentation**:
- `LANDMARK_README.md` - Facial landmark detection guide
- `DEEPFAKE_README.md` - Deepfake detection guide
- `MEDIAPIPE_UPGRADE_GUIDE.md` - MediaPipe integration

**Technical Documentation**:
- `COMPREHENSIVE_ANALYSIS.md` - Deep technical analysis
- `EDGE_REFINEMENT_TECHNICAL_NOTES.md` - Refinement techniques
- `VIDEO_SEGMENTATION_FIX.md` - Video pipeline details

**Change & Improvement Logs**:
- `CHANGE_LOG.md` - Chronological change history
- `BEFORE_AFTER_COMPARISON.md` - Improvements summary
- `IMPROVEMENTS_SUMMARY.md` - Feature improvements
- `FIX_EXPLANATION.md` - Bug fixes explained

**UI & Deployment**:
- `UI_IMPROVEMENTS.md` - UI enhancements
- `UI_RESTRUCTURE_SUMMARY.md` - UI organization
- `VISUAL_GUIDE.md` - Visual documentation

**Root Level Documentation**:
- `README_PROJECT_STRUCTURE.md` - File structure (in Required/)
- `DEPLOYMENT.md` (in deploy/) - Deployment instructions

---

## 🛠️ SCRIPTS

### **scripts/** folder - Utility Scripts

| Script | Purpose | When to Run |
|--------|---------|------------|
| `setup_mongodb.py` | Initialize MongoDB connection | First time setup |
| `clear_users.py` | Delete all users from database | Database cleanup |
| `list_users.py` | List all users in database | Check database |
| `debug_prediction.py` | Debug model predictions | Troubleshooting |

**Run Examples**:
```bash
python scripts/setup_mongodb.py
python scripts/list_users.py
python scripts/clear_users.py
python scripts/debug_prediction.py
```

---

## ✅ TESTS

### **tests/** folder - Test Scripts

| Test | Purpose | Coverage |
|------|---------|----------|
| `test_landmark_simple.py` | Basic landmark detection | Single image |
| `test_mediapipe_accuracy.py` | MediaPipe accuracy validation | 468-landmark test |
| `test_all_combinations.py` | Comprehensive model testing | All model variants |
| `test_sample2.py` | Video processing test | Video pipeline |

**Run Examples**:
```bash
python tests/test_landmark_simple.py
python tests/test_all_combinations.py
```

---

## 🚀 DEPLOYMENT

### **deploy/identix-deploy/** folder - Production Package

**Purpose**: Complete GitHub/Render deployment package

**Contents**:
- `app.py` - Production-optimized Flask app
- `deepfake_detector.py` - Deepfake module (copy)
- `mongodb_utils.py` - Database utilities (copy)
- `best_model.pth` - Deployed model weight
- `requirements.txt` - Python dependencies
- `render.yaml` - Render platform configuration
- `Procfile` - Heroku/Render process definition
- `templates/` - All HTML templates (copy)
- `static/` - Static assets (copy)
- `DEPLOYMENT.md` - Deployment instructions
- `CODE_REVIEW.md` - Code audit report

**How to Use**:
1. Push to GitHub: `git push`
2. Deploy to Render: `git push heroku main`
3. Or use deploy script: `bash deploy.sh`

---

## 📊 FILE STATISTICS

| Category | Count | Status |
|----------|-------|--------|
| Python Files | 8 | ✅ Active |
| HTML Templates | 16 | ✅ Active |
| Model Weights | 10 | ✅ Available |
| Documentation Files | 40+ | ✅ Complete |
| Test Scripts | 5 | ✅ Ready |
| Utility Scripts | 4 | ✅ Available |
| **TOTAL** | **250+** | **✅ Organized** |

---

## 🎯 QUICK REFERENCE

**To RUN the application**:
```bash
cd d:\link2\Capstone 4-1\Code_try_1\Required
python landmark_app.py
# Open http://localhost:5000
```

**To TEST detection**:
```bash
cd Required/tests
python test_landmark_simple.py
```

**To DEBUG**:
```bash
python scripts/debug_prediction.py
```

**To DEPLOY**:
```bash
cd Required/deploy/identix-deploy/
# Follow DEPLOYMENT.md instructions
```

---

## 📝 SUMMARY

This document provides a **complete inventory** of all files in the Required/ folder with:
- ✅ Clear descriptions of what each file does
- ✅ Dependencies and requirements
- ✅ Key components and functions
- ✅ When and how to use each file
- ✅ Current status (Active/Archive/Optional)
- ✅ Integration points with other files

**Use this document as a reference whenever you need to understand any file in the project.**

---

**Last Updated**: January 31, 2026
**Version**: 1.0.0
**Maintainer**: IDENTIX Team

