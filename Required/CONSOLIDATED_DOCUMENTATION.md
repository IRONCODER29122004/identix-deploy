# 📚 IDENTIX CONSOLIDATED DOCUMENTATION

**Merged from multiple documentation files**
**Last Updated**: January 31, 2026
**Version**: 1.0.0 Consolidated

---

## TABLE OF CONTENTS

1. [Quick Start Guide](#quick-start-guide)
2. [Project Overview](#project-overview)
3. [What is IDENTIX?](#what-is-identix)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Running the Application](#running-the-application)
7. [Facial Landmark Detection Guide](#facial-landmark-detection-guide)
8. [Deepfake Detection System](#deepfake-detection-system)
9. [Video Segmentation Pipeline](#video-segmentation-pipeline)
10. [Technical Architecture](#technical-architecture)
11. [Deployment Guide](#deployment-guide)
12. [Troubleshooting](#troubleshooting)

---

## QUICK START GUIDE

### 30-Second Startup

```bash
# Navigate to project
cd d:\link2\Capstone 4-1\Code_try_1\Required

# Run the main application
python landmark_app.py

# Open browser
# http://localhost:5000
```

### What to expect:
- App starts in ~30 seconds
- BiSeNet model loads on first run (shows progress)
- Server listens on http://localhost:5000
- Upload an image or video to test

### First Test:
1. Open http://localhost:5000
2. Click "🎯 Facial Landmarks" tab
3. Upload a face image
4. See landmarks detected in 11 classes

---

## PROJECT OVERVIEW

### What is IDENTIX?

IDENTIX is a comprehensive **AI-powered facial analysis platform** with three integrated applications:

#### 1. **Facial Landmark Detection** (Primary) ⭐
- Detect and segment 11 facial landmark classes
- BiSeNet deep learning model (98% accuracy)
- Supports images and videos
- Real-time processing
- Multi-face detection

**Classes Detected:**
```
0: Background    1: Skin          2: Left Eyebrow
3: Right Eyebrow 4: Left Eye      5: Right Eye
6: Nose          7: Upper Lip     8: Inner Mouth
9: Lower Lip     10: Hair
```

#### 2. **Deepfake Detection** (Advanced) 🔍
- Analyze videos for manipulation signs
- 4 detection methods combined:
  - Temporal consistency analysis
  - Boundary artifact detection
  - Blink pattern analysis
  - Landmark stability metrics
- Confidence scoring (0-100%)
- Verdict: Authentic / Suspicious / Deepfake

#### 3. **Video Segmentation Pipeline** (Optional) 🎬
- Extract frames from videos
- Detect faces across frames
- Apply segmentation to all faces
- Generate comparison videos
- Batch processing support

### Key Statistics

| Metric | Value |
|--------|-------|
| **Model Accuracy** | 98% on test set |
| **Processing Speed** | <100ms per image |
| **Supported Formats** | JPG, PNG, MP4, AVI, MOV |
| **Max Upload Size** | 500MB |
| **Classes Detected** | 11 facial landmark classes |
| **Video FPS** | Configurable (1-30) |

---

## PROJECT STRUCTURE

### Complete Directory Layout

```
Required/ (MAIN PROJECT - USE THIS!)
│
├── 📄 CORE APPLICATION FILES
│   ├── landmark_app.py          ⭐ Main facial landmark detection app
│   ├── app.py                   # Video segmentation pipeline
│   ├── deepfake_detector.py     # Deepfake analysis module
│   ├── pipeline_runner.py       # Video processing orchestrator
│   ├── mongodb_utils.py         # Database connection manager
│   ├── model.py                 # Model architecture definitions
│   ├── resnet.py                # ResNet backbone components
│   └── mediapipe_landmark_detector.py # MediaPipe integration
│
├── 📦 MODELS & WEIGHTS
│   └── models/
│       ├── best_model.pth       ⭐ Primary BiSeNet model
│       ├── best_model_512.pth   # High-res fine-tuned version
│       ├── unet_model.keras     # UNet alternative
│       ├── deeplab_model.keras  # DeepLab alternative
│       ├── vit_model.keras      # Vision Transformer
│       └── [6 more model variants]
│
├── 💾 DATA DIRECTORIES (auto-created)
│   └── data/
│       ├── uploads/             # User uploaded files
│       ├── pipelines_frames/    # Extracted video frames
│       └── pipelines_crops/     # Cropped face regions
│
├── 🎨 USER INTERFACE
│   ├── templates/               # HTML templates
│   │   ├── index.html          # Landing page
│   │   ├── landmark_index.html  # Landmark UI
│   │   ├── deepfake_detection.html # Deepfake UI
│   │   ├── image_analysis.html  # Image analysis UI
│   │   ├── video_analysis.html  # Video analysis UI
│   │   ├── profile.html         # User profile
│   │   ├── results.html         # Results display
│   │   └── [8 more templates]
│   │
│   └── static/                  # CSS, JS, Images
│       ├── css/                 # Stylesheets
│       ├── js/                  # JavaScript files
│       │   └── deepfake_frontend.js
│       └── images/              # UI images
│
├── 📚 DOCUMENTATION
│   ├── docs/                    # Complete technical docs (40+ files)
│   │   ├── QUICK_REFERENCE.md   # Fast start
│   │   ├── PROJECT_DOCUMENTATION.md # Technical details
│   │   ├── DEEPFAKE_README.md   # Deepfake guide
│   │   ├── LANDMARK_README.md   # Landmark guide
│   │   ├── IDENTIX_GUIDE.md     # Platform overview
│   │   └── [35+ more docs]
│   │
│   ├── FILE_DESCRIPTIONS.md     # File inventory
│   ├── RULES_PAGE.md            # Project guidelines
│   └── README_PROJECT_STRUCTURE.md
│
├── 🛠️ UTILITY SCRIPTS
│   └── scripts/
│       ├── setup_mongodb.py     # Initialize MongoDB
│       ├── clear_users.py       # Delete users
│       ├── list_users.py        # List users
│       └── debug_prediction.py  # Debug predictions
│
├── ✅ TEST SCRIPTS
│   └── tests/
│       ├── test_landmark_simple.py
│       ├── test_mediapipe_accuracy.py
│       ├── test_all_combinations.py
│       └── test_sample2.py
│
├── 🚀 DEPLOYMENT PACKAGE
│   └── deploy/identix-deploy/
│       ├── app.py               # Optimized for deployment
│       ├── requirements.txt     # Dependencies
│       ├── render.yaml          # Render config
│       ├── Procfile            # Process config
│       ├── best_model.pth       # Deployed model
│       ├── templates/           # Deployment templates
│       ├── DEPLOYMENT.md        # Deployment instructions
│       └── CODE_REVIEW.md       # Code audit
│
├── ⚙️ CONFIGURATION
│   └── .env                     # Environment variables (SECRET)
│
└── 📋 THIS DOCUMENTATION
    ├── CONSOLIDATED_DOCUMENTATION.md (YOU ARE HERE)
    ├── FILE_DESCRIPTIONS.md
    └── RULES_PAGE.md
```

### Key Subdirectories

#### **models/** - Pre-trained Weights
- **Primary**: `best_model.pth` (BiSeNet 256×256)
- **Fine-tuned**: `best_model_512.pth` (BiSeNet 512×512)
- **Alternatives**: UNet, DeepLab, Vision Transformer variants
- All models ready to use; no training required

#### **data/** - Runtime Data (Auto-created)
- `uploads/` - Where user files are stored temporarily
- `pipelines_frames/` - Video frame extraction cache
- `pipelines_crops/` - Cropped face regions cache
- Safe to delete; app recreates automatically

#### **templates/** - HTML UI
- **Main**: index.html (landing), landmark_index.html (landmark app)
- **Analysis**: image_analysis.html, video_analysis.html, deepfake_detection.html
- **User**: profile.html, settings.html
- **Legal**: privacy-policy.html, terms-of-service.html, gdpr.html, etc.

#### **docs/** - Complete Documentation
- 40+ markdown files covering all aspects
- From quick start to detailed technical guides
- Architecture, API, deployment, troubleshooting

---

## INSTALLATION & SETUP

### Prerequisites

✅ **Required:**
- Python 3.8+ (Tested on 3.11.9)
- 4GB RAM minimum
- 2GB disk space (models + dependencies)
- Windows / macOS / Linux

✅ **Optional:**
- CUDA-capable GPU (for faster inference)
- MongoDB Atlas account (for user history)

### Step 1: Install Python Dependencies

```bash
# Navigate to project
cd d:\link2\Capstone 4-1\Code_try_1\Required

# Install all dependencies
pip install -r deploy/identix-deploy/requirements.txt
```

**OR manually install**:
```bash
pip install flask==2.3.3
pip install torch==2.5.1 torchvision==0.20.1
pip install opencv-python==4.8.1
pip install pillow numpy scipy
pip install pymongo==4.6.0
pip install python-dotenv
pip install mediapipe
pip install tensorflow keras
```

### Step 2: Configure Environment

```bash
# Create .env file (if not exists)
cd Required
cp .env.example .env

# Edit .env file with your settings
# Open .env in text editor and set:
```

**.env file template:**
```bash
# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your-secret-key-here-use-this-command-to-generate: python -c "import secrets; print(secrets.token_hex(32))"
PORT=5000
DEBUG=True

# Database Configuration (OPTIONAL)
MONGODB_URI=
# Leave empty to run without MongoDB
# Or set: MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true

# Model Configuration
MODEL_PATH=models/best_model.pth
DEVICE=cpu
# Set to 'cuda' if you have NVIDIA GPU

# Application Configuration
DEMO_MODE=true
LAZY_LOAD=true
MAX_CONTENT_LENGTH=524288000  # 500MB max upload
```

### Step 3: Verify Installation

Test that everything is installed correctly:

```bash
# Test Python version
python --version
# Should show: Python 3.x.x

# Test imports
python -c "import torch; print('✅ PyTorch OK'); import cv2; print('✅ OpenCV OK'); import flask; print('✅ Flask OK')"

# Should show all three ✅ marks
```

---

## RUNNING THE APPLICATION

### Option 1: Run Main Application (Recommended)

```bash
cd d:\link2\Capstone 4-1\Code_try_1\Required
python landmark_app.py
```

**Output:**
```
 * Running on http://127.0.0.1:5000
 * WARNING in app.run_app: This is a development server. Do not use it in production directly.
 * Model loaded successfully: BiSeNet
 * Press CTRL+C to quit
```

**Access the app**: Open your browser and go to **http://localhost:5000**

### Option 2: Run Video Segmentation Pipeline

```bash
cd Required
python app.py
```

Runs on a different port (5001 or configured port)

### Option 3: Run with Custom Settings

```bash
# Set environment variable for custom port
set FLASK_ENV=production
set PORT=8080
python landmark_app.py

# Or on Linux/Mac:
export FLASK_ENV=production
export PORT=8080
python landmark_app.py
```

### Option 4: Run with MongoDB

Setup MongoDB first:
```bash
# Interactive setup
python scripts/setup_mongodb.py

# Then run app with DB enabled
python landmark_app.py
```

### Verification

After starting the app, verify it's working:

```bash
# In another terminal/PowerShell window:
curl http://localhost:5000/health
# Should return: {"status": "healthy", "model_loaded": true}
```

---

## FACIAL LANDMARK DETECTION GUIDE

### What It Does

Detects and segments 11 facial landmark classes from images and videos using deep learning.

### Features

✅ **Image Support**
- Drag-and-drop interface
- Support: JPG, PNG, BMP, GIF
- Auto-detects face
- Multi-face detection
- Instant visualization

✅ **Video Support**
- MP4, AVI, MOV formats
- Frame-by-frame analysis
- Best-frame extraction
- Temporal consistency
- Progress tracking

✅ **Visualization**
- Original image overlay
- Segmentation mask
- Colored class regions
- Confidence heatmap
- Comparison view

### How to Use

#### Via Web Interface

1. **Open the app**: http://localhost:5000
2. **Click "🎯 Facial Landmarks" tab**
3. **Choose image or video**:
   - Click upload area or drag-and-drop file
   - Wait for processing (30-500ms per image)
4. **View results**:
   - Original image with landmarks
   - Segmentation mask (11 colors)
   - Statistics (pixel counts per class)
   - Download option

#### Via Python API

```python
import requests
from PIL import Image

# Prepare image
with open('face.jpg', 'rb') as f:
    files = {'image': f}
    
# Send to server
response = requests.post(
    'http://localhost:5000/api/predict_image',
    files=files
)

# Get results
result = response.json()
print(f"Detected {result['num_faces']} face(s)")
print(f"Landmarks: {result['landmark_stats']}")
```

### Processing Pipeline

```
Input Image/Video
    ↓
[Face Detection] (Haar Cascade)
    ↓
[Multiple Candidates Evaluated]
    ↓
[Best Face Selected]
    ↓
[Cropped & Resized to 512×512]
    ↓
[BiSeNet Model Inference]
    ↓
[11-Class Segmentation Mask]
    ↓
[Edge Refinement (Optional)]
    ↓
[Colored Visualization]
    ↓
[Results + Statistics]
    ↓
Output: PNG image + JSON data
```

### Classes Detected (11 Total)

| ID | Class | Color | Purpose |
|----|-------|-------|---------|
| 0 | Background | Gray | Non-face area |
| 1 | Skin | Peach | Face skin region |
| 2 | Left Eyebrow | Brown | Left eyebrow |
| 3 | Right Eyebrow | Brown | Right eyebrow |
| 4 | Left Eye | Blue | Left eye region |
| 5 | Right Eye | Blue | Right eye region |
| 6 | Nose | Green | Nose area |
| 7 | Upper Lip | Pink | Upper lip |
| 8 | Inner Mouth | Red | Mouth interior |
| 9 | Lower Lip | Pink | Lower lip |
| 10 | Hair | Black | Hair region |

### Model Architecture

**BiSeNet** (Bilateral Segmentation Network):
- **Context Path**: ResNet50 backbone for receptive field
- **Spatial Path**: Lightweight parallel path for details
- **Fusion Module**: Attention-based feature combination
- **Output**: 11-class segmentation at original resolution

### Performance

- **Accuracy**: 98% on test set
- **Speed**: <100ms per image (CPU)
- **GPU Speed**: <20ms per image (NVIDIA GPU)
- **Memory**: 2GB RAM minimum
- **Model Size**: 25MB

---

## DEEPFAKE DETECTION SYSTEM

### What It Does

Analyzes videos to identify signs of manipulation using 4 different detection methods.

### How It Works

The system combines 4 complementary analysis techniques:

#### 1. **Temporal Consistency Analysis** 🎬
- Tracks how facial landmarks move between frames
- Detects unnatural jumps or inconsistent movements
- **Real videos**: Smooth, natural transitions
- **Deepfakes**: Jerky, unrealistic movements

#### 2. **Boundary Artifact Detection** 🔍
- Analyzes edges around facial boundaries
- Identifies suspicious gradients and blending artifacts
- **Real videos**: Clean, natural edges
- **Deepfakes**: Blurry, artifact-filled edges

#### 3. **Blink Pattern Analysis** 👁️
- Tracks eye blinking frequency and regularity
- Checks for natural blink duration and intervals
- **Real videos**: 15-20 blinks per minute, 100-400ms duration
- **Deepfakes**: Missing blinks or abnormal patterns

#### 4. **Landmark Stability** ⚡
- Measures jitter and instability in facial features
- Detects micro-movements and tremors
- **Real videos**: Stable, smooth landmarks
- **Deepfakes**: Visible jitter due to frame-by-frame generation

### Verdict Levels

| Confidence | Verdict | Meaning |
|------------|---------|---------|
| 70-100% | ✅ LIKELY AUTHENTIC | Genuine video with confidence |
| 50-70% | ❓ SUSPICIOUS | Manual review recommended |
| 0-50% | ⚠️ LIKELY DEEPFAKE | Probable manipulation detected |

### How to Use

#### Via Web Interface

1. Open **http://localhost:5000**
2. Click **"🔍 Deepfake Detection"** tab
3. Upload a video file (MP4, AVI, MOV)
4. Set number of frames to analyze (default: 100)
5. Click **"Analyze Video"**
6. Wait for analysis (1-3 minutes)
7. View detailed report with metrics

#### Via Python API

```python
import requests

url = 'http://localhost:5000/detect_deepfake'
files = {'video': open('suspect_video.mp4', 'rb')}
data = {'max_frames': 100}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Verdict: {result['report']['verdict']}")
print(f"Confidence: {result['report']['confidence']}%")
print(f"Temporal Score: {result['metrics']['temporal']}")
print(f"Boundary Score: {result['metrics']['boundary']}")
print(f"Blink Score: {result['metrics']['blink']}")
print(f"Stability Score: {result['metrics']['stability']}")
```

### Detection Metrics Explained

#### Temporal Consistency (Landmark Movement)
- **Score 80-100%**: Smooth, natural movements between frames
- **Score 50-80%**: Some irregular transitions, possibly fast motion
- **Score 0-50%**: Many unnatural jumps, likely deepfake

#### Boundary Artifacts (Edge Quality)
- **Score 80-100%**: Clean face boundaries, natural transitions
- **Score 50-80%**: Some suspicious edges, slight blending issues
- **Score 0-50%**: Significant artifacts, poor blending

#### Blink Patterns (Eye Authenticity)
- **Score 80-100%**: Natural blink rate and rhythm (15-20/min)
- **Score 50-80%**: Slightly irregular blinking, fewer blinks
- **Score 0-50%**: Abnormal or missing blinks

#### Landmark Stability (Jitter)
- **Score 80-100%**: Smooth, stable landmarks with minimal jitter
- **Score 50-80%**: Slight instability, some tremors
- **Score 0-50%**: Significant jitter, visible micro-movements

### Example Report

```json
{
  "verdict": "LIKELY DEEPFAKE",
  "confidence": 32,
  "metrics": {
    "temporal": 45,
    "boundary": 28,
    "blink": 55,
    "stability": 20
  },
  "frame_analysis": {
    "total_frames": 100,
    "valid_faces": 95,
    "average_face_confidence": 0.92
  },
  "recommendation": "Video likely contains deepfake elements. Manual review recommended."
}
```

### Important Notes

- ✅ **Confidence Score**: 0-100% (higher = more authentic)
- ✅ **Processing Time**: ~1 second per frame
- ✅ **Accuracy**: ~85% on known deepfakes (reference)
- ⚠️ **Not Guaranteed**: No system is 100% accurate
- ⚠️ **Manual Review**: Recommended for legal/security purposes

---

## VIDEO SEGMENTATION PIPELINE

### What It Does

Extracts frames from videos, detects facial landmarks per frame, and segments faces across the video.

### Pipeline Steps

```
Input Video (MP4, AVI, MOV)
    ↓
[Step 1] Extract Frames at specified FPS (configurable 1-30)
    ↓
[Step 2] Detect Landmarks per frame (MediaPipe 468-point detection)
    ↓
[Step 3] Face Tracking (IoU-based face matching across frames)
    ↓
[Step 4] Quality Scoring (for each detected face)
    ↓
[Step 5] Best Frame Selection (overall and per-landmark)
    ↓
[Step 6] Segmentation Inference (BiSeNet on selected frames)
    ↓
[Step 7] Generate Results & Visualizations
    ↓
Output: Cropped faces + segmentation masks + statistics
```

### How to Use

#### Via Web Interface

1. Open **http://localhost:5000**
2. Click **"🎬 Video Analysis"** tab
3. Upload a video file
4. Configure settings:
   - FPS: Number of frames per second to extract
   - Max frames: Maximum frames to process
   - Model: Which segmentation model to use
5. Click **"Process Video"**
6. Download results when complete

#### Via Python API

```python
import requests

url = 'http://localhost:5000/upload'
files = {'video': open('video.mp4', 'rb')}
data = {'fps': 5, 'max_frames': 100, 'model_type': 'bisenet'}

response = requests.post(url, files=files, data=data)
status = response.json()

# Poll for results
import time
filename = status['filename']

while True:
    status_response = requests.get(
        f'http://localhost:5000/status/{filename}'
    )
    status = status_response.json()
    
    if status['completed']:
        results = requests.get(
            f'http://localhost:5000/results/{filename}'
        )
        break
    
    print(f"Progress: {status['progress']}%")
    time.sleep(2)
```

### Features

✅ **Frame Extraction**
- Configurable FPS (1-30)
- Maximum frame limit
- Smart sampling

✅ **Face Tracking**
- Multi-face detection
- IoU-based tracking
- Quality scoring

✅ **Best Frame Selection**
- Best overall frame (highest quality)
- Best per-landmark frames
- Temporal smoothing

✅ **Batch Processing**
- Parallel inference available
- Progress tracking
- Partial result saving

### Configuration Options

| Option | Default | Range | Purpose |
|--------|---------|-------|---------|
| FPS | 5 | 1-30 | Frames per second to extract |
| Max Frames | 100 | 1-1000 | Maximum frames to process |
| Model | bisenet | - | bisenet, unet, deeplab, vit |
| Min Confidence | 0.5 | 0-1 | Minimum face detection confidence |
| Batch Size | 32 | 1-128 | GPU batch size |

---

## TECHNICAL ARCHITECTURE

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      WEB INTERFACE                          │
│  (HTML5 + JavaScript + Bootstrap CSS)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   FLASK SERVER                              │
│  (landmark_app.py or app.py)                               │
│  - Route handling                                          │
│  - File management                                         │
│  - Session management                                      │
│  - Database integration (optional)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐   ┌─────▼─────┐   ┌────▼──────┐
    │  FACE   │   │  DEEPFAKE │   │  VIDEO    │
    │ DETECTOR│   │ DETECTOR  │   │ PROCESSOR │
    │(Haar)   │   │(Multi-    │   │(Pipeline) │
    │         │   │ method)   │   │           │
    └────┬────┘   └─────┬─────┘   └────┬──────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │   LANDMARK SEGMENTATION       │
         │   (BiSeNet Model)             │
         │  - 11-class output            │
         │  - 256x256 or 512x512         │
         │  - Edge refinement (optional) │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │      MODEL WEIGHTS (PyTorch)  │
         │  - best_model.pth (PRIMARY)   │
         │  - best_model_512.pth (FINE)  │
         │  - Alternatives (UNet, etc.)  │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │      RESULTS & STORAGE        │
         │  - Segmentation masks         │
         │  - Cropped face regions       │
         │  - Statistics JSON            │
         │  - MongoDB (optional)         │
         └───────────────────────────────┘
```

### Data Flow

#### Image Processing

```
User uploads image
    ↓ (HTTP POST)
Flask receives upload
    ↓
Face detection (Haar Cascade)
    ↓
Multiple candidates evaluated
    ↓
Best candidate selected
    ↓
Crop and resize (512×512)
    ↓
BiSeNet inference
    ↓
11-class segmentation mask
    ↓
Edge refinement (optional)
    ↓
Create visualization
    ↓
Return results (base64 + JSON)
    ↓ (HTTP response)
Display in browser
```

#### Video Processing

```
User uploads video
    ↓
Extract frames (at specified FPS)
    ↓
Detect landmarks per frame (MediaPipe)
    ↓
Track faces across frames
    ↓
Score face quality
    ↓
Select main character
    ↓
Select best frames
    ↓
BiSeNet inference on selected frames
    ↓
Generate visualizations
    ↓
Save results to data/
    ↓
Return summary + links
    ↓
User downloads results
```

### Component Details

#### BiSeNet Model

**Purpose**: 11-class facial landmark segmentation
**Input**: 256×256 or 512×512 RGB image
**Output**: 11-class segmentation mask at input resolution
**Architecture**:
- **Context Path**: ResNet50 + ARM modules
- **Spatial Path**: 3-layer lightweight CNN
- **Fusion Module**: Channel-wise attention + addition

**Key Stats**:
- Parameters: ~2.5M
- File Size: ~25MB (PyTorch)
- Inference Speed: <100ms (CPU)
- Training Data: Custom facial landmark dataset (10,000+ images)

#### Face Detection

**Method**: Haar Cascade Classifiers (OpenCV)
**Advantages**:
- Fast and lightweight
- Multiple cascade detection (face, left ear, right ear)
- Permissive settings for robustness

**Fallback Logic**:
1. Try frontal face detection
2. If fails, try multiple cascade windows
3. If fails, use largest candidate
4. If fails, use full image

#### Landmark Detection

**Method 1**: MediaPipe Face Mesh (468 points)
- Used in pipeline_runner.py
- Real-time face tracking
- Confidence scoring

**Method 2**: BiSeNet (11 classes)
- Primary method
- Semantic segmentation
- Detailed feature boundaries

### Performance Optimization

✅ **Model Loading**:
- Lazy loading on first request
- Singleton pattern (loaded once)
- Weights cached in memory

✅ **Inference Optimization**:
- Batch processing support
- GPU acceleration (if available)
- Image resizing caching

✅ **Database**:
- Lazy loading (optional)
- Connection pooling
- Timeout handling (5s)

### Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Web Framework** | Flask | 2.3.3 |
| **Deep Learning** | PyTorch | 2.5.1 |
| **Vision** | OpenCV | 4.8.1 |
| **Models** | TorchVision | 0.20.1 |
| **Database** | MongoDB | (optional) |
| **Frontend** | Bootstrap5 + JS | - |
| **Image Proc** | Pillow | 10.0+ |

---

## DEPLOYMENT GUIDE

### Prepare for Deployment

The project includes a complete deployment package in `Required/deploy/identix-deploy/`

#### What's in the deployment package?

```
deploy/identix-deploy/
├── app.py                 # Production-optimized app
├── deepfake_detector.py   # Deepfake module
├── mongodb_utils.py       # Database module
├── best_model.pth         # Model weights
├── requirements.txt       # Dependencies list
├── render.yaml           # Render configuration
├── Procfile              # Process definition
├── templates/            # HTML templates
├── static/               # CSS, JS, images
├── DEPLOYMENT.md         # Deployment instructions
└── CODE_REVIEW.md        # Code audit report
```

### Deploy to Render (Recommended)

#### Step 1: Prepare Code
```bash
# Copy deployment package to a new folder
cp -r Required/deploy/identix-deploy/ identix-render
cd identix-render
```

#### Step 2: Create GitHub Repository
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/identix.git
git push -u origin main
```

#### Step 3: Deploy to Render
1. Go to https://render.com/
2. Click "New +" → "Web Service"
3. Connect GitHub repository
4. Configure:
   - Name: `identix`
   - Environment: `Python 3.11`
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
5. Add environment variables (from .env)
6. Click "Create Web Service"

#### Step 4: Monitor Deployment
- Watch logs on Render dashboard
- App will be available at: `https://identix-[random].onrender.com`

### Deploy to Docker

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV FLASK_ENV=production
ENV PORT=5000

CMD ["python", "landmark_app.py"]
```

#### Step 2: Build Image
```bash
docker build -t identix:latest .
```

#### Step 3: Run Container
```bash
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e MONGODB_URI="" \
  -e SECRET_KEY="your-secret" \
  identix:latest
```

### Deploy to Heroku

```bash
# Install Heroku CLI
# Then:
heroku login
heroku create identix
heroku config:set FLASK_ENV=production
heroku config:set MONGODB_URI="..."
git push heroku main

# Monitor
heroku logs --tail
```

---

## TROUBLESHOOTING

### Common Issues & Solutions

#### ❌ "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
# Reinstall PyTorch
pip install torch torchvision torchaudio --force-reinstall

# Verify
python -c "import torch; print(torch.__version__)"
```

#### ❌ "Port 5000 already in use"

**Solution:**
```bash
# Option 1: Kill process on port 5000
# Windows PowerShell:
taskkill /PID $(netstat -ano | findstr :5000 | findstr LISTENING | % {$_.split()[-1]}) /F

# Option 2: Use different port
set PORT=5001
python landmark_app.py
```

#### ❌ "Model not found: models/best_model.pth"

**Solution:**
```bash
# Verify models directory exists
dir models/

# If missing, download from:
# https://huggingface.co/[identix-models]

# Or check path in code
python -c "import os; print(os.path.exists('models/best_model.pth'))"
```

#### ❌ "MongoDB connection timeout"

**Solution:**
```bash
# Option 1: Disable MongoDB
# Edit .env:
MONGODB_URI=""

# Option 2: Test connection
python -c "from mongodb_utils import get_client; get_client().server_info()"

# Option 3: Setup MongoDB
python scripts/setup_mongodb.py
```

#### ❌ "CUDA out of memory" (if using GPU)

**Solution:**
```bash
# Set device to CPU
set DEVICE=cpu

# Or reduce batch size
# Edit code to use batch_size=8 instead of 32
```

#### ❌ "Slow image processing (>1 second)"

**Causes & Solutions:**
- First run is slow (model loading) - ✅ Normal
- CPU processing is slower than GPU - Use GPU
- Image too large - Resize to <2MB
- System under heavy load - Close other apps

**Verify performance:**
```bash
# Run test
python tests/test_landmark_simple.py

# Should complete in <2 seconds for CPU
```

#### ❌ "Flask won't start - ImportError"

**Solution:**
```bash
# Reinstall all dependencies
pip install --upgrade -r deploy/identix-deploy/requirements.txt

# Clear cache
pip cache purge

# Verify Flask installation
python -c "from flask import Flask; print('✅ Flask OK')"
```

### Getting Help

**Resources:**
1. Check [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
2. Check [FILE_DESCRIPTIONS.md](FILE_DESCRIPTIONS.md)
3. Check [RULES_PAGE.md](RULES_PAGE.md)
4. Review CHANGE_LOG.md for recent fixes
5. Check test output: `python tests/test_landmark_simple.py`

### Debug Mode

```bash
# Enable debug logging
set FLASK_DEBUG=True
set FLASK_ENV=development

python landmark_app.py

# App will auto-reload on code changes
# More detailed error messages shown
```

---

## ADDITIONAL RESOURCES

### Quick Links

| Resource | Location | Purpose |
|----------|----------|---------|
| **Rules & Guidelines** | [RULES_PAGE.md](RULES_PAGE.md) | Project rules & naming conventions |
| **File Inventory** | [FILE_DESCRIPTIONS.md](FILE_DESCRIPTIONS.md) | Description of every file |
| **Quick Start** | [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Fast 5-minute guide |
| **Deployment** | [deploy/identix-deploy/DEPLOYMENT.md](deploy/identix-deploy/DEPLOYMENT.md) | Cloud deployment |
| **Technical Docs** | [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) | Technical architecture |
| **Change Log** | [docs/CHANGE_LOG.md](docs/CHANGE_LOG.md) | History of changes |

### Key Commands

```bash
# Start main app
python landmark_app.py

# Start video pipeline
python app.py

# Run tests
python tests/test_landmark_simple.py

# Setup MongoDB
python scripts/setup_mongodb.py

# List users
python scripts/list_users.py

# Clear users
python scripts/clear_users.py

# Debug predictions
python scripts/debug_prediction.py

# Install requirements
pip install -r deploy/identix-deploy/requirements.txt
```

---

## SUMMARY

IDENTIX is a **production-ready facial analysis platform** with:

✅ **Facial Landmark Detection** - Detect 11 facial features using BiSeNet
✅ **Deepfake Detection** - Analyze videos using 4 detection methods
✅ **Video Segmentation** - Process videos to extract and segment faces
✅ **Complete Documentation** - 40+ guides covering all aspects
✅ **Deployment Ready** - Push-button deployment to Render/Docker
✅ **User Friendly** - Web interface with drag-and-drop uploads

**Start in 3 commands:**
```bash
cd Required
pip install -r deploy/identix-deploy/requirements.txt
python landmark_app.py
```

Then visit: **http://localhost:5000**

---

**Last Updated**: January 31, 2026
**Version**: 1.0.0 Consolidated
**Status**: Production Ready ✅

