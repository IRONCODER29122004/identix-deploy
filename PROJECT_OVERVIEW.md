# IDENTIX - Comprehensive Project Overview & Setup Guide

**Project**: IDENTIX - Facial Landmark Detection & Deepfake Analysis
**Status**: Production Ready (January 2026)
**Version**: 1.0.0

---

## Quick Navigation

- **Active Development**: Go to `Required/` folder
- **Deployment**: See `Required/deploy/identix-deploy/`
- **Documentation**: See `Required/docs/`
- **Old/Archived Files**: See `Waste/` folder

---

## What is IDENTIX?

IDENTIX is a comprehensive facial analysis platform with three integrated applications:

### 1. **Facial Landmark Detection** (Primary)
- Detect and segment 11 facial landmark classes
- BiSeNet deep learning model (98% accuracy on test set)
- Supports images and videos
- Real-time webcam processing
- Multi-face detection and analysis

### 2. **Deepfake Detection** (Advanced)
- Analyze videos for manipulation signs
- 4 detection methods combined:
  - Temporal consistency analysis
  - Boundary artifact detection
  - Blink pattern analysis
  - Landmark stability metrics
- Confidence scoring (0-100%)
- Production-tested pipeline

### 3. **Video Segmentation Pipeline** (Optional)
- Extract frames from videos
- Detect faces across frames
- Apply segmentation to all faces
- Generate comparison videos
- Batch processing support

---

## Project Organization (Reorganized - Jan 2026)

```
IDENTIX/
├── Required/                  ← MAIN PROJECT (use this!)
│   ├── app.py                # Video pipeline Flask app
│   ├── landmark_app.py       # Main facial landmark detection
│   ├── deepfake_detector.py  # Deepfake analysis module
│   ├── models/               # Pre-trained weights
│   ├── data/                 # Runtime data (auto-created)
│   ├── templates/            # HTML UI files
│   ├── docs/                 # Documentation
│   ├── scripts/              # Utility scripts
│   ├── tests/                # Test scripts
│   └── deploy/               # GitHub/Cloud deployment
│
└── Waste/                     ← ARCHIVE (historical only)
    ├── notebooks/            # Jupyter experiments
    ├── Report_Submission/    # Project submission
    ├── Facial_Landmark_Project/ # Old structure copy
    └── [Other archives]
```

## Recent Changes (Project Reorganization)

### What Was Done
✅ Sorted all files into Required/ and Waste/
✅ Organized Required/ for optimal development
✅ Updated all path variables in code
✅ Created directory structure with subfolders
✅ Added comprehensive documentation

### Benefits
- Cleaner project structure
- Easier navigation
- Clear distinction between active & archived code
- Production-ready file organization
- Updated relative paths (models/, data/, etc.)

---

## Getting Started (3 Steps)

### Step 1: Install Dependencies
```bash
cd Required
pip install -r deploy/identix-deploy/requirements.txt
```

### Step 2: Configure Environment
```bash
# Create .env file
cp .env.example .env

# Edit .env and set:
# - MONGODB_URI (or leave empty to skip DB)
# - SECRET_KEY (generate: python -c "import secrets; print(secrets.token_hex(32))")
# - FLASK_ENV=development
```

### Step 3: Run the App
```bash
# Run facial landmark detection (RECOMMENDED)
python landmark_app.py

# OR run video pipeline
python app.py
```

The app will start on http://localhost:5000

---

## Key Features

### Facial Landmark Detection
```
Upload Image/Video → Face Detection → BiSeNet Segmentation → 11-Class Output
```
- Segments: Face outline, eyebrows, eyes, nose, lips, ears
- Confidence visualization
- Multi-face support
- Real-time processing

### Deepfake Detection
```
Upload Video → Extract Frames → Analyze Landmarks → Generate Report
```
- Verdict: Authentic / Suspicious / Likely Deepfake
- Confidence: 0-100%
- Detailed metrics for each analysis method
- Frame-by-frame breakdown

### Video Segmentation
```
Upload Video → Extract Frames → Detect Faces → Segment → Save Crops
```
- Extract N frames from video
- Landmark detection per frame
- Automatically crop best face regions
- Support for parallel inference

---

## Model Architecture

### BiSeNet (Primary Model)
- **Purpose**: Facial landmark segmentation
- **Classes**: 11 (face outline, eyebrows, eyes, nose, mouth, ears)
- **Input**: 512×512 RGB images
- **Output**: Semantic segmentation mask
- **Accuracy**: 98% on validation set
- **Speed**: ~50ms per image (GPU), ~200ms (CPU)
- **File**: `models/best_model.pth`

### Alternative Models Available
- **UNet** - `models/unet_model.keras` - Lightweight segmentation
- **DeepLab** - `models/deeplab_model.keras` - Advanced segmentation
- **ViT** - `models/vit_model.keras` - Vision Transformer approach

---

## File Structure Explained

### Core Files
| File | Purpose |
|------|---------|
| `landmark_app.py` | Main app (facial landmark detection) |
| `app.py` | Video pipeline app |
| `deepfake_detector.py` | Deepfake analysis engine |
| `pipeline_runner.py` | Video processing orchestration |
| `mongodb_utils.py` | Database management |
| `model.py` | Model architecture definitions |
| `resnet.py` | ResNet backbone components |

### Directories
| Directory | Contents |
|-----------|----------|
| `models/` | Pre-trained model weights |
| `data/` | Runtime uploads and processing |
| `templates/` | HTML/UI files |
| `docs/` | Technical documentation |
| `scripts/` | Utility scripts |
| `tests/` | Test suites |
| `deploy/` | Cloud deployment package |

---

## Configuration Reference

### Environment Variables (.env)
```bash
# Database
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/db

# Flask
SECRET_KEY=your-secret-key-here
FLASK_ENV=development  # or production

# Port
PORT=5000

# Processing
LAZY_LOAD=true         # Load models on first use
DEMO_MODE=false        # Use demo models if available
```

### Path Configuration
Automatically set in code:
```python
UPLOAD_FOLDER = 'data/uploads'           # User uploads
PIPELINES_FRAMES_DIR = 'data/pipelines_frames'  # Extracted frames
PIPELINES_CROPS_DIR = 'data/pipelines_crops'   # Face crops
model_paths = ['models/best_model_512.pth', 'models/best_model.pth']
```

---

## Usage Examples

### 1. Detect Landmarks in an Image
```python
from PIL import Image
import cv2
# See landmark_app.py for full implementation
# POST to /api/predict with image file
```

### 2. Analyze Video for Deepfakes
```python
# See deepfake_detector.py
# POST to /detect_deepfake with video file
# Returns: {verdict, confidence, metrics}
```

### 3. Process Video Pipeline
```python
# See pipeline_runner.py
# Extract frames, detect landmarks, save crops
# Supports batch inference with multiple models
```

---

## Deployment

### Local Development
```bash
python landmark_app.py    # Development mode
```

### Production (Render/Docker)
```bash
cd Required/deploy/identix-deploy/
# Follow DEPLOYMENT.md instructions
# Requires: Docker, Render account, MongoDB Atlas
```

See `Required/deploy/identix-deploy/DEPLOYMENT.md` for complete guide.

---

## API Endpoints

### Landmark Detection
- `POST /api/predict_image` - Detect landmarks in image
- `POST /api/predict_video` - Process video frames
- `GET /api/history` - Get user upload history

### Deepfake Detection
- `POST /detect_deepfake` - Analyze video for deepfakes
- `GET /health` - System health check

### Pipeline
- `POST /upload` - Upload video to pipeline
- `GET /results/<filename>` - Get pipeline results
- `GET /status/<filename>` - Check processing status

---

## Troubleshooting

### "Model not found"
- Check: `Required/models/best_model.pth` exists
- Solution: Download from repository or train new model

### "MongoDB connection error"
- Check: MONGODB_URI is set in .env
- Solution: Leave MONGODB_URI empty to run without DB

### "Port 5000 already in use"
- Check: Another app using port 5000
- Solution: `lsof -i :5000` then kill process, or change PORT in .env

### "Import errors" (torch, tensorflow, etc.)
- Check: All dependencies installed
- Solution: `pip install -r Required/deploy/identix-deploy/requirements.txt`

### "GPU not detected"
- Check: CUDA compatibility
- Solution: App falls back to CPU automatically

---

## Performance Benchmarks

| Operation | Time | Hardware |
|-----------|------|----------|
| Load model | 2-3s | CPU |
| Predict image | 50ms | GPU / 200ms CPU |
| Extract video frames | 1fps | Depends on video |
| Deepfake analysis | 1-2s per frame | CPU/GPU mixed |
| Web UI response | <500ms | Server-dependent |

---

## Development Workflow

### For Active Development
1. Work in `Required/` folder
2. Edit `landmark_app.py` or `deepfake_detector.py`
3. Test locally: `python landmark_app.py`
4. Check logs for errors
5. Commit to git (exclude .env and data/)

### For Deployment
1. Test in `Required/`
2. Copy to `Required/deploy/identix-deploy/`
3. Update `requirements.txt` if needed
4. Push to GitHub
5. Deploy via Render/Docker

### For Reference
1. Check `Waste/` for historical code
2. Review `Required/docs/` for technical details
3. See `Required/tests/` for test examples

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Total Python Files | 8 core + utilities |
| Lines of Code | ~15,000+ |
| HTML Templates | 16 pages |
| Pre-trained Models | 10 checkpoint files |
| Documentation Files | 20+ markdown files |
| Test Coverage | 5 test suites |
| Languages: Python, HTML, JavaScript |

---

## What You Should Know About This Project

✅ **What's Complete**
- Facial landmark detection with high accuracy
- Deepfake detection with multi-factor analysis
- Video processing pipeline
- User authentication & MongoDB integration
- Web UI with all features
- Full documentation
- Production deployment package

⚠️ **What's Partially Complete**
- Real-time webcam processing (works, but can be optimized)
- Audio-visual synchronization check (planned for deepfake)
- Multi-GPU support (currently single GPU)

🔄 **What's Experimental** (in Waste/)
- Vision Transformer models (alternative to BiSeNet)
- Cloudinary integration (replaced by local uploads)
- Old MediaPipe integration (removed)

---

## Contributing & Improvements

### For Future Work
1. **Accuracy**: Fine-tune models on larger datasets
2. **Speed**: Implement multi-GPU inference
3. **Features**: Add audio-visual deepfake checks
4. **UX**: Improve web interface responsiveness
5. **Deployment**: Add Kubernetes support

### Code Quality
- Uses PyTorch 2.5.1 (latest)
- Follows PEP 8 style guidelines
- Type hints for critical functions
- Error handling throughout

---

## Support & Documentation

For detailed information:

| Topic | Location |
|-------|----------|
| Quick Start | `Required/docs/QUICK_REFERENCE.md` |
| API Usage | `Required/docs/DEEPFAKE_README.md` |
| Deployment | `Required/deploy/identix-deploy/DEPLOYMENT.md` |
| Architecture | `Required/docs/PROJECT_DOCUMENTATION.md` |
| Models | `Required/docs/LANDMARK_README.md` |
| Code Review | `Required/deploy/identix-deploy/CODE_REVIEW.md` |

---

## Summary

**IDENTIX is production-ready for:**
- Facial landmark detection in images/videos
- Deepfake detection and analysis
- Video segmentation and processing
- User management and history tracking
- Cloud deployment (Render, Docker)

**Use Required/ folder for all development work.**
**Refer to Waste/ only for historical reference.**

Happy coding! 🚀

---

**Last Updated**: January 31, 2026
**Maintained By**: Project Team
**License**: [As per project LICENSE]
