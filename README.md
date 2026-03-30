# IDENTIX - Facial Landmark Detection & Deepfake Analysis

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)](https://pytorch.org)

**Production-ready facial landmark detection and deepfake analysis platform using BiSeNet and Deep Learning**

## 🎯 Features

- ✅ **Facial Landmark Detection** - 11-class segmentation with 91.58% accuracy
- ✅ **Deepfake Detection** - Multi-factor analysis with confidence scoring
- ✅ **Real-time Processing** - GPU support for live webcam analysis
- ✅ **Web UI** - Beautiful responsive interface with dark/light themes
- ✅ **User Authentication** - Secure MongoDB-backed account management
- ✅ **Cloud Deployment** - Ready for Render, Hugging Face, or self-hosted
- ✅ **Batch Processing** - Process entire videos frame-by-frame
- ✅ **Mobile Friendly** - Works on any device with a modern browser

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR-USERNAME/identix.git
cd identix
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your MongoDB URI and secret key
```

### 3. Run
```bash
python landmark_app.py
# Visit: http://localhost:5000/facial-landmarks
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 91.58% |
| **Inference Speed (CPU)** | 50-100ms |
| **Inference Speed (GPU)** | 15-30ms |
| **Model Size** | 49.4 MB |
| **Input Resolution** | 256×256 RGB |
| **Output Classes** | 11 facial regions |

## 🏗️ Architecture

```
Input Image (256×256)
    ↓
Face Detection (Haar Cascade)
    ↓
BiSeNet Inference (Context + Spatial Paths)
    ↓
11-Class Segmentation Mask
    ↓
Colorization & Confidence Scoring
```

**Model**: BiSeNet with ResNet-50 backbone
- **Accuracy**: 91.58% validation
- **Training**: 18K+ images from CelebAMask-HQ
- **Optimization**: Weighted cross-entropy + Dice loss

## 📁 Project Structure

```
├── landmark_app.py          Main Flask application
├── deepfake_detector.py     Deepfake analysis module
├── model.py                 BiSeNet architecture
├── templates/               HTML UI pages
├── static/                  CSS/JavaScript assets
├── models/                  Pre-trained weights
│   └── best_model.pth      BiSeNet model (49.4MB)
├── docs/                    Comprehensive documentation
├── tests/                   Test suites
├── scripts/                 Utility scripts
├── requirements.txt         Dependencies
├── render.yaml              Render.com config
└── Dockerfile               Docker containerization
```

## 📚 Documentation

- **[IDENTIX_MASTER_DOCUMENTATION.md](IDENTIX_MASTER_DOCUMENTATION.md)** - Complete reference (start here!)
- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Project vision & architecture
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Step-by-step deployment guide
- **[docs/API_ENDPOINTS.md](docs/API_ENDPOINTS.md)** - API reference
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Debugging guide

## 🔧 Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| **Render** | ✅ Supported | Free tier: $0/month (sleeps after 15min) |
| **Hugging Face Spaces** | ✅ Supported | Free tier: 16GB RAM, no limits |
| **Docker** | ✅ Supported | Self-hosted deployment |
| **Windows/Mac/Linux** | ✅ Supported | Local development & testing |

## 🔑 API Endpoints

**Analysis**:
- `POST /predict` - Analyze image for facial landmarks
- `POST /predict_video` - Process video frames
- `POST /detect_deepfake` - Analyze video for deepfake

**Authentication**:
- `POST /register` - Create user account
- `POST /login` - Authenticate user
- `GET /check-auth` - Check authentication status

See [docs/API_ENDPOINTS.md](docs/API_ENDPOINTS.md) for complete reference.

## 💻 System Requirements

**Minimum**:
- 4GB RAM
- Python 3.8+
- 2GB disk space

**Recommended**:
- 8GB+ RAM
- GPU (NVIDIA/Apple/Intel)
- SSD storage

## 📦 Dependencies

Core libraries:
- **PyTorch 2.5.1** - Deep learning framework
- **OpenCV** - Image processing
- **Flask 2.3.3** - Web framework
- **MongoDB** - User database
- **Pillow** - Image manipulation

See `requirements.txt` for complete list.

## 🧪 Testing

Run automated tests:
```bash
# Smoke test (model loading & inference)
python tests/TEST_landmark_simple.py

# Accuracy comparison
python tests/test_mediapipe_accuracy.py

# Regression testing
python tests/test_all_combinations.py

# API health check
python tests/test_ping.py
```

## 🚢 Deployment

### Option 1: Render (Recommended)
```bash
git push origin main
# Render auto-deploys from GitHub
# App available at: https://identix.onrender.com
```

### Option 2: Hugging Face Spaces
1. Create Space with Docker SDK
2. Link GitHub repository
3. Space auto-deploys with build
4. Access at: huggingface.co/spaces/YOUR-ID/identix

### Option 3: Docker
```bash
docker build -t identix:v2 .
docker run -p 5000:5000 -e MONGODB_URI="..." identix:v2
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions.

## 🔐 Security

- **Password Hashing**: SHA256 (upgradeable to bcrypt)
- **Session Management**: 30-day persistence
- **Input Validation**: XSS prevention, email validation
- **Environment Variables**: Sensitive data via .env
- **HTTPS**: Ready for production HTTPS deployment

## 📈 Performance Optimization

**For Production**:
- Use GPU when available (40-50x speedup)
- Enable model caching (load once, use many times)
- Use FP16 precision for faster inference
- Implement batch processing for videos
- CDN for static files

**Benchmarks**:
- Single image: 50-100ms (CPU), 15-30ms (GPU)
- Video (10 FPS extraction): 500-1000ms per frame
- Batch (10 images): 500-1000ms (CPUs), 150-300ms (GPU)

## 🎓 Model Architecture

**BiSeNet Overview**:
1. **Context Path** (semantic) - ResNet-50 backbone for global features
2. **Spatial Path** (detail) - Lightweight path for fine details
3. **Feature Fusion** - Combine context + spatial features
4. **Output Head** - 11-class classification per pixel

**Classes**:
0. Background, 1. Skin, 2. Left Eyebrow, 3. Right Eyebrow, 4. Left Eye, 
5. Right Eye, 6. Nose, 7. Upper Lip, 8. Inner Mouth, 9. Lower Lip, 10. Hair

## 🐛 Troubleshooting

**Model not loading**:
```bash
# Verify file exists and is not corrupted
ls -la models/best_model.pth

# Test model loading
python -c "from model import BiSeNet; m = BiSeNet(11); print('OK')"
```

**Out of memory**:
```python
# Use CPU instead of GPU
device = torch.device('cpu')

# Or reduce batch size
batch_size = 1
```

**Port already in use**:
```bash
# Use different port
python landmark_app.py --port 5001
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.

## 📊 Training Details

**Dataset**: CelebAMask-HQ (LaPa variant)
- 18.2K training images
- 3.6K validation images
- 256×256 resolution

**Training Configuration**:
- Batch size: 8
- Optimizer: Adam (lr=0.0001)
- Loss: Weighted CrossEntropy + Dice
- Best epoch: 11/50 (91.58% validation accuracy)
- Training time: ~4 hours (GPU)

See [docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md) for detailed training guide.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name** - Facial Landmark Detection & Deepfake Analysis  
- GitHub: [@YOUR-USERNAME](https://github.com/YOUR-USERNAME)
- Email: your-email@example.com

## 🙏 Acknowledgments

- **CelebAMask-HQ** - Dataset providers
- **BiSeNet Authors** - Architecture research
- **PyTorch Team** - Deep learning framework
- **OpenCV** - Computer vision library

## 📞 Support

- 📖 **Documentation**: See [IDENTIX_MASTER_DOCUMENTATION.md](IDENTIX_MASTER_DOCUMENTATION.md)
- 🐛 **Issues**: Open GitHub issue for bugs
- 💬 **Discussions**: GitHub Discussions for questions
- 📧 **Email**: your-email@example.com

## 🗺️ Roadmap

- [ ] Multi-GPU support
- [ ] Mobile app (iOS/Android)
- [ ] Advanced deepfake detection methods
- [ ] Real-time video streaming (WebRTC)
- [ ] Federated learning for privacy
- [ ] Admin dashboard

---

**Made with ❤️ by [Your Name]**

⭐ If this project helps you, please consider giving it a star!

