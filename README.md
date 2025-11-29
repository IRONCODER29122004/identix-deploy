---
title: IDENTIX Facial Landmarks
emoji: 👤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# IDENTIX - Facial Landmark Detection & Deepfake Analysis

🚀 **Production-Ready Deployment Package**

## What is IDENTIX?

IDENTIX is an advanced AI-powered web application that performs:
- **Facial Landmark Detection** - 11-class segmentation using BiSeNet architecture
- **Deepfake Detection** - Multi-factor analysis for video authenticity
- **User Authentication** - Secure MongoDB-backed user management
- **Real-time Processing** - Image and video analysis with instant results

## 🎯 Key Features

- ✅ High-accuracy facial landmark segmentation (trained BiSeNet model)
- ✅ Multi-person detection and tracking
- ✅ Deepfake detection with confidence scoring
- ✅ Secure user registration and authentication
- ✅ Beautiful responsive UI with light/dark themes
- ✅ MongoDB Atlas integration for user data
- ✅ Production-ready Flask application
- ✅ Optimized for Render deployment

## 📋 Quick Start

**For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)**

### Local Testing

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB connection string
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open browser:**
   ```
   http://localhost:5000
   ```

## 🏗️ Architecture

### Tech Stack
- **Backend**: Flask (Python web framework)
- **ML Framework**: PyTorch (BiSeNet model)
- **Database**: MongoDB Atlas (user authentication)
- **Image Processing**: OpenCV, PIL
- **Server**: Gunicorn (production)

### Model Details
- **Architecture**: BiSeNet (Bilateral Segmentation Network)
- **Backbone**: ResNet-50
- **Classes**: 11 facial landmarks
  1. Skin
  2. Left Eyebrow
  3. Right Eyebrow
  4. Left Eye
  5. Right Eye
  6. Nose
  7. Upper Lip
  8. Inner Mouth
  9. Lower Lip
  10. Hair
  11. Background

### Security Features
- SHA256 password hashing (upgradable to bcrypt)
- Email validation with regex
- Input sanitization (XSS prevention)
- Unique email constraint in MongoDB
- Session-based authentication
- Environment variable configuration

## 📁 Project Structure

```
identix-deploy/
├── app.py                  # Main Flask application
├── mongodb_utils.py        # Database connection utilities
├── deepfake_detector.py    # Deepfake analysis module
├── best_model.pth          # Trained BiSeNet model (~95MB)
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
├── DEPLOYMENT.md           # Detailed deployment guide
├── README.md               # This file
└── templates/              # HTML templates
    ├── index.html          # Landing page
    ├── image_analysis.html # Image upload page
    ├── video_analysis.html # Video upload page
    ├── deepfake_detection.html
    └── ... (other pages)
```

## 🔑 Environment Variables

Required variables (see `.env.example`):

| Variable | Description | Example |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB Atlas connection string | `mongodb+srv://user:pass@cluster.mongodb.net/db` |
| `SECRET_KEY` | Flask session secret | Generate with `secrets.token_hex(32)` |
| `FLASK_ENV` | Environment mode | `production` or `development` |
| `PORT` | Server port (auto-set by Render) | `5000` |

## 🧪 API Endpoints

### Authentication
- `POST /register` - Create new user account
- `POST /login` - Authenticate user
- `POST /logout` - End user session
- `GET /check-auth` - Check authentication status

### Analysis
- `POST /predict` - Analyze image for facial landmarks
- `POST /predict_video` - Process video frames
- `POST /detect_deepfake` - Analyze video for deepfake

### Other
- `GET /health` - Health check endpoint
- `GET /` - Landing page
- `GET /facial-landmarks` - Image analysis UI
- `GET /video-analysis` - Video analysis UI

## 🛡️ Security Considerations

### Current Implementation
✅ Password hashing (SHA256)  
✅ Input validation  
✅ XSS prevention  
✅ Secure session management  
✅ Environment-based secrets  

### Recommended Improvements
⚠️ Upgrade to bcrypt/argon2 password hashing  
⚠️ Add rate limiting (Flask-Limiter)  
⚠️ Add CSRF protection (Flask-WTF)  
⚠️ Implement password reset functionality  
⚠️ Add email verification  

See [DEPLOYMENT.md](DEPLOYMENT.md) for implementation details.

## 📊 Performance

### Model Performance
- **Input Size**: 256×256 pixels
- **Output**: 11-class segmentation mask
- **Inference Time**: ~100-300ms per image (CPU)

### Resource Requirements
- **RAM**: ~1GB minimum
- **Storage**: ~500MB (model + dependencies)
- **CPU**: Single core sufficient for free tier

### Render Free Tier
- 750 hours/month
- Sleeps after 15 minutes inactivity
- First request after sleep: ~30 seconds

## 🐛 Known Issues & Limitations

1. **Free Tier Sleep**: App sleeps on inactivity (Render limitation)
2. **Model Size**: Large model file (~95MB) affects cold start
3. **SHA256 Hashing**: Basic password security (upgrade recommended)
4. **No Rate Limiting**: Vulnerable to brute force (mitigation available)
5. **CPU Inference**: Slower than GPU (acceptable for free tier)

## 🔄 Updates & Maintenance

### Updating Code
```bash
git add .
git commit -m "Update description"
git push origin main
# Render auto-deploys
```

### Monitoring
- Check Render dashboard for logs
- Monitor `/health` endpoint
- Review MongoDB Atlas metrics

## 📖 Documentation

- **Deployment Guide**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Render Setup Summary**:
   - Build: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start: `gunicorn app:app -b 0.0.0.0:$PORT -w 2`
   - Health: `/health`
   - Env vars: `MONGODB_URI`, `SECRET_KEY`, `FLASK_ENV=production`
   - Plan: `standard` recommended for PyTorch model
- **Code Documentation**: Inline docstrings in Python files
- **API Reference**: See "API Endpoints" section above

## 🤝 Contributing

This is a capstone project. For modifications:
1. Test locally first
2. Update requirements.txt if adding dependencies
3. Document changes in commit messages
4. Check security implications

## 📝 License

Educational/Academic Project - Capstone 4-1

## 👥 Authors

Capstone Team - Facial Landmark Detection Project

## 🙏 Acknowledgments

- BiSeNet architecture inspiration
- MongoDB Atlas free tier
- Render free hosting
- PyTorch framework
- Flask community

---

**Ready to deploy?** Follow the comprehensive guide in [DEPLOYMENT.md](DEPLOYMENT.md)

**Questions?** Check troubleshooting section in deployment guide.

**Need help?** Review the security and performance sections above.
