# Deploy IDENTIX to Hugging Face Spaces

**NEW: Comprehensive guide available in [HF_SPACES_DEPLOYMENT.md](HF_SPACES_DEPLOYMENT.md)** ← Start here!

## Why Hugging Face Spaces?
- **2GB RAM** (vs Render Free 512MB) - easily handles your 138MB model
- **Free forever** for public projects
- **Built for ML demos** - optimized for PyTorch apps
- **Simple deployment** - push to repo and done
- **No cold starts** - app stays warm (unlike Render sleep)
- **Better community** - showcase to ML audience

## ⚡ Quick Setup (10 minutes)

### 1. Create Hugging Face Account
- Go to https://huggingface.co/join
- Sign up (free)

### 2. Create New Space
- Go to https://huggingface.co/spaces
- Click "Create new Space"
- Name: `identix-facial-landmarks`
- License: `apache-2.0`
- SDK: **Docker** (recommended for Flask)
- Hardware: **CPU basic (free)** - 2GB RAM
- Visibility: **Public** (required for free tier)

### 3. Connect GitHub Repo (EASIEST)
In your Space → Settings → Repository:
- Click "Link to GitHub"
- Select: `IRONCODER29122004/identix-deploy`
- Enable "Auto-sync from GitHub"
- ✅ Space rebuilds automatically when you push!

### 4. Add Secrets (Optional - only if using MongoDB)
In Space → Settings → Repository secrets:
- `MONGODB_URI`: your MongoDB connection string
- `SECRET_KEY`: generate random secret
- `FLASK_ENV`: `production`

App works fine **without** MongoDB (user auth disabled).

### 5. Deploy
✅ **Automatic** - Files already in your repo:
- `app.py` (UPDATED with HF port support)
- `requirements.txt` (all dependencies)
- `best_model.pth` (138MB model via Git LFS)
- `.gitattributes` (LFS tracking)
- `templates/` + `static/` (UI files)

### 6. Build & Launch
Space automatically:
```
✓ Detects Flask app
✓ Installs dependencies (5-8 min first time)
✓ Downloads model via Git LFS
✓ Starts Flask on port 7860
✓ App goes LIVE!
```

Your app: `https://huggingface.co/spaces/YOUR_USERNAME/identix-facial-landmarks`

## ✅ What Was Fixed

### Port Configuration (FIXED!)
**Before:** App hardcoded to port 5000  
**Now:** Auto-detects environment and uses correct port:
- 🟢 Hugging Face Spaces → Port 7860
- 🟢 Render → Uses PORT env var
- 🟢 Local dev → Defaults to 5000

Code detail: Updated [app.py](app.py) with smart port detection.

### Platform Auto-Detection (NEW!)
App now detects if running on HF Spaces and adjusts automatically:
```python
is_huggingface = os.environ.get('SPACE_ID') is not None
```

## 🧪 Post-Deployment Testing

**Test 1: Health Check**
```
GET https://YOUR_SPACE/health
→ {"status": "healthy", "model_loaded": true}
```

**Test 2: Upload Image**
- Go to `/facial-landmarks`
- Upload a photo
- See 11-class facial segmentation

**Test 3: Analyze Video**
- Go to `/video-analysis`
- Upload a video
- Get deepfake detection analysis

## 📊 Comparison: HF vs Render

| Feature | Render | HF Spaces |
|---------|--------|-----------|
| **RAM** | 512MB | 2GB ✅ |
| **Cold starts** | 30 seconds | None ✅ |
| **Idle sleep** | After 15 min | Never ✅ |
| **Build time** | Quick | 5-8 min |
| **GPU support** | Paid | Free tier available |
| **Uptime** | 99.9% | 99.9% |

## 🔧 Troubleshooting

**Build takes too long / times out**
```
→ Normal! PyTorch download is large
→ Usually succeeds on 2nd attempt
→ Check logs: Space → App → Logs
```

**"Module not found" errors**
```
→ Verify all files pushed: git status
→ Check requirements.txt has all imports
→ Common: missing opencv-python-headless
```

**MongoDB connection error**
```
→ THIS IS NORMAL without MongoDB secret!
→ App still works perfectly
→ User auth just disabled
→ Add MongoDB secret to enable user accounts
```

**Port/network issues**
```
→ FIXED! App auto-detects environment
→ Check logs show: "Platform: Hugging Face Spaces"
→ Check logs show: "Server starting on port 7860"
```

**Model file missing**
```powershell
# Verify Git LFS tracking
git lfs ls-files | grep best_model

# If missing, fix it:
git rm --cached best_model.pth
git add best_model.pth
git push origin main
```

## 🚀 Continuous Deployment

With auto-sync enabled, just push changes:
```powershell
# Make changes
# Update templates, fix bugs, etc

# Push to GitHub
git add .
git commit -m "Update: Add new feature"
git push origin main

# Space auto-rebuilds in ~5 minutes
# Changes live after build completes
```

## 📈 Expected Performance

| Metric | Value |
|--------|-------|
| Inference time | 100-300ms |
| API response | <100ms |
| Concurrent users | 2-5 (free) |
| Uptime | 99.9% |
| Cold start | 0ms (no sleep) |

## 🔐 Security

Already implemented:
- ✅ SHA256 password hashing
- ✅ Input validation
- ✅ XSS prevention
- ✅ Environment variables for secrets

## 💰 Cost

| Item | Cost |
|------|------|
| CPU Inference | **FREE** |
| 2GB RAM | **FREE** |
| Storage | **FREE** |
| Bandwidth | **FREE** |
| **Monthly Total** | **$0** |

Optional GPU: $0.60/hr (T4) - only if you upgrade

## 📋 Deployment Checklist

- [ ] Create HF account
- [ ] Create new Space (Docker, CPU basic, Public)
- [ ] Link GitHub repo
- [ ] Add MongoDB secret (optional)
- [ ] Space builds successfully
- [ ] Test `/health` endpoint
- [ ] Test image upload
- [ ] Test video analysis
- [ ] Monitor Space analytics
- [ ] Share Space URL!

## 🎯 Next Steps

1. **Today**: Create HF Space & link GitHub (10 min)
2. **Tomorrow**: Test all features, share with friends!
3. **Later**: Monitor analytics, plan GPU upgrade if popular

## 📚 Full Documentation

**For complete setup guide and advanced config:**
👉 See [HF_SPACES_DEPLOYMENT.md](HF_SPACES_DEPLOYMENT.md)

**Your Space will be available at:**
```
https://huggingface.co/spaces/YOUR_USERNAME/identix-facial-landmarks
```

---

**🚀 Ready to deploy? Start with Step 1 above!**

**Questions?** Check HF_SPACES_DEPLOYMENT.md for detailed troubleshooting.
