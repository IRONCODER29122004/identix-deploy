# Deploy IDENTIX to Hugging Face Spaces

## Why Hugging Face Spaces?
- **2GB RAM** (vs Render Free 512MB) - easily handles your 138MB model
- **Free forever** for public projects
- **Built for ML demos** - optimized for PyTorch apps
- **Simple deployment** - push to repo and done

## Quick Setup (5 minutes)

### 1. Create Hugging Face Account
- Go to https://huggingface.co/join
- Sign up (free)

### 2. Create New Space
- Go to https://huggingface.co/spaces
- Click "Create new Space"
- Name: `identix-facial-landmarks`
- License: `apache-2.0`
- SDK: **Gradio** (select from dropdown)
- Hardware: **CPU basic (free)** - 2GB RAM
- Visibility: **Public** (required for free tier)

### 3. Connect GitHub Repo
Option A: Push directly to HF Space repo
```powershell
# Add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/identix-facial-landmarks
git push hf main
```

Option B: Link GitHub repo (easier)
- In Space settings → Repository
- Click "Link to GitHub"
- Select `IRONCODER29122004/identix-deploy`
- Auto-sync enabled

### 4. Add Secrets (Environment Variables)
- In your Space → Settings → Repository secrets
- Add these secrets:
  - `MONGODB_URI`: your Atlas connection string
  - `SECRET_KEY`: random secret (generate with the button)
  - `FLASK_ENV`: `production`

### 5. Files Needed (Already in Repo)
- ✅ `app.py` - main Flask app
- ✅ `requirements.txt` - dependencies
- ✅ `best_model.pth` - model weights (via Git LFS)
- ✅ `.gitattributes` - LFS tracking
- ✅ `templates/` - HTML templates
- ✅ Supporting files (`mongodb_utils.py`, `deepfake_detector.py`)

### 6. Deploy
- HF automatically detects `app.py` and `requirements.txt`
- Build starts immediately
- Check logs for "Running on http://0.0.0.0:7860"
- Your app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/identix-facial-landmarks`

## Expected Build Time
- Initial: ~5-8 minutes (installing PyTorch, OpenCV)
- Rebuilds: ~2-3 minutes (cached dependencies)

## Post-Deployment
- Test `/health` endpoint
- Upload an image for segmentation
- Monitor usage in Space analytics

## Advantages Over Render
- ✅ 4x more RAM (2GB vs 512MB)
- ✅ No cold starts for active spaces
- ✅ Built-in GPU upgrade path ($0.60/hr T4)
- ✅ Better for ML community visibility
- ✅ Direct Gradio integration option

## Troubleshooting
- **Build fails**: Check logs in Space → App → Logs
- **Model not loading**: Verify `best_model.pth` pushed via LFS
- **MongoDB errors**: Check secrets are set correctly
- **Port issues**: HF expects port 7860 (app.py already configured)

## Custom Domain (Optional)
- Spaces support custom domains
- Settings → Repository → Custom domain
- Add CNAME: `your-domain.com` → `YOUR_USERNAME-identix-facial-landmarks.hf.space`

## Upgrade Path
- Start: Free 2GB CPU
- If popular: Upgrade to T4 GPU ($0.60/hr, pause when idle)
- Or: Zero GPU (free GPU for popular spaces)
