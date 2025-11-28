Run locally on Windows with Waitress to avoid reloader issues.
Example local run (Windows):
```
python -m pip install -r requirements.txt
python -c "import os; os.environ['FLASK_ENV']='production'; os.environ['MONGODB_URI']='your-uri'; os.environ['SECRET_KEY']='your-secret'; from app import app; from waitress import serve; serve(app, host='0.0.0.0', port=5000)"
```
Example server run (Linux/Render):
```
gunicorn app:app -b 0.0.0.0:${PORT} -w 2
```
# IDENTIX Deployment Guide

## 📦 Deployment Package Contents

This folder contains everything needed to deploy IDENTIX (Facial Landmark Detection & Deepfake Analysis) to Render.

### Files Included:
- `app.py` - Main Flask application (production-ready)
- `mongodb_utils.py` - MongoDB connection utilities
- `deepfake_detector.py` - Deepfake detection module
- `best_model.pth` - Trained BiSeNet model (11-class facial landmarks)
- `templates/` - All HTML templates
- `requirements.txt` - Python dependencies
- `render.yaml` - Render platform configuration
- `.env.example` - Environment variable template
- `.gitignore` - Git ignore rules

---

## 🚀 Quick Deployment to Render

### Prerequisites
1. **MongoDB Atlas Account** - Free tier available at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. **GitHub Account** - To host your repository
3. **Render Account** - Free tier at [render.com](https://render.com)

---

## Step 1: Prepare MongoDB Database

### 1.1 Create MongoDB Atlas Cluster (if not already done)
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Sign up or log in
3. Create a new cluster (free M0 tier is sufficient)
4. Wait for cluster to be created (~5 minutes)

### 1.2 Configure Database Access
1. In Atlas dashboard, go to **Database Access**
2. Click **Add New Database User**
   - Authentication Method: Password
   - Username: `identix_app`
   - Password: Generate secure password (save it!)
   - Database User Privileges: **Read and write to any database**
3. Click **Add User**

### 1.3 Configure Network Access
1. Go to **Network Access**
2. Click **Add IP Address**
3. Choose **Allow Access from Anywhere** (0.0.0.0/0)
   - This is required for Render's dynamic IPs
4. Click **Confirm**

### 1.4 Get Connection String
1. Go to **Database** → **Connect**
2. Choose **Connect your application**
3. Driver: Python, Version: 3.11 or later
4. Copy the connection string:
   ```
   mongodb+srv://identix_app:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
5. **IMPORTANT**: Replace `<password>` with your actual password
6. Add database name before the `?`: `...mongodb.net/facial_landmark_db?retryWrites=true...`

---

## Step 2: Prepare GitHub Repository

### 2.1 Create New Repository
1. Go to [GitHub](https://github.com)
2. Click **New Repository**
3. Name: `identix` (or your preferred name)
4. Visibility: Public or Private (both work with Render)
5. **Do NOT** initialize with README, .gitignore, or license (we have those)
6. Click **Create Repository**

### 2.2 Push Code to GitHub

Open PowerShell/Terminal in the `identix-deploy` folder and run:

```powershell
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial deployment setup for IDENTIX"

# Add remote (replace YOUR_USERNAME and YOUR_REPO)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Note**: Make sure `.env` is NOT committed (it's in .gitignore)

---

## Step 3: Deploy to Render

### 3.1 Create New Web Service
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub account if not already done
4. Select your `identix` repository
5. Click **Connect**

### 3.2 Configure Service

**Basic Settings:**
- **Name**: `identix` (or your preferred name)
- **Region**: Choose closest to your users
- **Branch**: `main`
- **Root Directory**: Leave blank (or `.` if needed)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`

**Instance Type:**
- Select **Free** ($0/month)
- Note: Free tier sleeps after 15 min of inactivity

### 3.3 Add Environment Variables

Click **Advanced** → **Add Environment Variable**

Add these variables:

| Key | Value | Notes |
|-----|-------|-------|
| `MONGODB_URI` | Your MongoDB connection string | From Step 1.4 |
| `SECRET_KEY` | Generate with command below | Keep secret! |
| `FLASK_ENV` | `production` | Disables debug mode |
| `PYTHON_VERSION` | `3.11.0` | Python version |

**Generate SECRET_KEY:**
```python
# Run in Python:
import secrets
print(secrets.token_hex(32))
```

### 3.4 Deploy
1. Click **Create Web Service**
2. Render will start building (takes 5-10 minutes first time)
3. Watch the logs for any errors
4. Once deployed, you'll get a URL like: `https://identix.onrender.com`

---

## Step 4: Verify Deployment

### 4.1 Check Health Endpoint
Visit: `https://your-app.onrender.com/health`

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### 4.2 Test Authentication
1. Visit your app URL
2. Click **Sign Up**
3. Create an account
4. Try logging in
5. Upload a test image

---

## 🔒 Security Improvements (Post-Deployment)

After successful deployment, consider these security enhancements:

### 1. Upgrade Password Hashing
Current: SHA256 (basic)
Recommended: bcrypt or argon2

Add to `requirements.txt`:
```
bcrypt==4.1.1
```

Update `app.py` registration:
```python
import bcrypt
# Replace SHA256 hashing with:
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
```

### 2. Add Rate Limiting
Install Flask-Limiter:
```bash
pip install Flask-Limiter
```

Add to `app.py`:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    # existing code
```

### 3. Add CSRF Protection
Install Flask-WTF:
```bash
pip install Flask-WTF
```

### 4. Enable HTTPS Only
Already enabled by Render automatically!

---

## 🐛 Troubleshooting

### Issue: "Build Failed"
**Solution**: Check logs for missing dependencies. Common fixes:
- Ensure `requirements.txt` is correct
- Check Python version compatibility
- Verify model file `best_model.pth` exists

### Issue: "MongoDB Connection Failed"
**Solution**:
1. Verify connection string is correct
2. Check MongoDB Atlas network access (0.0.0.0/0 allowed)
3. Verify database user credentials
4. Ensure database name is in connection string

### Issue: "Model Not Loading"
**Solution**:
1. Check if `best_model.pth` is in repository
2. Verify file wasn't corrupted during git push
3. Check logs for torch/CUDA errors
4. Model requires ~100MB, ensure it's committed

### Issue: "App Sleeps on Free Tier"
**Solution**: This is normal for Render free tier
- App sleeps after 15 minutes of inactivity
- First request after sleep takes ~30 seconds
- Upgrade to paid tier for always-on service

### Issue: "Slow Performance"
**Solution**:
- Free tier has limited CPU/RAM
- Large model files take time to load
- Consider paid tier for better performance
- Optimize image sizes before upload

---

## 📊 Monitoring & Logs

### View Logs
1. Go to Render Dashboard
2. Select your service
3. Click **Logs** tab
4. Monitor real-time logs for errors

### Check Metrics
- **Events**: Deployment history
- **Metrics**: CPU, memory usage
- **Environment**: Verify env variables

---

## 🔄 Updating Deployment

### Update Code
```bash
# Make changes to code
git add .
git commit -m "Description of changes"
git push origin main
```

Render automatically redeploys on push to `main` branch.

### Update Dependencies
1. Modify `requirements.txt`
2. Commit and push
3. Render rebuilds automatically

### Update Environment Variables
1. Go to Render Dashboard
2. Select your service
3. **Environment** → Edit variables
4. Save (triggers redeployment)

---

## 💰 Cost Considerations

### Free Tier Limits
- **Render**: 750 hours/month, sleeps after 15 min
- **MongoDB Atlas**: 512MB storage, shared cluster
- **Total Cost**: $0/month

### Paid Tier Benefits
- **Render Starter ($7/month)**:
  - Always-on service
  - Better performance
  - More resources
- **MongoDB M2 ($9/month)**:
  - 2GB storage
  - Better performance

---

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [MongoDB Atlas Docs](https://docs.atlas.mongodb.com)
- [Flask Deployment Guide](https://flask.palletsprojects.com/en/latest/deploying/)
- [Gunicorn Documentation](https://docs.gunicorn.org)

---

## 🆘 Getting Help

### Common Commands

**Check Python version locally:**
```bash
python --version
```

**Test app locally:**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (Windows PowerShell)
$env:MONGODB_URI="your_connection_string"
$env:SECRET_KEY="your_secret_key"

# Run app
python app.py
```

**Generate new secret key:**
```python
import secrets
print(secrets.token_hex(32))
```

---

## ✅ Deployment Checklist

Before deploying, verify:

- [ ] MongoDB Atlas cluster created
- [ ] Database user created with correct permissions
- [ ] Network access allows 0.0.0.0/0
- [ ] Connection string includes database name
- [ ] GitHub repository created and code pushed
- [ ] `.env` file NOT committed to git
- [ ] `best_model.pth` file is in repository
- [ ] All template files copied correctly
- [ ] Render account created
- [ ] Environment variables configured in Render
- [ ] Health endpoint returns `{"status": "healthy"}`
- [ ] Can register new user
- [ ] Can log in successfully
- [ ] Can upload and process images

---

## 🎉 Success!

Your IDENTIX application is now deployed and accessible worldwide!

Share your app URL: `https://your-app-name.onrender.com`

---

**Project**: IDENTIX - Facial Landmark Detection & Deepfake Analysis  
**Version**: 1.0  
**Deployment Platform**: Render  
**Database**: MongoDB Atlas  
**Framework**: Flask + PyTorch
