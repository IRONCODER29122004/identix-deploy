#!/usr/bin/env python3
"""
HYBRID ENSEMBLE SETUP LAUNCHER
Run this script to automatically set up hybrid deepfake detection
"""

import os
import sys
import subprocess
import platform

class HybridSetup:
    """Automated setup for hybrid deepfake detection"""
    
    def __init__(self):
        self.os_type = platform.system()
        self.python_path = sys.executable
        self.errors = []
        self.warnings = []
    
    def header(self, text):
        """Print colored header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70)
    
    def step(self, num, text):
        """Print step"""
        print(f"\n[{num}] {text}...")
    
    def success(self, text):
        """Print success message"""
        print(f"  ✓ {text}")
    
    def error(self, text):
        """Print error message"""
        print(f"  ✗ {text}")
        self.errors.append(text)
    
    def warning(self, text):
        """Print warning message"""
        print(f"  ⚠ {text}")
        self.warnings.append(text)
    
    def check_python(self):
        """Check Python version"""
        self.step(1, "Checking Python version")
        
        version_info = sys.version_info
        if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 8):
            self.error(f"Python 3.8+ required (you have {version_info.major}.{version_info.minor})")
            return False
        
        self.success(f"Python {version_info.major}.{version_info.minor}.{version_info.micro}")
        return True
    
    def install_dependencies(self):
        """Install required packages"""
        self.step(2, "Installing dependencies")
        
        requirements_file = "requirements_hybrid.txt"
        
        if not os.path.exists(requirements_file):
            self.error(f"Could not find {requirements_file}")
            return False
        
        try:
            subprocess.check_call([
                self.python_path, "-m", "pip", "install", 
                "-r", requirements_file, "-q"
            ])
            self.success("Dependencies installed")
            return True
        except subprocess.CalledProcessError:
            self.error("Failed to install dependencies")
            return False
    
    def download_models(self):
        """Download pre-trained models"""
        self.step(3, "Downloading pre-trained models")
        
        model_dir = "Required/models"
        os.makedirs(model_dir, exist_ok=True)
        
        xception_path = os.path.join(model_dir, "xception_ff.pth")
        efficientnet_path = os.path.join(model_dir, "efficientnet_dfdc.pth")
        
        if os.path.exists(xception_path):
            self.success(f"Xception already downloaded ({os.path.getsize(xception_path) / 1024 / 1024:.0f} MB)")
        else:
            print("  Downloading Xception (119 MB)... This may take 2-5 minutes")
            try:
                import gdown
                gdown.download(
                    "https://drive.google.com/uc?id=1z_fwWnuAjeKwz65DO94STg9vv47kWL8P",
                    output=xception_path,
                    quiet=False
                )
                if os.path.exists(xception_path):
                    self.success(f"Xception downloaded ({os.path.getsize(xception_path) / 1024 / 1024:.0f} MB)")
                else:
                    self.warning("Xception download failed - will work without it")
            except Exception as e:
                self.warning(f"Could not auto-download Xception: {e}")
                print("  → Run manually: python download_models.py setup")
    
    def verify_files(self):
        """Verify all required files exist"""
        self.step(4, "Verifying files")
        
        required_files = [
            "Required/deepfake_detector.py",
            "Required/ml_deepfake_detector.py",
            "Required/hybrid_detector.py",
            "Required/flask_hybrid_api.py",
            "Required/download_models.py"
        ]
        
        all_exist = True
        for file_path in required_files:
            if os.path.exists(file_path):
                self.success(f"Found {file_path.split('/')[-1]}")
            else:
                self.error(f"Missing {file_path}")
                all_exist = False
        
        return all_exist
    
    def check_gpu(self):
        """Check GPU availability"""
        self.step(5, "Checking GPU availability")
        
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.success(f"GPU detected: {device_name}")
                self.success("Processing will be 2-3x faster!")
                return True
            else:
                self.warning("No NVIDIA GPU found. Detection will use CPU (slower but fine)")
                return False
        except ImportError:
            self.warning("PyTorch not loaded yet")
            return False
    
    def show_next_steps(self):
        """Show next steps"""
        self.header("NEXT STEPS")
        
        print("""
1. START FLASK BACKEND:
   cd Required
   python flask_hybrid_api.py
   
   This starts the API server at http://localhost:5000

2. TEST THE API:
   - In another terminal:
   curl -X GET http://localhost:5000/api/health
   
   Should return: {"status": "online", ...}

3. ADD CHROME EXTENSION:
   - See: HYBRID_ENSEMBLE_SETUP.md
   - Copy manifest.json, popup.html, popup.js, styles.css
   - Load unpacked in chrome://extensions/

4. TEST DETECTION:
   - Upload a video in extension
   - Should see: ✅ AUTHENTIC or ❌ DEEPFAKE
   - Check confidence: 90%+

5. INTEGRATE WITH YOUR APP:
   - See: flask_hybrid_api.py for API examples
   - Same pattern works for generation endpoint
""")
    
    def run(self):
        """Run complete setup"""
        self.header("HYBRID ENSEMBLE SETUP")
        
        print("""
This script will:
1. Check Python version
2. Install dependencies
3. Download pre-trained models (~200 MB)
4. Verify all files
5. Check GPU availability

Let's go! ⚡
""")
        
        # Run checks
        if not self.check_python():
            self.error("Setup failed: Python version incompatible")
            return False
        
        if not self.verify_files():
            self.warning("Some files missing - continue anyway?")
        
        if not self.install_dependencies():
            print("\nManual install:")
            print(f"  {self.python_path} -m pip install -r requirements_hybrid.txt")
        
        self.download_models()
        self.check_gpu()
        
        # Summary
        self.header("SETUP SUMMARY")
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"   - {error}")
        
        if self.warnings:
            print("\n⚠ WARNINGS:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if not self.errors:
            print("\n✓ Setup complete! Ready to use.")
        else:
            print("\n✗ Setup incomplete. Fix errors above.")
            return False
        
        self.show_next_steps()
        return True


# ============================================================================
# QUICK TEST AFTER SETUP
# ============================================================================

def test_imports():
    """Test that all imports work"""
    print("\n[TEST] Testing imports...")
    
    try:
        from deepfake_detector import DeepfakeDetector
        print("  ✓ deepfake_detector imported")
    except ImportError as e:
        print(f"  ✗ deepfake_detector failed: {e}")
        return False
    
    try:
        from ml_deepfake_detector import XceptionDeepfakeDetector
        print("  ✓ ml_deepfake_detector imported")
    except ImportError as e:
        print(f"  ✗ ml_deepfake_detector failed: {e}")
    
    try:
        from hybrid_detector import HybridDeepfakeDetector
        print("  ✓ hybrid_detector imported")
    except ImportError as e:
        print(f"  ✗ hybrid_detector failed: {e}")
        return False
    
    try:
        import flask
        print("  ✓ flask imported")
    except ImportError as e:
        print(f"  ✗ flask failed: {e}")
        return False
    
    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    # Change to Required directory
    if os.path.exists("Required"):
        os.chdir("Required")
    
    # Run setup
    setup = HybridSetup()
    success = setup.run()
    
    if success:
        print("\n" + "="*70)
        print("  ✅ Ready to launch server!")
        print("="*70)
        print("\nRun: python flask_hybrid_api.py")
    else:
        print("\n" + "="*70)
        print("  ⚠ Setup incomplete. See errors above.")
        print("="*70)
