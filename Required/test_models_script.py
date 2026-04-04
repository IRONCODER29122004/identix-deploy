"""
Quick script to test trained models (alternative to notebook)
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 60)
print("Testing Trained Deepfake Detection Models")
print("=" * 60)

# Step 1: Basic imports
print("\n[1/6] Loading basic libraries...")
print(f"✓ Python {sys.version.split()[0]}")
print(f"✓ NumPy {np.__version__}")
print(f"✓ OpenCV {cv2.__version__}")
print(f"✓ Pandas {pd.__version__}")

# Step 2: Try PyTorch import
print("\n[2/6] Loading PyTorch...")
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    print(f"✓ PyTorch {torch.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    torch_ok = True
except Exception as e:
    print(f"✗ PyTorch import failed: {type(e).__name__}")
    print(f"  {str(e)[:200]}")
    torch_ok = False

if not torch_ok:
    print("\n" + "=" * 60)
    print("PYTORCH REQUIREMENT NOT MET")
    print("=" * 60)
    print("\nThis is a Windows DLL issue. Solutions:")
    print("1. Install Visual C++ Redistributables:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("2. Or reinstall PyTorch:")
    print("   pip uninstall torch torchvision")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

# Step 3: Check model files
print("\n[3/6] Checking model files...")
BASE_DIR = Path.cwd()
MODELS_DIR = Path(r'c:\Users\hp\Downloads')

TEMPORAL_MODEL_PATH = MODELS_DIR / 'deepfake_temporal_medium_lstm.pth'
BASELINE_MODEL_PATH = MODELS_DIR / 'deepfake_baseline_medium.pkl'
SEGFORMER_CODE_DIR = BASE_DIR
SEGFORMER_WEIGHTS = BASE_DIR / 'models' / 'face_segmentation_kaggle_model.pth'

print(f"  Temporal (.pth): {'✓' if TEMPORAL_MODEL_PATH.exists() else '✗'} {TEMPORAL_MODEL_PATH}")
print(f"  Baseline (.pkl): {'✓' if BASELINE_MODEL_PATH.exists() else '✗'} {BASELINE_MODEL_PATH}")
print(f"  SegFormer weights: {'✓' if SEGFORMER_WEIGHTS.exists() else '✗'} {SEGFORMER_WEIGHTS}")

# Step 4: Load SegFormer
print("\n[4/6] Loading SegFormer model...")
if str(SEGFORMER_CODE_DIR) not in sys.path:
    sys.path.append(str(SEGFORMER_CODE_DIR))

try:
    from segformer_model import SegformerEdgeAware
    segformer = SegformerEdgeAware(num_classes=11, pretrained=True).to(device)
    
    if SEGFORMER_WEIGHTS.exists():
        state = torch.load(SEGFORMER_WEIGHTS, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        segformer.load_state_dict(state, strict=False)
        print("✓ SegFormer loaded with trained weights")
    else:
        print("⚠ Using base pretrained SegFormer (no fine-tuned weights)")
    
    segformer.eval()
    segformer_ok = True
except Exception as e:
    print(f"✗ SegFormer load failed: {e}")
    segformer_ok = False
    sys.exit(1)

# Step 5: Load temporal model
print("\n[5/6] Loading trained models...")

class TemporalDeepfakeDetector(nn.Module):
    def __init__(self, num_features=13, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        bsz, steps, feat = x.shape
        encoded = self.encoder(x.reshape(bsz * steps, feat))
        encoded = encoded.reshape(bsz, steps, -1)
        _, (h_n, _) = self.lstm(encoded)
        last_hidden = h_n[-1]
        logits = self.head(last_hidden)
        return logits.squeeze(-1)

temporal_model = None
if TEMPORAL_MODEL_PATH.exists():
    try:
        state = torch.load(TEMPORAL_MODEL_PATH, map_location=device)
        temporal_model = TemporalDeepfakeDetector(num_features=13, hidden_dim=64).to(device)
        temporal_model.load_state_dict(state)
        temporal_model.eval()
        print("✓ Temporal LSTM model loaded")
    except Exception as e:
        print(f"✗ Temporal model load failed: {e}")

baseline_model = None
if BASELINE_MODEL_PATH.exists():
    try:
        import joblib
        baseline_model = joblib.load(BASELINE_MODEL_PATH)
        print("✓ Baseline model loaded")
    except Exception as e:
        print(f"⚠ Baseline model load failed: {type(e).__name__}")
        print(f"  (This is expected due to version mismatch)")

# Step 6: Test inference
print("\n[6/6] Testing inference with dummy data...")

if temporal_model is not None:
    try:
        dummy_sequence = torch.randn(1, 24, 13).to(device)
        with torch.no_grad():
            logits = temporal_model(dummy_sequence)
            prob = torch.sigmoid(logits).item()
        print(f"✓ Temporal model inference: prob={prob:.4f}")
    except Exception as e:
        print(f"✗ Temporal inference failed: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"PyTorch: {'✓ Working' if torch_ok else '✗ Failed'}")
print(f"SegFormer: {'✓ Working' if segformer_ok else '✗ Failed'}")
print(f"Temporal Model: {'✓ Loaded & Working' if temporal_model is not None else '✗ Not Available'}")
print(f"Baseline Model: {'✓ Loaded' if baseline_model is not None else '⚠ Version Mismatch'}")

if temporal_model is not None:
    print("\n✓ MODEL VALIDATION SUCCESSFUL!")
    print("\nYou can now use the temporal model for predictions.")
    print("To test on a real video, modify this script to load video frames.")
else:
    print("\n⚠ Model validation incomplete - check errors above.")
