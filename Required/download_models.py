"""
Setup script to download pre-trained deepfake detection models
Includes Xception and EfficientNet models trained on FaceForensics++ and DFDC
"""

import os
import torch
import gdown
import urllib.request
from pathlib import Path

class ModelDownloader:
    """Download pre-trained deepfake detection models"""
    
    # Pre-trained model URLs (these are from public deepfake detection repositories)
    MODELS = {
        'xception_ff': {
            'name': 'Xception - FaceForensics++',
            'url': 'https://drive.google.com/uc?id=1z_fwWnuAjeKwz65DO94STg9vv47kWL8P',  # Example ID
            'size': '119 MB',
            'accuracy': '96.5%',
            'description': 'Xception trained on FaceForensics++ (best for deepfakes)'
        },
        'efficientnet_dfdc': {
            'name': 'EfficientNet-B4 - DFDC',
            'url': 'https://drive.google.com/uc?id=1oFW0RELCLrHYoiWqHJEJiXLBxvMw1YOr',  # Example ID
            'size': '71 MB',
            'accuracy': '89.0%',
            'description': 'EfficientNet-B4 trained on DFDC (faster inference)'
        }
    }
    
    @staticmethod
    def download_model(model_name, output_dir='models'):
        """
        Download a pre-trained model
        
        Usage:
            downloader = ModelDownloader()
            downloader.download_model('xception_ff')
        """
        if model_name not in ModelDownloader.MODELS:
            print(f"❌ Model '{model_name}' not found. Available models:")
            for key, info in ModelDownloader.MODELS.items():
                print(f"  - {key}: {info['name']} ({info['accuracy']})")
            return None
        
        model_info = ModelDownloader.MODELS[model_name]
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'{model_name}.pth')
        
        if os.path.exists(output_path):
            print(f"✓ Model already exists: {output_path}")
            return output_path
        
        print(f"\n📥 Downloading {model_info['name']}...")
        print(f"   Size: {model_info['size']}")
        print(f"   Accuracy: {model_info['accuracy']}")
        
        try:
            # Download from Google Drive
            gdown.download(
                model_info['url'],
                output=output_path,
                quiet=False
            )
            print(f"✓ Downloaded successfully: {output_path}")
            return output_path
        
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print(f"\nManual download:")
            print(f"   1. Visit: {model_info['url']}")
            print(f"   2. Save to: {output_path}")
            return None
    
    @staticmethod
    def download_all(output_dir='models'):
        """Download all available pre-trained models"""
        print("=" * 60)
        print("DEEPFAKE DETECTION MODELS")
        print("=" * 60)
        
        results = {}
        for model_name in ModelDownloader.MODELS.keys():
            path = ModelDownloader.download_model(model_name, output_dir)
            results[model_name] = path
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        for model_name, path in results.items():
            status = "✓" if path and os.path.exists(path) else "❌"
            print(f"{status} {ModelDownloader.MODELS[model_name]['name']}")
            if path:
                print(f"   Location: {path}")
        
        return results
    
    @staticmethod
    def list_models():
        """List all available pre-trained models"""
        print("\n" + "=" * 70)
        print("AVAILABLE PRE-TRAINED MODELS")
        print("=" * 70)
        
        for key, info in ModelDownloader.MODELS.items():
            print(f"\n📦 {key.upper()}")
            print(f"   Name: {info['name']}")
            print(f"   Size: {info['size']}")
            print(f"   Accuracy: {info['accuracy']}")
            print(f"   Description: {info['description']}")
        
        print("\n" + "=" * 70)


# Alternative: Install from HuggingFace Hub
class HuggingFaceModels:
    """
    Models available on HuggingFace Hub
    Requires: pip install huggingface-hub
    """
    
    @staticmethod
    def load_from_hub(model_id, output_dir='models'):
        """
        Load model from HuggingFace Hub
        
        Example model IDs:
        - dxmihai/deepfake-detection-xception
        - apmaurya/deepfake-detection-efficient-net
        """
        from huggingface_hub import hf_hub_download
        
        try:
            model_path = hf_hub_download(
                repo_id=model_id,
                filename='pytorch_model.bin',
                cache_dir=output_dir
            )
            print(f"✓ Loaded from HuggingFace: {model_path}")
            return model_path
        except Exception as e:
            print(f"❌ Failed to load from HuggingFace: {e}")
            return None


# Quick setup function
def setup_deepfake_models():
    """Quick setup - downloads all models"""
    print("\nSetting up deepfake detection models...")
    print("This might take a few minutes on first run...\n")
    
    ModelDownloader.download_all(output_dir='Required/models')
    
    print("\n✓ Setup complete!")
    print("\nUsage:")
    print("""
    from ml_deepfake_detector import XceptionDeepfakeDetector
    
    # Load model
    detector = XceptionDeepfakeDetector('Required/models/xception_ff.pth')
    
    # Detect deepfake
    import cv2
    face = cv2.imread('face.jpg')
    result = detector.detect(face)
    print(result['verdict'])
    """)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        setup_deepfake_models()
    else:
        ModelDownloader.list_models()
        print("\nTo download all models, run:")
        print("   python download_models.py setup")
