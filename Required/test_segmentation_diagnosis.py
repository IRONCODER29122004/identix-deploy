#!/usr/bin/env python3
"""
Diagnostic script to test segmentation pipeline.
Tests face detection, preprocessing, and landmark detection on a real photo.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import traceback

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import models
from landmark_app import (
    model, device, face_cascade, preprocess_image_advanced,
    _predict_mask_for_crop_stable, _env_float
)

def test_segmentation_pipeline(image_path):
    """Test the entire segmentation pipeline on a single image."""
    print(f"\n{'='*70}")
    print(f"TESTING SEGMENTATION PIPELINE")
    print(f"{'='*70}")
    print(f"Image: {image_path}")
    
    # 1. Load image
    print("\n[1] Loading image...")
    try:
        if not os.path.exists(image_path):
            print(f"ERROR: Image not found: {image_path}")
            return
        
        img_pil = Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil)
        print(f"  Loaded: {img_pil.size}, dtype={img_np.dtype}, shape={img_np.shape}")
    except Exception as e:
        print(f"  ERROR loading image: {e}")
        traceback.print_exc()
        return
    
    # 2. Detect faces
    print("\n[2] Face detection (Haar cascade)...")
    try:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            img_gray, 
            scaleFactor=1.05, 
            minNeighbors=5, 
            minSize=(60, 60),
            maxSize=(img_gray.shape[0], img_gray.shape[1])
        )
        print(f"  Detected {len(faces)} face(s)")
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                print(f"    Face {i}: ({x}, {y}) {w}x{h}, area={w*h}")
        else:
            print("  ERROR: No faces detected!")
            return
    except Exception as e:
        print(f"  ERROR in face detection: {e}")
        traceback.print_exc()
        return
    
    # Pick largest face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    print(f"\n  Selected largest face: ({x}, {y}) {w}x{h}")
    
    # 3. Extract face crop with padding
    print("\n[3] Extract face crop...")
    try:
        face_padding = _env_float('BISENET_FACE_PADDING', 0.2)
        pad = int(max(w, h) * face_padding)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_np.shape[1], x + w + pad)
        y2 = min(img_np.shape[0], y + h + pad)
        
        face_crop_np = img_np[y1:y2, x1:x2]
        face_crop_pil = Image.fromarray(face_crop_np)
        print(f"  Crop: ({x1}, {y1}) - ({x2}, {y2}) = {face_crop_pil.size}")
        print(f"  Crop dtype: {face_crop_np.dtype}, shape: {face_crop_np.shape}")
    except Exception as e:
        print(f"  ERROR extracting crop: {e}")
        traceback.print_exc()
        return
    
    # 4. Preprocess
    print("\n[4] Preprocessing...")
    try:
        preprocessed = preprocess_image_advanced(face_crop_pil)
        preprocessed_np = np.array(preprocessed)
        print(f"  After preprocess: {preprocessed.size}, dtype={preprocessed_np.dtype}")
    except Exception as e:
        print(f"  ERROR in preprocessing: {e}")
        traceback.print_exc()
        preprocessed = face_crop_pil
        preprocessed_np = face_crop_np
    
    # 5. Resize to 256x256 for model
    print("\n[5] Resize to model input (256x256)...")
    try:
        resized = preprocessed.resize((256, 256), Image.BILINEAR)
        resized_np = np.array(resized)
        print(f"  Resized: {resized.size}, dtype={resized_np.dtype}")
        print(f"  Min={resized_np.min()}, Max={resized_np.max()}, Mean={resized_np.mean():.1f}")
    except Exception as e:
        print(f"  ERROR resizing: {e}")
        traceback.print_exc()
        return
    
    # 6. Run segmentation
    print("\n[6] Running segmentation model...")
    try:
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            mask, size_label = _predict_mask_for_crop_stable(resized)
            sys.stdout = old_stdout
        
        print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"  Mask range: {mask.min()} - {mask.max()}")
        print(f"  Unique labels: {np.unique(mask)}")
        print(f"  Non-zero pixels: {np.count_nonzero(mask)} / {mask.size}")
        
        # Calculate non-background percentage
        bg_count = np.sum(mask == 0)
        non_bg = 100.0 * (1.0 - bg_count / mask.size)
        print(f"  Non-background: {non_bg:.1f}%")
        
        if non_bg < 1.0:
            print("  CRITICAL: Mask is almost all background!")
    except Exception as e:
        print(f"  ERROR in segmentation: {e}")
        traceback.print_exc()
        return
    
    # 7. Map back to original crop size
    print("\n[7] Map mask back to crop size...")
    try:
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (face_crop_np.shape[1], face_crop_np.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        print(f"  Resized mask: {mask_resized.shape}")
        non_bg_crop = 100.0 * (1.0 - np.sum(mask_resized == 0) / mask_resized.size)
        print(f"  Non-background in crop: {non_bg_crop:.1f}%")
    except Exception as e:
        print(f"  ERROR mapping mask: {e}")
        traceback.print_exc()
        return
    
    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*70}\n")
    
    return {
        'image_path': image_path,
        'face_detected': len(faces) > 0,
        'face_count': len(faces),
        'face_bbox': tuple(largest_face),
        'mask_non_bg_percent': non_bg,
        'crop_non_bg_percent': non_bg_crop,
    }

if __name__ == '__main__':
    # Test on the uploaded image (if available)
    test_image = 'data/uploads/test_selfie.jpg'
    
    # Try common locations
    candidates = [
        'data/uploads/test_selfie.jpg',
        'Required/data/uploads/test_selfie.jpg',
        'temp/uploads/test_selfie.jpg',
    ]
    
    for cand in candidates:
        if os.path.exists(cand):
            test_image = cand
            break
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    result = test_segmentation_pipeline(test_image)
    
    if result:
        print("\nRESULT SUMMARY:")
        for k, v in result.items():
            print(f"  {k}: {v}")
