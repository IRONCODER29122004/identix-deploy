#!/usr/bin/env python3
"""
Standalone FaceSwap Generator
Uses MediaPipe + OpenCV for real-time face swapping
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import time

# Paths
BASE_DIR = Path(r'D:\link2\Capstone 4-1\Code_try_1\Required')
SOURCE_IMAGE = BASE_DIR / 'faceswap_samples' / 'portrait_man.jpg'
TARGET_VIDEO = BASE_DIR / 'test_videos' / 'test_person_video.mp4'
OUTPUT_VIDEO = BASE_DIR / 'output_deepfake.mp4'

print("=" * 70)
print("FaceSwap Video Generation")
print("=" * 70)
print(f"\nSource: {SOURCE_IMAGE.name}")
print(f"Target: {TARGET_VIDEO.name}")
print(f"Output: {OUTPUT_VIDEO.name}\n")

# Load source image
source_img = cv2.imread(str(SOURCE_IMAGE))
if source_img is None:
    print(f"ERROR: Could not load {SOURCE_IMAGE}")
    exit(1)

print(f"Source image loaded: {source_img.shape}")

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("MediaPipe FaceMesh initialized")

# Helper function for face detection
def get_face_landmarks(image_bgr, model_instance):
    """Extract 468 face landmark points"""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = model_instance.process(rgb)
    
    if not result.multi_face_landmarks:
        return None
    
    lm = result.multi_face_landmarks[0].landmark
    points = np.array([(int(p.x * w), int(p.y * h)) for p in lm], dtype=np.int32)
    return points

# Extract source landmarks
src_landmarks = get_face_landmarks(source_img, mp_face_mesh)
if src_landmarks is None:
    print("ERROR: No face detected in source image")
    exit(1)

print(f"Source landmarks: {src_landmarks.shape}")

# Open target video
cap = cv2.VideoCapture(str(TARGET_VIDEO))
if not cap.isOpened():
    print(f"ERROR: Could not open {TARGET_VIDEO}")
    exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Target video: {frame_count} frames @ {fps} FPS, {w}x{h}")

# Create output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (w, h))

# Process frames
print(f"\nProcessing {frame_count} frames...")
t0 = time.time()
frames_swapped = 0
frames_failed = 0

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get target landmarks
    tgt_landmarks = get_face_landmarks(frame, mp_face_mesh)
    
    if tgt_landmarks is None:
        writer.write(frame)
        frames_failed += 1
    else:
        try:
            # Compute convex hull for both faces
            hull_idx = cv2.convexHull(tgt_landmarks, returnPoints=False)
            src_hull = src_landmarks[hull_idx[:, 0]]
            tgt_hull = tgt_landmarks[hull_idx[:, 0]]
            
            # Estimate affine transformation
            M, _ = cv2.estimateAffinePartial2D(
                src_hull.astype(np.float32),
                tgt_hull.astype(np.float32)
            )
            
            if M is not None:
                # Warp source face to target position
                warped = cv2.warpAffine(source_img, M, (w, h))
                
                # Create mask
                mask = np.zeros((h, w), dtype=np.uint8)
                tgt_hull_2d = tgt_hull.reshape((-1, 1, 2))
                cv2.drawContours(mask, [tgt_hull_2d], 0, 255, -1)
                
                # Find center for seamless clone
                x, y, w_m, h_m = cv2.boundingRect(tgt_hull_2d)
                center = (x + w_m // 2, y + h_m // 2)
                
                # Seamless blend
                result_frame = cv2.seamlessClone(warped, frame, mask, center, cv2.NORMAL_CLONE)
                writer.write(result_frame)
                frames_swapped += 1
            else:
                writer.write(frame)
                frames_failed += 1
                
        except Exception as e:
            if frames_failed < 3:
                print(f"  Frame {frame_idx}: {str(e)[:60]}")
            writer.write(frame)
            frames_failed += 1
    
    if (frame_idx + 1) % 30 == 0:
        print(f"  Processed {frame_idx + 1}/{frame_count} frames...")

cap.release()
writer.release()

elapsed = time.time() - t0
total = frames_swapped + frames_failed
success_rate = (frames_swapped / total * 100) if total > 0 else 0

print("\n" + "=" * 70)    
print("RESULTS")
print("=" * 70)
print(f"Frames processed: {frames_swapped}/{total} ({success_rate:.1f}%)")
print(f"Frames failed: {frames_failed}")
print(f"Time: {elapsed:.2f}s ({1000*elapsed/total:.1f}ms per frame)")
print(f"\nOutput: {OUTPUT_VIDEO}")
size_mb = OUTPUT_VIDEO.stat().st_size / (1024**2) if OUTPUT_VIDEO.exists() else 0
print(f"File size: {size_mb:.2f} MB")
print("\nSUCCESS! Deepfake video generated!")
print("=" * 70)
