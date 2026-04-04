#!/usr/bin/env python3
"""
Create a synthetic video from a portrait image for testing faceswap
This will create a video where the same face appears with slight scale/rotation variations
"""

import cv2
import numpy as np
from pathlib import Path

INPUT_DIR = Path(__file__).parent / 'faceswap_samples'
OUTPUT_DIR = Path(__file__).parent / 'test_videos'
OUTPUT_DIR.mkdir(exist_ok=True)

# Read a portrait image
portrait_path = INPUT_DIR / 'portrait_woman.jpg'
portrait = cv2.imread(str(portrait_path))

if portrait is None:
    print(f"❌ Could not load: {portrait_path}")
    exit(1)

print(f"✓ Loaded: {portrait_path.name}")
print(f"✓ Image size: {portrait.shape[1]}x{portrait.shape[0]}")

# Create output video
output_path = OUTPUT_DIR / 'test_person_video.mp4'
fps = 30
duration_seconds = 5
frame_count = fps * duration_seconds

# Video codec and writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(output_path), fourcc, fps, (642, 480))

print(f"\n🎬 Generating {duration_seconds}s video ({frame_count} frames)...")

# Create frames with slight variations
for frame_idx in range(frame_count):
    # Normalize frame number to 0-1
    progress = frame_idx / max(frame_count - 1, 1)
    
    # Apply slight rotation (-10 to +10 degrees)
    angle = -10 + (progress * 20)
    
    # Apply slight zoom (0.9 to 1.1)
    scale = 0.9 + (progress * 0.2)
    
    # Create a base frame with background
    frame = np.ones((480, 642, 3), dtype=np.uint8) * 230  # Light gray background
    
    # Calculate rotation and scaling
    h, w = portrait.shape[:2]
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    M_rotate = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(portrait, M_rotate, (w, h))
    
    # Center the rotated image in the frame
    y_offset = (480 - h) // 2
    x_offset = (642 - w) // 2
    
    # Ensure valid coordinates
    y1 = max(0, y_offset)
    x1 = max(0, x_offset)
    y2 = min(480, y_offset + h)
    x2 = min(642, x_offset + w)
    
    # Adjust rotated image crop if needed
    ry1 = max(0, -y_offset)
    rx1 = max(0, -x_offset)
    ry2 = ry1 + (y2 - y1)
    rx2 = rx1 + (x2 - x1)
    
    if y2 > y1 and x2 > x1:
        frame[y1:y2, x1:x2] = rotated[ry1:ry2, rx1:rx2]
    
    # Add frame counter text
    cv2.putText(frame, f'Frame {frame_idx+1}/{frame_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    writer.write(frame)
    
    if (frame_idx + 1) % 30 == 0:
        print(f"  {frame_idx + 1}/{frame_count} frames written...")

writer.release()

size_mb = output_path.stat().st_size / (1024 ** 2)
print(f"\n✅ SUCCESS! Video created: {output_path.name}")
print(f"   Size: {size_mb:.2f} MB")
print(f"   Duration: {duration_seconds}s @ {fps} FPS")
print(f"   Resolution: 642x480")
print(f"\n🎯 Now you have:")
print(f"   ✓ Source portrait: portrait_woman.jpg")
print(f"   ✓ Target video: test_person_video.mp4 (contains this same face)")
print(f"\n📝 To see faceswap in action:")
print(f"   1. Use portrait_man.jpg as SOURCE")
print(f"   2. Use test_person_video.mp4 as TARGET")
print(f"   3. Watch the woman's face get replaced with the man's face!")
