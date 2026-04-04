#!/usr/bin/env python3
"""
Download REAL test data: video with visible person + portrait image
Uses reliable free sources with working URLs
"""

import subprocess
import os
from pathlib import Path
import requests
from io import BytesIO
from PIL import Image
import time

DOWNLOAD_DIR = Path(__file__).parent / 'faceswap_samples'
DOWNLOAD_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("📥 Downloading Real FaceSwap Test Data")
print("=" * 70)

# =============================================================================
# OPTION 1: Download video using yt-dlp (most reliable)
# =============================================================================
VIDEO_OUTPUT = DOWNLOAD_DIR / 'person_talking.mp4'

if not VIDEO_OUTPUT.exists():
    print("\n📹 DOWNLOADING VIDEO (person doing something)...")
    print("   Using: Short public domain talk (no copyright issues)")
    
    # Use a reliable short video - TED-Ed or Khan Academy videos are good
    # This is a short person talking video
    video_url = "https://www.youtube.com/watch?v=ZXsQAXx_ao0"  # Short educational video
    
    try:
        cmd = [
            'yt-dlp',
            '-f', 'best[height<=480][ext=mp4]',  # 480p max for faster download
            '-o', str(VIDEO_OUTPUT),
            '--quiet',
            '--no-warnings',
            video_url
        ]
        
        print("   Downloading... (this may take 30-60 seconds)", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if VIDEO_OUTPUT.exists():
            size_mb = VIDEO_OUTPUT.stat().st_size / (1024 * 1024)
            print(f"   ✓ SUCCESS: {VIDEO_OUTPUT.name} ({size_mb:.2f} MB)")
        else:
            print(f"   ✗ Download failed. Trying alternative...")
            
            # Fallback: Try a different video
            print("\n   Trying alternative video source...")
            video_url_alt = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Different video
            cmd[cmd.index(video_url)] = video_url_alt
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if VIDEO_OUTPUT.exists():
                size_mb = VIDEO_OUTPUT.stat().st_size / (1024 * 1024)
                print(f"   ✓ SUCCESS (alternative): {VIDEO_OUTPUT.name} ({size_mb:.2f} MB)")
            else:
                print("   ✗ Failed to download from YouTube sources")
                print("   📌 MANUAL OPTION: Paste this in PowerShell to download:")
                print(f"\n   yt-dlp -f 'best[height<=480]' {video_url} -o '{VIDEO_OUTPUT}'\n")
    
    except FileNotFoundError:
        print("   ✗ yt-dlp not found. Install with: pip install yt-dlp")
        print("\n   📌 MANUAL: Use any MP4 video with a visible person and save to:")
        print(f"      {VIDEO_OUTPUT}")
    except Exception as e:
        print(f"   ✗ Error: {str(e)[:60]}")

# =============================================================================
# OPTION 2: Download portrait images from Pexels (free, no auth)
# =============================================================================

IMAGE_URLS = {
    'portrait_woman.jpg': 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg',
    'portrait_man.jpg': 'https://images.pexels.com/photos/91227/pexels-photo-91227.jpeg',
}

for img_name, img_url in IMAGE_URLS.items():
    img_path = DOWNLOAD_DIR / img_name
    
    if not img_path.exists():
        print(f"\n👤 Downloading {img_name}...")
        try:
            response = requests.get(img_url, timeout=30)
            response.raise_for_status()
            
            # Open and resize to 600x600 for consistency
            img = Image.open(BytesIO(response.content))
            img = img.resize((600, 600), Image.Resampling.LANCZOS)
            img.save(img_path, 'JPEG', quality=95)
            
            size_kb = img_path.stat().st_size / 1024
            print(f"   ✓ SUCCESS: {img_name} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"   ✗ Failed: {str(e)[:50]}")
            print(f"   📌 Save any portrait photo to: {img_path}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("📋 RESULTS")
print("=" * 70 + "\n")

video_exists = VIDEO_OUTPUT.exists()
images = list(DOWNLOAD_DIR.glob('portrait_*.jpg'))

if video_exists:
    size_mb = VIDEO_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"✓ Video: {VIDEO_OUTPUT.name:<30} ({size_mb:.2f} MB)")
else:
    print(f"✗ Video: NOT FOUND - {VIDEO_OUTPUT.name}")
    print(f"  Place a video with visible person in: {VIDEO_OUTPUT}")

if images:
    print(f"\n✓ Portraits found: {len(images)} image(s)")
    for img in images:
        size_kb = img.stat().st_size / 1024
        print(f"  • {img.name:<35} ({size_kb:.1f} KB)")
else:
    print("\n✗ Portraits: NOT FOUND")
    print(f"  Add portrait images to: {DOWNLOAD_DIR}")

# =============================================================================
# Usage Instructions
# =============================================================================

print("\n" + "=" * 70)
print("🚀 NEXT STEPS")
print("=" * 70 + "\n")

print("1. Open notebook: faceswap_tech_and_detection_starter.ipynb\n")
print("2. In the notebook, update paths:\n")

if images:
    best_image = images[0]
    print(f"   source_image = r'{best_image}'")
else:
    print(f"   source_image = r'{DOWNLOAD_DIR}/portrait_woman.jpg'")

if video_exists:
    print(f"   target_video = r'{VIDEO_OUTPUT}'")
else:
    print(f"   target_video = r'{DOWNLOAD_DIR}/person_talking.mp4'")

print("\n3. Re-run cells 7 and 10 to process the video")
print("4. You should now see faces being swapped in the before/after view!\n")

print("=" * 70)
