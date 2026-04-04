#!/usr/bin/env python3
"""
Download sample videos and images for FaceSwap testing
- Video: Real person (target for faceswap)
- Image: Person face (source to swap onto video)
"""

import os
import sys
from pathlib import Path
import requests
from urllib.request import urlretrieve
import time

# Create downloads directory
DOWNLOAD_DIR = Path(__file__).parent / 'faceswap_samples'
DOWNLOAD_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("📥 FaceSwap Sample Downloader")
print("=" * 70)

# ============================================================================
# OPTION 1: Download from Pexels (Free stock videos & images)
# ============================================================================

def download_pexels_resources():
    """
    Download free resources from Pexels (no API key needed for basic use)
    """
    print("\n📹 Downloading sample resources from Pexels...")
    print("(Free stock videos and images - Creative Commons license)\n")
    
    resources = [
        {
            'name': 'person_video_1.mp4',
            'url': 'https://videos.pexels.com/video-files/3692185/3692185-sd_640_360_30fps.mp4',
            'description': 'Person speaking (640x360)'
        },
        {
            'name': 'person_face_image_1.jpg',
            'url': 'https://images.pexels.com/photos/1181690/pexels-photo-1181690.jpeg?auto=compress&cs=tinysrgb&w=600',
            'description': 'Clear face portrait (600x600)'
        },
        {
            'name': 'person_face_image_2.jpg',
            'url': 'https://images.pexels.com/photos/1239291/pexels-photo-1239291.jpeg?auto=compress&cs=tinysrgb&w=600',
            'description': 'Person portrait (600x600)'
        }
    ]
    
    for resource in resources:
        output_path = DOWNLOAD_DIR / resource['name']
        
        if output_path.exists():
            print(f"✓ Already exists: {resource['name']}")
            continue
        
        try:
            print(f"⏳ Downloading: {resource['description']}")
            print(f"   URL: {resource['url'][:60]}...")
            
            urlretrieve(resource['url'], output_path)
            
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Downloaded: {resource['name']} ({size_mb:.2f} MB)")
            
        except Exception as e:
            print(f"✗ Failed: {resource['name']}")
            print(f"  Error: {str(e)[:80]}")
    
    print("\n✓ Download complete!")
    return True


# ============================================================================
# OPTION 2: Manual file creation for testing
# ============================================================================

def create_test_video_from_existing():
    """
    Create a short test video from existing frames
    Uses OpenCV to generate a simple video
    """
    print("\n🎬 Creating synthetic test video...")
    
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("⚠ OpenCV not available. Skipping synthetic video creation.")
        return False
    
    # Create a simple test video
    video_path = DOWNLOAD_DIR / 'synthetic_test_video.mp4'
    
    if video_path.exists():
        print(f"✓ Test video already exists: {video_path}")
        return True
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        
        # Generate 150 frames (5 seconds at 30fps)
        print("  Generating frames...", end='', flush=True)
        for frame_idx in range(150):
            # Create a frame with changing colors/patterns
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add some patterns
            frame[:240, :320] = [100 + frame_idx, 150, 100]  # Top-left
            frame[:240, 320:] = [100, 100 + frame_idx, 150]  # Top-right
            frame[240:, :320] = [150, 100, 100 + frame_idx]  # Bottom-left
            frame[240:, 320:] = [100 + frame_idx, 100, 150]  # Bottom-right
            
            # Add text
            cv2.putText(frame, f'Frame {frame_idx}', (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
            
            if (frame_idx + 1) % 50 == 0:
                print(f" {frame_idx+1}/150", end='', flush=True)
        
        out.release()
        print(" ✓")
        
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"✓ Created synthetic test video: {video_path} ({size_mb:.2f} MB)")
        return True
    
    except Exception as e:
        print(f"✗ Failed to create video: {str(e)[:80]}")
        return False


def create_test_face_image():
    """
    Create a synthetic face image for testing
    """
    print("\n👤 Creating synthetic face image...")
    
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("⚠ OpenCV not available. Skipping synthetic image creation.")
        return False
    
    image_path = DOWNLOAD_DIR / 'synthetic_face.jpg'
    
    if image_path.exists():
        print(f"✓ Face image already exists: {image_path}")
        return True
    
    try:
        # Create a simple face-like image
        img = np.ones((400, 300, 3), dtype=np.uint8) * 240
        
        # Draw face oval
        cv2.ellipse(img, (150, 150), (80, 100), 0, 0, 360, (200, 160, 140), -1)
        
        # Draw eyes
        cv2.circle(img, (120, 120), 12, (50, 50, 50), -1)
        cv2.circle(img, (180, 120), 12, (50, 50, 50), -1)
        cv2.circle(img, (118, 118), 5, (255, 255, 255), -1)
        cv2.circle(img, (178, 118), 5, (255, 255, 255), -1)
        
        # Draw nose
        pts = np.array([[150, 140], [145, 160], [155, 160]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (200, 150, 120))
        
        # Draw mouth
        cv2.ellipse(img, (150, 190), (30, 15), 0, 0, 180, (150, 80, 80), 2)
        
        cv2.imwrite(str(image_path), img)
        
        size_kb = image_path.stat().st_size / 1024
        print(f"✓ Created synthetic face image: {image_path} ({size_kb:.1f} KB)")
        return True
    
    except Exception as e:
        print(f"✗ Failed to create image: {str(e)[:80]}")
        return False


# ============================================================================
# Main Download Process
# ============================================================================

def main():
    print(f"\n📂 Download directory: {DOWNLOAD_DIR}\n")
    
    # Try to download from Pexels
    try:
        download_pexels_resources()
    except Exception as e:
        print(f"\n⚠ Could not download from Pexels: {str(e)[:100]}")
        print("   Falling back to synthetic file creation...\n")
    
    # Create synthetic files as fallback
    create_test_video_from_existing()
    create_test_face_image()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 70)
    
    files = list(DOWNLOAD_DIR.glob('*'))
    if files:
        print(f"\n✓ {len(files)} file(s) ready in: {DOWNLOAD_DIR}\n")
        for f in sorted(files):
            size = f.stat().st_size
            if size > 1024*1024:
                size_str = f"{size / (1024*1024):.2f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  • {f.name:<40} ({size_str})")
    else:
        print("\n⚠ No files downloaded. Please check your internet connection.")
    
    print("\n" + "=" * 70)
    print("🚀 NEXT STEPS")
    print("=" * 70)
    print(f"""
To use these files for FaceSwap:

1. Open your Jupyter notebook: faceswap_tech_and_detection_starter.ipynb

2. Use the video as TARGET (background):
   target_video = r'{DOWNLOAD_DIR}/person_video_1.mp4'
   
3. Use the image as SOURCE (face to swap):
   source_image = r'{DOWNLOAD_DIR}/person_face_image_1.jpg'
   
4. Generate deepfake:
   result = swap_faces_in_video(
       source_path=source_image,
       target_path=target_video,
       output_path='output_deepfake.mp4'
   )

5. View results in the HTML viewer:
   faceswap_results_viewer.html

""")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Download cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
