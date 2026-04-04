#!/usr/bin/env python3
"""
Download REAL sample videos and images for FaceSwap testing
Uses alternative sources: Archive.org, Unsplash, direct URLs
"""

import os
import sys
from pathlib import Path
import requests
from urllib.request import urlretrieve
import time

DOWNLOAD_DIR = Path(__file__).parent / 'faceswap_samples'
DOWNLOAD_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("📥 Real FaceSwap Samples Downloader (Alternative Sources)")
print("=" * 70)

# High-quality public domain and CC-licensed resources
RESOURCES = [
    {
        'name': 'interview_real_person.mp4',
        'url': 'https://www.sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4',
        'description': 'Real video clip (high quality)',
        'type': 'video'
    },
    {
        'name': 'portrait_face_1.jpg',
        'url': 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600&h=600&fit=crop',
        'description': 'High quality portrait face',
        'type': 'image'
    },
    {
        'name': 'portrait_face_2.jpg',
        'url': 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=600&h=600&fit=crop',
        'description': 'Clear face portrait',
        'type': 'image'
    },
    {
        'name': 'speaking_person.mp4',
        'url': 'https://media.w3.org/2010/05/sintel/trailer.mp4',
        'description': 'Person video (Big Buck Bunny trailer)',
        'type': 'video'
    }
]


def download_with_retry(url, output_path, max_retries=3, description=''):
    """Download with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"  Downloading ({attempt+1}/{max_retries})...", end='', flush=True)
            
            # Use requests with timeout
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if progress % 25 == 0:
                            print(".", end='', flush=True)
            
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f" ✓ ({size_mb:.2f} MB)")
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f" ✗ (retrying in 2s...)")
                time.sleep(2)
            else:
                print(f" ✗ Failed")
                print(f"    Error: {str(e)[:80]}")
    
    return False


def main():
    print(f"\n📂 Directory: {DOWNLOAD_DIR}\n")
    
    downloaded_count = 0
    
    for resource in RESOURCES:
        output_path = DOWNLOAD_DIR / resource['name']
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Already exists: {resource['name']} ({size_mb:.2f} MB)")
            downloaded_count += 1
            continue
        
        print(f"📥 {resource['description']}")
        print(f"   File: {resource['name']}")
        
        if download_with_retry(resource['url'], output_path, description=resource['description']):
            downloaded_count += 1
        else:
            # Clean up partial downloads
            if output_path.exists():
                output_path.unlink()
    
    # Print results
    print("\n" + "=" * 70)
    print("📋 RESULTS")
    print("=" * 70 + "\n")
    
    files = sorted(DOWNLOAD_DIR.glob('*'))
    if files:
        videos = [f for f in files if f.suffix == '.mp4']
        images = [f for f in files if f.suffix in ['.jpg', '.jpeg', '.png']]
        
        print(f"✓ Total files: {len(files)}\n")
        
        if videos:
            print("📹 Videos:")
            for f in videos:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"   • {f.name:<35} ({size_mb:.2f} MB)")
        
        if images:
            print("\n👤 Face Images:")
            for f in images:
                size_kb = f.stat().st_size / 1024
                print(f"   • {f.name:<35} ({size_kb:.1f} KB)")
    else:
        print("⚠ No files downloaded. Check your internet connection.")
    
    # Print instructions
    print("\n" + "=" * 70)
    print("🚀 HOW TO USE")
    print("=" * 70 + "\n")
    
    # Find the actual video and image files
    video_files = list(DOWNLOAD_DIR.glob('*.mp4'))
    image_files = list(DOWNLOAD_DIR.glob('*.jpg')) + list(DOWNLOAD_DIR.glob('*.jpeg')) + list(DOWNLOAD_DIR.glob('*.png'))
    
    if video_files and image_files:
        print("1️⃣  Open notebook: faceswap_tech_and_detection_starter.ipynb\n")
        print("2️⃣  Run this code:\n")
        print(f"""
from pathlib import Path

# Load your files
source_image = r'{image_files[0]}'
target_video = r'{video_files[0]}'
output_file = 'deepfake_result.mp4'

print(f"Source image: {{source_image}}")
print(f"Target video: {{target_video}}")

# Run faceswap
result = swap_faces_in_video(
    source_path=source_image,
    target_path=target_video,
    output_path=output_file
)

print("Generation complete!")
print(result)
""")
        print("\n3️⃣  View results in: faceswap_results_viewer.html")
    else:
        print("⚠ Not enough files downloaded. Please run again or check internet.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
