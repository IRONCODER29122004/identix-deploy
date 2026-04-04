#!/usr/bin/env python3
"""
Download video directly without yt-dlp (using direct URLs)
"""

import requests
from pathlib import Path

DOWNLOAD_DIR = Path(__file__).parent / 'faceswap_samples'
DOWNLOAD_DIR.mkdir(exist_ok=True)

VIDEO_OUTPUT = DOWNLOAD_DIR / 'person_talking.mp4'

print("📹 Downloading video with visible person...")
print("   Using direct download link...\n")

# Direct video URLs that should work
video_urls = [
    {
        'name': 'Big Buck Bunny (has character)',
        'url': 'https://download.blender.org/demo/movies/BBB_short_420p_H.264.mp4',
    },
    {
        'name': 'Sample video from archive',
        'url': 'https://www.w3schools.com/html/mov_bbb.mp4',
    },
]

for source in video_urls:
    print(f"Trying: {source['name']}...", flush=True)
    try:
        response = requests.get(source['url'], timeout=60, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(VIDEO_OUTPUT, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress_pct = (downloaded / total_size * 100) if total_size > 0 else 0
                    print(f"  Downloaded: {progress_pct:.1f}%", end='\r', flush=True)
        
        size_mb = VIDEO_OUTPUT.stat().st_size / (1024 * 1024)
        if size_mb > 1:  # Only accept if file is meaningful size
            print(f"\n✓ SUCCESS! Downloaded {size_mb:.2f} MB\n")
            break
        else:
            VIDEO_OUTPUT.unlink()
            print("File too small, trying next source...\n")
    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:60]}\n")
        if VIDEO_OUTPUT.exists():
            VIDEO_OUTPUT.unlink()

# Check if we got the video
if not VIDEO_OUTPUT.exists():
    print("\n⚠️  Could not download video from direct sources")
    print("\nHowever, you have 4 portrait images ready:")
    portraits = list(DOWNLOAD_DIR.glob('portrait_*.jpg'))
    for p in portraits:
        print(f"  ✓ {p.name}")
    print("\n📌 OPTIONS:")
    print("1. Use any MP4 video on your computer (record yourself, download from web)")
    print("2. Save it to: D:\\link2\\Capstone 4-1\\Code_try_1\\Required\\faceswap_samples\\person_talking.mp4")
    print("3. Or use video_20241219_102330.mp4 if it has visible people\n")
else:
    size_mb = VIDEO_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"✓ Video ready: {VIDEO_OUTPUT.name} ({size_mb:.2f} MB)")
    print(f"✓ Portraits ready: {len(list(DOWNLOAD_DIR.glob('portrait_*.jpg')))} images")
    print("\n🚀 You can now run the notebook!")
