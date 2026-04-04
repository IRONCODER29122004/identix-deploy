"""
Download sample test videos for deepfake detection testing
Quick script to get real and fake video samples
"""
import urllib.request
from pathlib import Path

# Create test videos directory
test_dir = Path("test_videos")
test_dir.mkdir(exist_ok=True)

print("Downloading sample test videos...")
print("=" * 60)

# Sample videos from public datasets (replace with actual URLs)
samples = {
    "real_sample.mp4": "https://example.com/real_video.mp4",  # Replace with actual URL
    "fake_sample.mp4": "https://example.com/fake_video.mp4",  # Replace with actual URL
}

print("\n⚠️  NOTE: This script needs actual video URLs.")
print("Please use one of these methods instead:\n")
print("1. Download from Kaggle datasets:")
print("   - Celeb-DF: https://www.kaggle.com/datasets/dagnelies/celebdf")
print("   - DFDC: https://www.kaggle.com/c/deepfake-detection-challenge/data")
print("\n2. Use yt-dlp to download YouTube videos:")
print("   pip install yt-dlp")
print("   yt-dlp -f 'best[height<=480]' <youtube_url> -o test_videos/real.mp4")
print("\n3. Record a webcam video:")
print("   - Use Windows Camera app")
print("   - Save as MP4 in test_videos folder")
print("\n4. Use any existing MP4 file you have")

print("\n" + "=" * 60)
print("Once you have videos, test with:")
print("  TEST_VIDEO_PATH = Path(r'test_videos/your_video.mp4')")
