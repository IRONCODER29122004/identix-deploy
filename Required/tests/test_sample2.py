"""
Test face detection on Sample2.mp4 using the video segmentation module
"""
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Report_Submission', '3_Website'))

from video_segmentation import process_video

print("=" * 70)
print("Testing Sample2.mp4 with video segmentation module")
print("=" * 70)

video_path = "Sample2.mp4"

try:
    main_person, other_people, processed, total = process_video(
        video_path,
        model_path=os.path.join('Report_Submission', '3_Website', 'best_model.pth'),
        max_frames=50
    )
    
    if main_person:
        print(f"\n✅ SUCCESS! Faces detected!")
        print(f"Main person: {main_person['screen_time']} frames")
        print(f"Other people: {len(other_people)}")
        print(f"Processed: {processed}/{total} frames")
    else:
        print(f"\n❌ FAILED: No faces detected")
        print(f"Processed: {processed}/{total} frames")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
