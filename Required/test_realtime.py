"""
Quick test script for real-time deepfake detection
Tests with webcam or video file
"""

import sys
import os

# Add path
sys.path.append(os.path.dirname(__file__))

from realtime_detector import RealtimeDeepfakeDetector


def test_webcam():
    """Test with webcam (simple mode)"""
    print("🎥 Testing with webcam...")
    print("Press 'Q' to quit")
    print("Press 'R' to reset history\n")
    
    config = {
        'detection_threshold': 0.5,
        'temporal_window': 30,
        'frame_skip': 3,  # Faster for testing (process every 3rd frame)
        'alert_cooldown': 2.0,
        'confidence_smoothing': 0.3,
        'enable_audio_alerts': True
    }
    
    detector = RealtimeDeepfakeDetector(config)
    
    try:
        # 0 = default webcam
        detector.process_video_stream(video_source=0, display=True)
    except KeyboardInterrupt:
        print("\n⏹ Stopped by user")
    finally:
        detector.export_session_report('webcam_test_report.json')
        print("✓ Test complete!")


def test_video_file(video_path):
    """Test with video file"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"🎥 Testing with video file: {video_path}")
    print("Press 'Q' to quit\n")
    
    config = {
        'detection_threshold': 0.5,
        'temporal_window': 30,
        'frame_skip': 2,  # Process every 2nd frame for videos
        'alert_cooldown': 1.5,
        'confidence_smoothing': 0.4,
        'enable_audio_alerts': False  # Disable audio for batch testing
    }
    
    detector = RealtimeDeepfakeDetector(config)
    
    try:
        detector.process_video_stream(video_source=video_path, display=True)
    except KeyboardInterrupt:
        print("\n⏹ Stopped by user")
    finally:
        report_name = f"video_test_{os.path.basename(video_path)}.json"
        detector.export_session_report(report_name)
        print(f"✓ Test complete! Report: {report_name}")


def quick_demo():
    """Quick demo - fastest settings for testing"""
    print("⚡ Quick Demo Mode (fastest settings)")
    print("Press 'Q' to quit\n")
    
    config = {
        'detection_threshold': 0.5,
        'temporal_window': 15,  # Shorter window
        'frame_skip': 5,  # Skip more frames
        'alert_cooldown': 3.0,
        'confidence_smoothing': 0.2,  # Less smoothing = faster response
        'enable_audio_alerts': True
    }
    
    detector = RealtimeDeepfakeDetector(config)
    
    try:
        detector.process_video_stream(video_source=0, display=True)
    except KeyboardInterrupt:
        print("\n⏹ Stopped")
    finally:
        detector.export_session_report('quick_demo_report.json')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test real-time deepfake detection')
    parser.add_argument('--mode', type=str, default='webcam',
                       choices=['webcam', 'video', 'demo'],
                       help='Test mode: webcam, video, or demo')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (for video mode)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("IDENTIX LiveGuard - Real-Time Deepfake Detection Test")
    print("="*70 + "\n")
    
    try:
        if args.mode == 'webcam':
            test_webcam()
        elif args.mode == 'video':
            if args.video:
                test_video_file(args.video)
            else:
                print("❌ Error: --video argument required for video mode")
                print("Example: python test_realtime.py --mode video --video path/to/video.mp4")
        elif args.mode == 'demo':
            quick_demo()
    
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
