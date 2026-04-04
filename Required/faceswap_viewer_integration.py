"""
Integration module for FaceSwap Results Viewer
Use this in your Jupyter notebook to automatically:
1. Generate detailed statistics from your faceswap runs
2. Export results in viewer-compatible format
3. Open the HTML viewer with one command

Usage in notebook:
    from faceswap_viewer_integration import FaceSwapResultsExporter
    
    exporter = FaceSwapResultsExporter()
    
    # After running swap_faces_in_video(...)
    exporter.export_generation_results(
        result=result,
        source_file='source.jpg',
        target_file='target.mp4',
        quality_score=8.7
    )
    
    # Then view results
    exporter.open_viewer()
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


class FaceSwapResultsExporter:
    """Export FaceSwap generation results to viewer format"""
    
    def __init__(self, output_dir=None):
        """
        Initialize exporter
        
        Args:
            output_dir: Directory to save stats (default: current dir)
        """
        self.output_dir = Path(output_dir or '.')
        self.output_dir.mkdir(exist_ok=True)
        self.stats = {}
    
    def export_generation_results(self, result, source_file=None, target_file=None, 
                                 quality_score=None, notes=None):
        """
        Export generation results to viewer format
        
        Args:
            result: Dictionary from swap_faces_in_video() with keys:
                    - frames_processed
                    - frames_failed
                    - fps
                    - total_frames
                    - success_rate
            source_file: Path to source face image/video
            target_file: Path to target video
            quality_score: 0-10 quality rating
            notes: Additional comments
        
        Returns:
            Path to saved JSON stats file
        """
        if not isinstance(result, dict):
            raise ValueError("result must be a dictionary")
        
        timestamp = datetime.now().isoformat()
        
        # Build stats structure
        stats = {
            "session_id": f"faceswap_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": timestamp,
            
            "generation_config": {
                "source_type": "image" if source_file and str(source_file).endswith(('.jpg', '.png')) else "video",
                "source_file": str(source_file) if source_file else "unknown",
                "target_video": str(target_file) if target_file else "unknown",
                "model": "MediaPipe FaceSwap v1.0",
                "blending_method": "cv2.seamlessClone (NORMAL_CLONE)",
                "landmarks": "468-point Face Mesh"
            },
            
            "processing_metrics": {
                "frames_processed": result.get('frames_processed', 0),
                "frames_failed": result.get('frames_failed', 0),
                "total_frames": result.get('total_frames', 0),
                "success_rate": result.get('success_rate', 0),
                "fps": result.get('fps', 30),
                "processing_time_seconds": result.get('processing_time_seconds', None),
            },
            
            "output_metrics": {
                "output_file": result.get('output_path', 'output_deepfake.mp4'),
                "output_duration": f"{result.get('total_frames', 0) / result.get('fps', 30):.1f}s" 
                                  if result.get('fps') else "unknown",
                "output_codec": "H.264",
                "output_fps": result.get('fps', 30),
            }
        }
        
        # Add quality assessment if provided
        if quality_score is not None:
            stats["quality_assessment"] = {
                "overall_quality_score": quality_score,
                "notes": notes or "Quality score assigned by user"
            }
        
        # Save to JSON
        stats_file = self.output_dir / f"faceswap_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Stats exported to: {stats_file}")
        print(f"\nGeneration Summary:")
        print(f"  • Frames processed: {stats['processing_metrics']['frames_processed']}")
        print(f"  • Success rate: {stats['processing_metrics']['success_rate']:.2f}%")
        print(f"  • Output: {stats['output_metrics']['output_file']}")
        
        self.stats = stats
        return stats_file
    
    def print_stats_for_viewer(self):
        """Print stats in format ready to paste into viewer"""
        if not self.stats:
            print("No stats generated yet. Run export_generation_results() first.")
            return
        
        print("\n" + "=" * 70)
        print("COPY THIS JSON TO STATISTICS TAB IN VIEWER")
        print("=" * 70)
        print(json.dumps(self.stats, indent=2))
        print("=" * 70 + "\n")
    
    @staticmethod
    def open_viewer(port=8000):
        """
        Open the faceswap viewer in browser
        
        Args:
            port: Port to run server on
        """
        viewer_file = Path(__file__).parent / 'faceswap_results_viewer.html'
        server_script = Path(__file__).parent / 'serve_faceswap_viewer.py'
        
        if not viewer_file.exists():
            print(f"❌ Viewer file not found: {viewer_file}")
            return
        
        print(f"\n🎬 Opening FaceSwap Results Viewer...")
        print(f"   File: {viewer_file.name}")
        
        try:
            # Try to start server
            if server_script.exists():
                print(f"\n📍 Starting server on port {port}...")
                subprocess.Popen([sys.executable, str(server_script)], 
                                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0)
                time.sleep(2)
                
                import webbrowser
                webbrowser.open(f'http://localhost:{port}')
                print(f"✓ Browser opened: http://localhost:{port}")
            else:
                # Fallback: open HTML directly
                import webbrowser
                webbrowser.open(f'file:///{viewer_file}')
                print(f"✓ Opened in browser: {viewer_file}")
                
        except Exception as e:
            print(f"⚠ Could not open automatically: {e}")
            print(f"   Open manually: {viewer_file}")


def create_stats_from_result(result, source_file=None, target_file=None):
    """
    Quick helper to create stats from swap_faces_in_video() result
    
    Args:
        result: Dictionary from swap_faces_in_video()
        source_file: Source image/video path
        target_file: Target video path
    
    Returns:
        Formatted stats dictionary
    """
    return {
        "frames_processed": result.get('frames_processed', 0),
        "frames_failed": result.get('frames_failed', 0),
        "fps": result.get('fps', 30),
        "total_frames": result.get('total_frames', 0),
        "processing_time_seconds": result.get('processing_time_seconds'),
        "success_rate": result.get('success_rate', 0),
        "source_file": str(source_file) if source_file else "unknown",
        "target_file": str(target_file) if target_file else "unknown",
        "output_path": result.get('output_path', 'output.mp4'),
        "timestamp": datetime.now().isoformat()
    }


# Example usage (will work in Jupyter notebook)
if __name__ == '__main__':
    import json
    
    # Test export
    print("Testing FaceSwap Results Exporter...\n")
    
    sample_result = {
        'frames_processed': 150,
        'frames_failed': 5,
        'fps': 30,
        'total_frames': 155,
        'success_rate': 96.77,
        'output_path': 'test_deepfake.mp4'
    }
    
    exporter = FaceSwapResultsExporter()
    exporter.export_generation_results(
        result=sample_result,
        source_file='test_source.jpg',
        target_file='test_target.mp4',
        quality_score=8.5,
        notes="Test generation"
    )
    
    exporter.print_stats_for_viewer()
    
    print("✓ Export test successful!")
