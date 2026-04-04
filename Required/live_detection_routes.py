"""
Live Deepfake Detection API Routes
Real-time webcam analysis endpoints for website integration
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import time
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from realtime_detector import RealtimeDeepfakeDetector
except ImportError:
    print("Warning: RealtimeDeepfakeDetector not available")
    RealtimeDeepfakeDetector = None


class LiveDetectionAPI:
    """API handler for live deepfake detection"""
    
    def __init__(self):
        self.detector = None
        self.initialize_detector()
    
    def initialize_detector(self):
        """Initialize the real-time detector"""
        if RealtimeDeepfakeDetector is None:
            print("Warning: Running in mock mode - detector unavailable")
            return
        
        try:
            config = {
                'detection_threshold': 0.5,
                'temporal_window': 30,
                'frame_skip': 5,
                'alert_cooldown': 2.0,
                'confidence_smoothing': 0.3,
                'enable_audio_alerts': False
            }
            self.detector = RealtimeDeepfakeDetector(config)
            print("✓ Live detection API initialized")
        except Exception as e:
            print(f"Error initializing detector: {e}")
            self.detector = None
    
    def decode_frame(self, base64_string):
        """Decode base64 image to numpy array"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            frame = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
        except Exception as e:
            print(f"Error decoding frame: {e}")
            return None
    
    def detect_face_simple(self, frame):
        """Simple face detection using Haar Cascade"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            return len(faces) > 0, faces
        except:
            return False, []
    
    def analyze_frame_mock(self, frame, sensitivity=50):
        """Mock analysis when detector is unavailable"""
        face_detected, faces = self.detect_face_simple(frame)
        
        if not face_detected:
            return {
                'face_detected': False,
                'is_authentic': True,
                'confidence': 0.0,
                'message': 'No face detected in frame'
            }
        
        # Mock detection based on simple heuristics
        # In real implementation, this would use the full detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Simple heuristic: very low blur might indicate artifact
        confidence = min(0.95, max(0.6, blur_score / 1000))
        is_authentic = confidence > (sensitivity / 100)
        
        return {
            'face_detected': True,
            'is_authentic': is_authentic,
            'confidence': float(confidence),
            'message': 'Analysis complete (mock mode)'
        }
    
    def analyze_frame(self, frame, sensitivity=50):
        """Analyze a single frame for deepfake detection"""
        if self.detector is None:
            return self.analyze_frame_mock(frame, sensitivity)
        
        try:
            # Use real detector
            result = self.detector.detect_single_frame(frame)
            
            if result is None:
                return {
                    'face_detected': False,
                    'is_authentic': True,
                    'confidence': 0.0,
                    'message': 'No face detected'
                }
            
            # Adjust threshold based on sensitivity (0-100)
            adjusted_threshold = 0.3 + (sensitivity / 100) * 0.4  # Range: 0.3-0.7
            is_authentic = result['confidence'] > adjusted_threshold
            
            return {
                'face_detected': True,
                'is_authentic': is_authentic,
                'confidence': float(result['confidence']),
                'message': 'Frame analyzed successfully',
                'details': result.get('details', {})
            }
        
        except Exception as e:
            print(f"Error in analysis: {e}")
            return {
                'face_detected': False,
                'is_authentic': True,
                'confidence': 0.0,
                'message': f'Analysis error: {str(e)}'
            }


# Global API instance
live_detection_api = LiveDetectionAPI()


def register_routes(app):
    """Register live detection API routes with Flask app"""
    
    @app.route('/api/live-detection/analyze', methods=['POST'])
    def analyze_live_frame():
        """
        Analyze a single frame from webcam
        
        Request JSON:
        {
            "frame": "base64_encoded_image",
            "sensitivity": 50  // 0-100
        }
        
        Response JSON:
        {
            "face_detected": true,
            "is_authentic": true,
            "confidence": 0.85,
            "message": "Analysis complete",
            "processing_time": 120.5
        }
        """
        start_time = time.time()
        
        try:
            data = request.get_json()
            
            if not data or 'frame' not in data:
                return jsonify({
                    'error': 'No frame data provided',
                    'face_detected': False,
                    'is_authentic': True,
                    'confidence': 0.0
                }), 400
            
            # Decode frame
            frame = live_detection_api.decode_frame(data['frame'])
            if frame is None:
                return jsonify({
                    'error': 'Invalid frame data',
                    'face_detected': False,
                    'is_authentic': True,
                    'confidence': 0.0
                }), 400
            
            # Get sensitivity setting
            sensitivity = data.get('sensitivity', 50)
            sensitivity = max(0, min(100, sensitivity))  # Clamp to 0-100
            
            # Analyze frame
            result = live_detection_api.analyze_frame(frame, sensitivity)
            
            # Add processing time
            processing_time = (time.time() - start_time) * 1000  # milliseconds
            result['processing_time'] = round(processing_time, 2)
            
            return jsonify(result)
        
        except Exception as e:
            print(f"Error in /api/live-detection/analyze: {e}")
            return jsonify({
                'error': str(e),
                'face_detected': False,
                'is_authentic': True,
                'confidence': 0.0
            }), 500
    
    @app.route('/api/live-detection/status', methods=['GET'])
    def detection_status():
        """Get detector status"""
        return jsonify({
            'initialized': live_detection_api.detector is not None,
            'mode': 'real' if live_detection_api.detector else 'mock',
            'message': 'Detector ready' if live_detection_api.detector else 'Running in mock mode'
        })
    
    print("✓ Live detection API routes registered")


# For standalone testing
if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    register_routes(app)
    
    print("\n" + "="*60)
    print("LIVE DEEPFAKE DETECTION API - Test Server")
    print("="*60)
    print("Available endpoints:")
    print("  POST /api/live-detection/analyze  - Analyze single frame")
    print("  GET  /api/live-detection/status   - Detector status")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5001)
