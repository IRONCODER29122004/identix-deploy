"""
Real-Time Deepfake Detector
Extends the existing DeepfakeDetector with real-time video processing capabilities
Adds temporal analysis, frame buffering, and live confidence scoring
"""

import cv2
import torch
import numpy as np
from collections import deque
from threading import Thread, Lock
from queue import Queue
import time
from datetime import datetime
import json

# Import your existing components
import sys
sys.path.append('Required/deploy/identix-deploy')
from deepfake_detector import DeepfakeDetector


class RealtimeDeepfakeDetector:
    """
    Real-time deepfake detector with temporal analysis
    
    Features:
    - Live video processing with threading
    - Frame-by-frame analysis with temporal smoothing
    - Multi-factor confidence scoring
    - Alert system with configurable thresholds
    """
    
    def __init__(self, config=None):
        # Default configuration
        self.config = config or {
            'detection_threshold': 0.5,
            'temporal_window': 30,      # Analyze last 30 frames (5 sec at 6fps)
            'frame_skip': 5,             # Process every 5th frame
            'alert_cooldown': 3.0,       # Seconds between alerts
            'confidence_smoothing': 0.3, # Smoothing factor (0-1)
            'enable_audio_alerts': True
        }
        
        # Load models and detectors
        print("Initializing real-time deepfake detector...")
        self.deepfake_detector = DeepfakeDetector()
        self.face_detector = self._load_face_detector()
        
        # Frame processing queues
        self.input_queue = Queue(maxsize=30)
        self.output_queue = Queue(maxsize=30)
        
        # Temporal tracking
        self.landmark_history = deque(maxlen=self.config['temporal_window'])
        self.confidence_history = deque(maxlen=self.config['temporal_window'])
        self.frame_history = deque(maxlen=self.config['temporal_window'])
        
        # State management
        self.is_running = False
        self.lock = Lock()
        self.current_verdict = 'UNKNOWN'
        self.current_confidence = 0.0
        self.last_alert_time = 0
        
        # Performance metrics
        self.fps = 0
        self.processing_time = 0
        self.total_frames_processed = 0
        
        # Alert log
        self.alert_log = []
        
        print("✓ Real-time detector initialized")
    
    def _load_face_detector(self):
        """Load face detection model (MTCNN or Haar Cascade)"""
        try:
            # Try loading MTCNN (better accuracy)
            from facenet_pytorch import MTCNN
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return MTCNN(keep_all=False, device=device)
        except ImportError:
            # Fallback to Haar Cascade (faster, less accurate)
            print("⚠ MTCNN not available, using Haar Cascade")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            return cv2.CascadeClassifier(cascade_path)
    
    def detect_face(self, frame):
        """
        Detect face in frame and return bounding box
        Returns: (x, y, w, h) or None
        """
        if isinstance(self.face_detector, cv2.CascadeClassifier):
            # Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                return faces[0]  # Return first face
            return None
        else:
            # MTCNN
            boxes, _ = self.face_detector.detect(frame)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                x, y, x2, y2 = box.astype(int)
                return (x, y, x2-x, y2-y)
            return None
    
    def extract_face_crop(self, frame, bbox, target_size=(256, 256)):
        """Extract and resize face region"""
        if bbox is None:
            return None
        
        x, y, w, h = bbox
        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)
        
        # Add padding
        padding = int(w * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
        
        # Resize
        face_crop = cv2.resize(face_crop, target_size)
        return face_crop
    
    def analyze_frame(self, frame, landmarks=None):
        """
        Analyze a single frame for deepfake indicators
        
        Args:
            frame: BGR image
            landmarks: Optional pre-computed landmarks
        
        Returns:
            dict with detection results
        """
        result = {
            'verdict': 'NO_FACE',
            'confidence': 0.0,
            'face_bbox': None,
            'timestamp': time.time(),
            'details': {}
        }
        
        # Detect face
        face_bbox = self.detect_face(frame)
        if face_bbox is None:
            return result
        
        result['face_bbox'] = face_bbox
        
        # Extract face crop
        face_crop = self.extract_face_crop(frame, face_bbox)
        if face_crop is None:
            return result
        
        # Initialize scores
        temporal_score = 1.0
        texture_score = 1.0
        stability_score = 1.0
        
        # 1. Temporal Consistency (if we have history)
        if len(self.landmark_history) >= 3:
            temporal_score = self._check_temporal_consistency()
            result['details']['temporal_score'] = temporal_score
        
        # 2. Texture Analysis
        texture_score = self._analyze_texture(face_crop)
        result['details']['texture_score'] = texture_score
        
        # 3. Landmark Stability (if using landmark predictions)
        if landmarks is not None:
            self.landmark_history.append(landmarks)
            
            if len(self.landmark_history) >= 5:
                stability = self.deepfake_detector.calculate_landmark_stability(
                    list(self.landmark_history)
                )
                avg_variance = np.mean(list(stability.values())) if stability else 0
                stability_score = max(0, 1.0 - (avg_variance / 100.0))
                result['details']['stability_score'] = stability_score
        
        # 4. Edge Artifact Detection
        artifact_pct, _ = self.deepfake_detector.detect_boundary_artifacts(
            frame, face_bbox
        )
        artifact_score = max(0, 1.0 - (artifact_pct / 100.0))
        result['details']['artifact_score'] = artifact_score
        
        # Fusion scoring with weights
        weights = {
            'temporal': 0.35,
            'texture': 0.25,
            'stability': 0.20,
            'artifact': 0.20
        }
        
        raw_score = (
            weights['temporal'] * temporal_score +
            weights['texture'] * texture_score +
            weights['stability'] * stability_score +
            weights['artifact'] * artifact_score
        )
        
        # Apply temporal smoothing
        smoothed_score = self._apply_temporal_smoothing(raw_score)
        result['confidence'] = smoothed_score
        
        # Make verdict
        if smoothed_score > 0.85:
            result['verdict'] = 'AUTHENTIC'
        elif smoothed_score > 0.5:
            result['verdict'] = 'SUSPICIOUS'
        else:
            result['verdict'] = 'DEEPFAKE'
        
        return result
    
    def _check_temporal_consistency(self):
        """
        Check frame-to-frame temporal consistency
        Returns score 0-1 (1 = consistent, 0 = inconsistent)
        """
        if len(self.landmark_history) < 3:
            return 1.0
        
        # Calculate movement between consecutive frames
        recent_landmarks = list(self.landmark_history)[-5:]
        movements = []
        
        for i in range(len(recent_landmarks) - 1):
            # Simple approach: compare prediction arrays directly
            prev = recent_landmarks[i]
            curr = recent_landmarks[i + 1]
            
            # Calculate pixel-wise difference
            diff = np.sum(prev != curr) / prev.size
            movements.append(diff)
        
        # Analyze movement pattern
        if len(movements) > 0:
            avg_movement = np.mean(movements)
            movement_variance = np.var(movements)
            
            # Consistent movement = low variance, moderate avg
            # Score based on variance (lower = better)
            temporal_score = 1.0 / (1.0 + movement_variance * 10)
            
            # Penalize excessive movement
            if avg_movement > 0.3:
                temporal_score *= 0.7
            
            return np.clip(temporal_score, 0, 1)
        
        return 1.0
    
    def _analyze_texture(self, face_crop):
        """
        Analyze face texture for deepfake artifacts
        Returns score 0-1 (1 = natural, 0 = artificial)
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # 1. Local Binary Pattern (texture consistency)
        lbp = self._compute_lbp(gray)
        lbp_score = self._score_lbp(lbp)
        
        # 2. Edge sharpness (deepfakes often have blurry edges)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size
        edge_score = 1.0 if 0.05 < edge_density < 0.15 else 0.7
        
        # 3. Frequency analysis
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
        
        # Real faces have specific frequency patterns
        high_freq_energy = np.mean(magnitude_spectrum[128:, 128:])
        freq_score = 1.0 / (1.0 + abs(high_freq_energy - 50) / 50)
        
        # Combine texture scores
        texture_score = 0.4 * lbp_score + 0.3 * edge_score + 0.3 * freq_score
        return np.clip(texture_score, 0, 1)
    
    def _compute_lbp(self, gray_image, radius=1, n_points=8):
        """Compute Local Binary Pattern"""
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        return lbp
    
    def _score_lbp(self, lbp):
        """Score LBP pattern (simplified)"""
        # Real faces have specific LBP histogram patterns
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256), density=True)
        
        # Entropy of histogram (higher = more texture variation = real)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Normalize (real faces typically have entropy 4-6)
        score = np.clip((entropy - 3) / 3, 0, 1)
        return score
    
    def _apply_temporal_smoothing(self, current_score):
        """
        Apply exponential moving average for smooth confidence changes
        """
        self.confidence_history.append(current_score)
        
        if len(self.confidence_history) < 3:
            return current_score
        
        alpha = self.config['confidence_smoothing']
        recent_scores = list(self.confidence_history)[-5:]
        
        # Exponential moving average
        smoothed = current_score
        for i, score in enumerate(reversed(recent_scores[:-1])):
            weight = alpha ** (i + 1)
            smoothed = smoothed * (1 - weight) + score * weight
        
        return smoothed
    
    def _trigger_alert(self, verdict, confidence):
        """Trigger alert if deepfake detected"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.config['alert_cooldown']:
            return
        
        if verdict == 'DEEPFAKE':
            alert = {
                'timestamp': datetime.now().isoformat(),
                'verdict': verdict,
                'confidence': confidence,
                'severity': 'HIGH' if confidence < 0.3 else 'MEDIUM'
            }
            
            with self.lock:
                self.alert_log.append(alert)
            
            self.last_alert_time = current_time
            
            # Audio alert (if enabled)
            if self.config['enable_audio_alerts']:
                self._play_alert_sound()
            
            print(f"🚨 ALERT: Deepfake detected (confidence: {confidence:.1%})")
    
    def _play_alert_sound(self):
        """Play audio alert (non-blocking)"""
        try:
            import winsound
            # Quick beep (duration=200ms, freq=1000Hz)
            Thread(target=lambda: winsound.Beep(1000, 200), daemon=True).start()
        except:
            pass  # Audio not critical
    
    def process_video_stream(self, video_source=0, display=True):
        """
        Process video stream in real-time
        
        Args:
            video_source: 0 for webcam, or video file path
            display: Whether to show video with overlay
        """
        print(f"\n{'='*70}")
        print("🎥 IDENTIX LiveGuard - Real-Time Deepfake Detection")
        print(f"{'='*70}")
        print(f"Video Source: {video_source}")
        print(f"Detection Threshold: {self.config['detection_threshold']}")
        print(f"Frame Skip: {self.config['frame_skip']} (processing every {self.config['frame_skip']}th frame)")
        print(f"{'='*70}\n")
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {video_source}")
            return
        
        self.is_running = True
        frame_count = 0
        
        try:
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("⚠ End of video or camera disconnected")
                    break
                
                # Process every Nth frame
                if frame_count % self.config['frame_skip'] == 0:
                    start_time = time.time()
                    
                    # Analyze frame
                    result = self.analyze_frame(frame)
                    
                    # Update state
                    with self.lock:
                        self.current_verdict = result['verdict']
                        self.current_confidence = result['confidence']
                        self.processing_time = time.time() - start_time
                        self.fps = 1.0 / self.processing_time if self.processing_time > 0 else 0
                        self.total_frames_processed += 1
                    
                    # Check for alerts
                    self._trigger_alert(result['verdict'], result['confidence'])
                    
                    # Store in history
                    self.frame_history.append(result)
                
                # Display frame with overlay
                if display:
                    frame_with_overlay = self._draw_overlay(frame, result if frame_count % self.config['frame_skip'] == 0 else None)
                    cv2.imshow('IDENTIX LiveGuard - Real-Time Detection', frame_with_overlay)
                    
                    # Check for quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⏹ Stopping detection...")
                        break
                    elif key == ord('r'):
                        # Reset history
                        self._reset_history()
                        print("🔄 History reset")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\n⏹ Detection interrupted by user")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            self.is_running = False
            self._print_summary()
    
    def _draw_overlay(self, frame, result=None):
        """Draw detection overlay on frame"""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        if result is None:
            # Use last known result
            with self.lock:
                result = {
                    'verdict': self.current_verdict,
                    'confidence': self.current_confidence,
                    'face_bbox': None
                }
        
        # Color based on verdict
        color_map = {
            'AUTHENTIC': (0, 255, 0),     # Green
            'SUSPICIOUS': (0, 255, 255),  # Yellow
            'DEEPFAKE': (0, 0, 255),      # Red
            'NO_FACE': (128, 128, 128),   # Gray
            'UNKNOWN': (128, 128, 128)
        }
        color = color_map.get(result['verdict'], (128, 128, 128))
        
        # Draw face bounding box
        if result.get('face_bbox') is not None:
            x, y, w, h = result['face_bbox']
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 3)
        
        # Draw verdict banner
        banner_height = 80
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 0), -1)
        
        verdict_text = f"{result['verdict']}"
        conf_text = f"{result['confidence']:.1%}"
        
        # Verdict
        cv2.putText(overlay, verdict_text, (20, 50),
                   cv2.FONT_HERSHEY_BOLD, 1.5, color, 3)
        
        # Confidence
        cv2.putText(overlay, conf_text, (width - 200, 50),
                   cv2.FONT_HERSHEY_BOLD, 1.5, color, 3)
        
        # Draw confidence meter
        meter_width = 300
        meter_height = 20
        meter_x = width - meter_width - 20
        meter_y = banner_height + 20
        
        # Background
        cv2.rectangle(overlay, 
                     (meter_x, meter_y),
                     (meter_x + meter_width, meter_y + meter_height),
                     (50, 50, 50), -1)
        
        # Confidence bar
        bar_width = int(meter_width * result['confidence'])
        cv2.rectangle(overlay,
                     (meter_x, meter_y),
                     (meter_x + bar_width, meter_y + meter_height),
                     color, -1)
        
        # Stats
        with self.lock:
            fps_text = f"FPS: {self.fps:.1f}"
            time_text = f"Time: {self.processing_time*1000:.0f}ms"
            frames_text = f"Frames: {self.total_frames_processed}"
        
        cv2.putText(overlay, fps_text, (20, height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, time_text, (20, height - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, frames_text, (20, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Alert indicator
        if result['verdict'] == 'DEEPFAKE':
            alert_text = "⚠ DEEPFAKE DETECTED ⚠"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_BOLD, 1.0, 2)[0]
            text_x = (width - text_size[0]) // 2
            cv2.putText(overlay, alert_text, (text_x, height - 100),
                       cv2.FONT_HERSHEY_BOLD, 1.0, (0, 0, 255), 3)
        
        # Blend overlay
        alpha = 0.8
        frame_with_overlay = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame_with_overlay
    
    def _reset_history(self):
        """Reset temporal history"""
        with self.lock:
            self.landmark_history.clear()
            self.confidence_history.clear()
            self.frame_history.clear()
            self.current_verdict = 'UNKNOWN'
            self.current_confidence = 0.0
    
    def _print_summary(self):
        """Print detection session summary"""
        print(f"\n{'='*70}")
        print("📊 DETECTION SESSION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Frames Processed: {self.total_frames_processed}")
        print(f"Average Processing Time: {np.mean([r.get('timestamp', 0) for r in self.frame_history]) if self.frame_history else 0:.2f}s")
        print(f"Alerts Triggered: {len(self.alert_log)}")
        
        if self.alert_log:
            print("\n🚨 Alert Log:")
            for i, alert in enumerate(self.alert_log[-5:], 1):
                print(f"  {i}. {alert['timestamp']} - {alert['verdict']} ({alert['confidence']:.1%})")
        
        print(f"{'='*70}\n")
    
    def export_session_report(self, output_path='detection_report.json'):
        """Export session data to JSON"""
        report = {
            'session_info': {
                'total_frames': self.total_frames_processed,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            },
            'alerts': self.alert_log,
            'frame_history': [
                {
                    'verdict': r['verdict'],
                    'confidence': r['confidence'],
                    'timestamp': r.get('timestamp', 0)
                }
                for r in list(self.frame_history)[-100:]  # Last 100 frames
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Session report saved to {output_path}")
    
    def stop(self):
        """Stop detection"""
        self.is_running = False


def main():
    """Main function for testing"""
    print("🚀 Starting IDENTIX LiveGuard Real-Time Detector\n")
    
    # Configuration
    config = {
        'detection_threshold': 0.5,
        'temporal_window': 30,
        'frame_skip': 5,
        'alert_cooldown': 3.0,
        'confidence_smoothing': 0.3,
        'enable_audio_alerts': True
    }
    
    # Initialize detector
    detector = RealtimeDeepfakeDetector(config)
    
    # Process video (0 = webcam, or provide video file path)
    video_source = 0  # Change to video file path for testing
    
    try:
        detector.process_video_stream(video_source, display=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Export report
        detector.export_session_report()


if __name__ == '__main__':
    main()
