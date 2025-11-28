"""
Deepfake Detection Module
Analyzes videos for signs of manipulation using facial landmarks
"""

import numpy as np
import cv2
from collections import defaultdict
from scipy import stats
from scipy.spatial import distance
import torch

class DeepfakeDetector:
    """
    Detects deepfakes by analyzing:
    1. Landmark movement consistency
    2. Temporal coherence
    3. Facial boundary artifacts
    4. Blink pattern regularity
    5. Landmark stability
    """
    
    def __init__(self):
        self.temporal_threshold = 0.15  # Max allowed frame-to-frame change
        self.artifact_threshold = 30    # Gradient threshold for artifacts
        self.blink_min_frames = 2       # Minimum frames for valid blink
        self.blink_max_frames = 8       # Maximum frames for valid blink
        
    def calculate_landmark_distances(self, prediction):
        """
        Calculate key distances between landmarks
        Returns: dict of distance measurements
        """
        distances = {}
        
        # Find centroids of key landmarks
        def get_centroid(pred, landmark_id):
            mask = (pred == landmark_id)
            if not np.any(mask):
                return None
            coords = np.column_stack(np.where(mask))
            return coords.mean(axis=0)
        
        # Get centroids
        left_eye = get_centroid(prediction, 4)
        right_eye = get_centroid(prediction, 5)
        nose = get_centroid(prediction, 6)
        mouth = get_centroid(prediction, 8)
        
        # Calculate distances
        if left_eye is not None and right_eye is not None:
            distances['eye_distance'] = np.linalg.norm(left_eye - right_eye)
        
        if left_eye is not None and nose is not None:
            distances['left_eye_nose'] = np.linalg.norm(left_eye - nose)
        
        if right_eye is not None and nose is not None:
            distances['right_eye_nose'] = np.linalg.norm(right_eye - nose)
        
        if nose is not None and mouth is not None:
            distances['nose_mouth'] = np.linalg.norm(nose - mouth)
        
        return distances
    
    def analyze_temporal_consistency(self, predictions_sequence):
        """
        Analyze how smoothly landmarks move between frames
        Deepfakes often have jumpy, inconsistent movements
        """
        inconsistencies = []
        distance_changes = defaultdict(list)
        
        for i in range(1, len(predictions_sequence)):
            prev_distances = self.calculate_landmark_distances(predictions_sequence[i-1])
            curr_distances = self.calculate_landmark_distances(predictions_sequence[i])
            
            # Compare changes in distances
            for key in prev_distances:
                if key in curr_distances:
                    prev_val = prev_distances[key]
                    curr_val = curr_distances[key]
                    
                    if prev_val > 0:  # Avoid division by zero
                        relative_change = abs(curr_val - prev_val) / prev_val
                        distance_changes[key].append(relative_change)
                        
                        # Flag if change is too large (unnatural movement)
                        if relative_change > self.temporal_threshold:
                            inconsistencies.append({
                                'frame': i,
                                'metric': key,
                                'change': relative_change
                            })
        
        # Calculate statistics
        avg_changes = {}
        for key, changes in distance_changes.items():
            if changes:
                avg_changes[key] = np.mean(changes)
        
        return inconsistencies, avg_changes
    
    def detect_boundary_artifacts(self, frame, face_bbox):
        """
        Detect artifacts around face boundaries (common in deepfakes)
        Uses gradient analysis to find unnatural edges
        """
        x, y, w, h = face_bbox
        
        # Expand bbox slightly to check boundary region
        padding = int(w * 0.1)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Extract boundary region
        boundary_region = frame[y1:y2, x1:x2]
        
        if boundary_region.size == 0:
            return 0, []
        
        # Convert to grayscale
        gray = cv2.cvtColor(boundary_region, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find high gradient regions (potential artifacts)
        high_gradients = gradient_magnitude > self.artifact_threshold
        artifact_count = np.sum(high_gradients)
        artifact_percentage = (artifact_count / high_gradients.size) * 100
        
        # Get artifact locations
        artifact_locations = np.column_stack(np.where(high_gradients))
        
        return artifact_percentage, artifact_locations
    
    def analyze_blink_pattern(self, predictions_sequence):
        """
        Analyze eye blink patterns
        Deepfakes often have irregular or missing blinks
        """
        blinks = []
        eye_states = []  # 1 = open, 0 = closed/partially closed
        
        for pred in predictions_sequence:
            # Check if eyes are visible
            left_eye_pixels = np.sum(pred == 4)
            right_eye_pixels = np.sum(pred == 5)
            
            # Eyes are "closed" if very few pixels detected
            threshold = 50
            if left_eye_pixels < threshold and right_eye_pixels < threshold:
                eye_states.append(0)
            else:
                eye_states.append(1)
        
        # Detect blink events (transition from open -> closed -> open)
        i = 0
        while i < len(eye_states) - 2:
            if eye_states[i] == 1:  # Eye open
                # Find closure
                close_start = None
                for j in range(i+1, len(eye_states)):
                    if eye_states[j] == 0:
                        close_start = j
                        break
                
                if close_start is not None:
                    # Find reopening
                    for k in range(close_start+1, len(eye_states)):
                        if eye_states[k] == 1:
                            blink_duration = k - close_start
                            if self.blink_min_frames <= blink_duration <= self.blink_max_frames:
                                blinks.append({
                                    'start_frame': close_start,
                                    'duration': blink_duration
                                })
                            i = k
                            break
                    else:
                        break
                else:
                    break
            i += 1
        
        # Analyze blink regularity
        if len(blinks) > 1:
            intervals = [blinks[i+1]['start_frame'] - blinks[i]['start_frame'] 
                        for i in range(len(blinks)-1)]
            blink_regularity = np.std(intervals) if intervals else 0
        else:
            blink_regularity = float('inf')  # No pattern
        
        return blinks, blink_regularity
    
    def calculate_landmark_stability(self, predictions_sequence):
        """
        Calculate how stable landmarks are (jitter detection)
        Real faces have smooth movements, deepfakes can be jittery
        """
        stability_scores = {}
        
        for landmark_id in range(1, 11):
            centroids = []
            
            for pred in predictions_sequence:
                mask = (pred == landmark_id)
                if np.any(mask):
                    coords = np.column_stack(np.where(mask))
                    centroid = coords.mean(axis=0)
                    centroids.append(centroid)
            
            if len(centroids) > 1:
                # Calculate variance in centroid positions
                centroids = np.array(centroids)
                variance = np.var(centroids, axis=0)
                stability_scores[landmark_id] = np.mean(variance)
        
        return stability_scores
    
    def detect_deepfake(self, predictions_sequence, frames, face_bboxes):
        """
        Main detection function - combines all analysis methods
        Returns: authenticity score (0-100), detailed report
        """
        report = {
            'is_authentic': True,
            'confidence': 100.0,
            'warnings': [],
            'details': {}
        }
        
        if len(predictions_sequence) < 3:
            report['warnings'].append("Not enough frames for reliable analysis")
            return report
        
        # 1. Temporal Consistency Analysis
        inconsistencies, avg_changes = self.analyze_temporal_consistency(predictions_sequence)
        temporal_score = max(0, 100 - len(inconsistencies) * 5)
        
        report['details']['temporal_consistency'] = {
            'score': temporal_score,
            'inconsistencies': len(inconsistencies),
            'avg_changes': avg_changes
        }
        
        if len(inconsistencies) > 5:
            report['warnings'].append(f"High temporal inconsistency detected ({len(inconsistencies)} jumps)")
        
        # 2. Boundary Artifact Detection
        artifact_scores = []
        for i, (frame, bbox) in enumerate(zip(frames, face_bboxes)):
            if bbox is not None:
                artifact_pct, _ = self.detect_boundary_artifacts(np.array(frame), bbox)
                artifact_scores.append(artifact_pct)
        
        avg_artifacts = np.mean(artifact_scores) if artifact_scores else 0
        artifact_score = max(0, 100 - avg_artifacts * 2)
        
        report['details']['boundary_artifacts'] = {
            'score': artifact_score,
            'avg_percentage': float(avg_artifacts)
        }
        
        if avg_artifacts > 15:
            report['warnings'].append(f"Suspicious artifacts detected ({avg_artifacts:.1f}%)")
        
        # 3. Blink Pattern Analysis
        blinks, blink_regularity = self.analyze_blink_pattern(predictions_sequence)
        expected_blinks = len(predictions_sequence) / 30  # Assuming ~30 fps, 1 blink per second
        blink_score = max(0, 100 - abs(len(blinks) - expected_blinks) * 10)
        
        report['details']['blink_analysis'] = {
            'score': blink_score,
            'blink_count': len(blinks),
            'regularity': float(blink_regularity) if blink_regularity != float('inf') else None
        }
        
        if len(blinks) < expected_blinks * 0.3:
            report['warnings'].append(f"Abnormal blink rate ({len(blinks)} blinks in {len(predictions_sequence)} frames)")
        
        # 4. Landmark Stability
        stability_scores = self.calculate_landmark_stability(predictions_sequence)
        avg_stability = np.mean(list(stability_scores.values())) if stability_scores else 0
        stability_score = max(0, 100 - avg_stability)
        
        report['details']['landmark_stability'] = {
            'score': stability_score,
            'avg_variance': float(avg_stability)
        }
        
        if avg_stability > 50:
            report['warnings'].append(f"High landmark jitter detected (variance: {avg_stability:.2f})")
        
        # Calculate overall authenticity score (weighted average)
        weights = {
            'temporal': 0.35,
            'artifacts': 0.30,
            'blinks': 0.20,
            'stability': 0.15
        }
        
        overall_score = (
            temporal_score * weights['temporal'] +
            artifact_score * weights['artifacts'] +
            blink_score * weights['blinks'] +
            stability_score * weights['stability']
        )
        
        report['confidence'] = round(overall_score, 2)
        
        # Determine authenticity
        if overall_score < 50:
            report['is_authentic'] = False
            report['verdict'] = "LIKELY DEEPFAKE"
        elif overall_score < 70:
            report['is_authentic'] = None  # Uncertain
            report['verdict'] = "SUSPICIOUS - Manual Review Recommended"
        else:
            report['is_authentic'] = True
            report['verdict'] = "LIKELY AUTHENTIC"
        
        return report
