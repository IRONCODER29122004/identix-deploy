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
        # RELAXED thresholds for phone videos (compression artifacts are normal)
        self.temporal_threshold = 0.50  # Max allowed frame-to-frame change (RELAXED for phones)
        self.artifact_threshold = 65    # Gradient threshold for artifacts (RELAXED for phones)  
        self.blink_min_frames = 2       # Minimum frames for valid blink
        self.blink_max_frames = 10      # Maximum frames for valid blink (phones may lag)
        
        # AI-generation detection thresholds
        self.ai_texture_smoothness_threshold = 50  # Texture variance threshold
        self.ai_color_uniformity_threshold = 15    # Color std deviation threshold
        self.ai_micro_movement_threshold = 20      # Micro-expression threshold
        
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
    
    def detect_ai_texture_smoothness(self, face_crop):
        """
        Detect overly smooth textures (AI-generated content indicator)
        Real skin has natural texture variations, AI is often too smooth
        """
        if face_crop is None or face_crop.size == 0:
            return False, 0
        
        try:
            # Ensure face_crop is numpy array and correct type
            if isinstance(face_crop, np.ndarray):
                face_array = face_crop
            else:
                face_array = np.array(face_crop)
            
            # Convert to grayscale
            if len(face_array.shape) == 3:
                gray = cv2.cvtColor(face_array.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                gray = face_array.astype(np.uint8)
            
            # Use Laplacian operator to detect edges (texture detail)
            # Higher variance means more texture detail
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            
            # Real skin: variance typically > 50
            # AI skin: variance often < 30 (too smooth)
            is_ai_smooth = texture_variance < self.ai_texture_smoothness_threshold
            
            return is_ai_smooth, texture_variance
        except Exception as e:
            print(f"[WARNING] Error in detect_ai_texture_smoothness: {e}")
            return False, 0
    
    def detect_ai_color_uniformity(self, face_crop):
        """
        Detect unnaturally uniform lighting/colors (AI-generated indicator)
        Real faces have natural lighting variation, AI is too uniform
        """
        if face_crop is None or face_crop.size == 0:
            return False, 0
        
        try:
            # Ensure face_crop is numpy array and correct type
            if isinstance(face_crop, np.ndarray):
                face_array = face_crop.copy()
            else:
                face_array = np.array(face_crop)
            
            # Ensure it's uint8 for color conversion
            if face_array.dtype != np.uint8:
                face_array = face_array.astype(np.uint8)
            
            # Convert to grayscale to measure luminance variation
            if len(face_array.shape) == 3 and face_array.shape[2] >= 3:
                gray = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_array if len(face_array.shape) == 2 else face_array[:,:,0]
            
            # Calculate standard deviation of brightness
            std_dev = np.std(gray)
            
            # Real faces: 20-50
            # AI faces: < 15 (too uniform)
            is_too_uniform = std_dev < self.ai_color_uniformity_threshold
            
            return is_too_uniform, std_dev
        except Exception as e:
            print(f"[WARNING] Error in detect_ai_color_uniformity: {e}")
            return False, 0
    
    def detect_ai_lack_micro_movements(self, predictions_sequence):
        """
        Detect lack of natural micro-expressions (AI-generated indicator) 
        Real humans have subtle micro-movements, AI is often too static
        """
        if len(predictions_sequence) < 5:
            return False, 0
        
        micro_movements = []
        
        for i in range(1, len(predictions_sequence)):
            try:
                current = predictions_sequence[i]
                previous = predictions_sequence[i-1]
                
                # Check if shapes match before subtraction
                if current.shape != previous.shape:
                    # Resize to match the smaller dimension
                    min_h = min(current.shape[0], previous.shape[0])
                    min_w = min(current.shape[1], previous.shape[1])
                    
                    if current.shape != (min_h, min_w):
                        current = cv2.resize(current, (min_w, min_h), interpolation=cv2.INTER_NEAREST)
                    if previous.shape != (min_h, min_w):
                        previous = cv2.resize(previous, (min_w, min_h), interpolation=cv2.INTER_NEAREST)
                
                # Calculate pixel-level changes
                diff = np.abs(current.astype(np.int16) - previous.astype(np.int16))
                
                # Count tiny 1-2 pixel changes
                micro_changes = np.count_nonzero((diff >= 1) & (diff <= 2))
                micro_movements.append(micro_changes)
            except Exception as e:
                print(f"[WARNING] Error comparing frames in micro-movement detection: {e}")
                continue
        
        if not micro_movements:
            return False, 0
        
        avg_micro = np.mean(micro_movements)
        
        # Real: 50-200 micro-movements
        # AI: < 20 (too static between frames)
        lacks_micro = avg_micro < self.ai_micro_movement_threshold
        
        return lacks_micro, avg_micro
    
    def analyze_ai_generation_indicators(self, frames, face_bboxes, predictions_sequence):
        """
        Analyze for AI-generated content (Gemini, Sora, etc.)
        Different from face-swap deepfake detection!
        """
        ai_indicators = {
            'is_ai_generated': False,
            'warnings': [],
            'scores': {}
        }
        
        # 1. Texture Smoothness Analysis
        smooth_count = 0
        texture_variances = []
        
        for frame, bbox in zip(frames[:20], face_bboxes[:20]):  # Sample first 20 frames
            if bbox is not None:
                try:
                    # Convert PIL Image to numpy array if needed
                    if hasattr(frame, 'mode'):  # PIL Image
                        frame_array = np.array(frame)
                    else:
                        frame_array = frame
                    
                    # Ensure it's BGR (OpenCV format)
                    if len(frame_array.shape) == 3:
                        if frame_array.shape[2] == 3:  # RGB or BGR
                            if hasattr(frame, 'mode') and frame.mode == 'RGB':
                                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    
                    x, y, w, h = bbox
                    # Ensure coordinates are integers
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    
                    # Extract face crop safely
                    face_crop = frame_array[y:y+h, x:x+w]
                    
                    if face_crop.size > 0:
                        is_smooth, variance = self.detect_ai_texture_smoothness(face_crop)
                        texture_variances.append(variance)
                        if is_smooth:
                            smooth_count += 1
                except Exception as e:
                    print(f"[DEBUG] Error extracting texture in frame: {e}")
                    continue
        
        if texture_variances:
            avg_texture_var = np.mean(texture_variances)
            ai_indicators['scores']['texture_variance'] = float(avg_texture_var)
            
            if smooth_count / len(texture_variances) > 0.7:
                ai_indicators['warnings'].append(
                    f"AI-GENERATED: Overly smooth textures (variance: {avg_texture_var:.1f})"
                )
                ai_indicators['is_ai_generated'] = True
        
        # 2. Color Uniformity Analysis
        uniform_count = 0
        color_stds = []
        
        for frame, bbox in zip(frames[:20], face_bboxes[:20]):
            if bbox is not None:
                try:
                    # Convert PIL Image to numpy array if needed
                    if hasattr(frame, 'mode'):  # PIL Image
                        frame_array = np.array(frame)
                    else:
                        frame_array = frame
                    
                    # Ensure it's BGR (OpenCV format)
                    if len(frame_array.shape) == 3:
                        if frame_array.shape[2] == 3:  # RGB or BGR
                            if hasattr(frame, 'mode') and frame.mode == 'RGB':
                                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    
                    x, y, w, h = bbox
                    # Ensure coordinates are integers
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    
                    # Extract face crop safely
                    face_crop = frame_array[y:y+h, x:x+w]
                    
                    if face_crop.size > 0:
                        is_uniform, std_dev = self.detect_ai_color_uniformity(face_crop)
                        color_stds.append(std_dev)
                        if is_uniform:
                            uniform_count += 1
                except Exception as e:
                    print(f"[DEBUG] Error extracting color uniformity in frame: {e}")
                    continue
        
        if color_stds:
            avg_color_std = np.mean(color_stds)
            ai_indicators['scores']['color_std'] = float(avg_color_std)
            
            if uniform_count / len(color_stds) > 0.7:
                ai_indicators['warnings'].append(
                    f"AI-GENERATED: Unnatural lighting uniformity (std: {avg_color_std:.1f})"
                )
                ai_indicators['is_ai_generated'] = True
        
        # 3. Micro-Movement Analysis
        lacks_micro, avg_micro = self.detect_ai_lack_micro_movements(predictions_sequence)
        ai_indicators['scores']['micro_movements'] = float(avg_micro)
        
        if lacks_micro:
            ai_indicators['warnings'].append(
                f"AI-GENERATED: Lack of natural micro-expressions (score: {avg_micro:.1f})"
            )
            ai_indicators['is_ai_generated'] = True
        
        return ai_indicators
    
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
        # Reduce penalty for phone videos (compression causes micro-jumps)
        temporal_score = max(0, 100 - len(inconsistencies) * 3)  # Reduced from * 5
        
        report['details']['temporal_consistency'] = {
            'score': temporal_score,
            'inconsistencies': len(inconsistencies),
            'avg_changes': avg_changes
        }
        
        if len(inconsistencies) > 8:  # Increased threshold from 5 (phones have more compression artifacts)
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
        
        # 5. AI-Generated Content Detection (NEW!)
        ai_indicators = self.analyze_ai_generation_indicators(frames, face_bboxes, predictions_sequence)
        report['details']['ai_generation'] = ai_indicators
        
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
        
        # Adjust score for AI-generated content
        if ai_indicators['is_ai_generated']:
            report['warnings'].extend(ai_indicators['warnings'])
            # Heavy penalty for AI-generated indicators
            overall_score = max(0, overall_score - 40)
            report['is_authentic'] = False
            report['verdict'] = "AI-GENERATED CONTENT DETECTED"
            report['confidence'] = round(overall_score, 2)
            return report
        
        report['confidence'] = round(overall_score, 2)
        
        # Determine authenticity (RELAXED thresholds for phone videos)
        if overall_score < 40:  # Was 50 - more lenient
            report['is_authentic'] = False
            report['verdict'] = "DEEPFAKE"
        elif overall_score < 60:  # Was 70 - more lenient  
            report['is_authentic'] = None  # Uncertain
            report['verdict'] = "SUSPICIOUS - MANUAL REVIEW RECOMMENDED"
        else:
            report['is_authentic'] = True
            report['verdict'] = "AUTHENTIC"
        
        return report
