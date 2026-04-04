"""
Hybrid Deepfake Detector
Combines rule-based detector + ML model for maximum reliability
Uses ensemble voting for final verdict
"""

import numpy as np
from deepfake_detector import DeepfakeDetector
from ml_deepfake_detector import XceptionDeepfakeDetector
import cv2

class HybridDeepfakeDetector:
    """
    Ensemble detector combining:
    1. Rule-based temporal/artifact analysis (fast)
    2. ML-based Xception model (accurate)
    
    Makes final decision based on consensus
    """
    
    def __init__(self, ml_model_path=None, use_ml=True, confidence_threshold=0.7):
        """
        Args:
            ml_model_path: Path to Xception model weights
            use_ml: Whether to use ML model (if False, only rule-based)
            confidence_threshold: Min confidence for final verdict
        """
        self.rule_detector = DeepfakeDetector()
        self.use_ml = use_ml
        self.confidence_threshold = confidence_threshold
        
        self.ml_detector = None
        if use_ml and ml_model_path:
            try:
                self.ml_detector = XceptionDeepfakeDetector(ml_model_path)
                print("✓ ML model loaded successfully")
            except Exception as e:
                print(f"⚠ Could not load ML model: {e}")
                self.use_ml = False
    
    def detect_deepfake(self, predictions_sequence, frames, face_bboxes):
        """
        Detect deepfake using both methods
        
        Returns:
            Combined report with verdicts from both models
        """
        # 1. Rule-based detection
        rule_report = self.rule_detector.detect_deepfake(
            predictions_sequence, frames, face_bboxes
        )
        
        result = {
            'rule_based': rule_report,
            'ml_based': None,
            'final_verdict': rule_report['verdict'],
            'final_confidence': rule_report['confidence'],
            'method': 'rule-based'
        }
        
        # 2. ML-based detection (if available and we have face crops)
        if self.use_ml and self.ml_detector and len(frames) > 0:
            ml_results = self._analyze_with_ml(frames, face_bboxes)
            result['ml_based'] = ml_results
            
            # 3. Combine results
            final_verdict = self._ensemble_verdict(rule_report, ml_results)
            result['final_verdict'] = final_verdict['verdict']
            result['final_confidence'] = final_verdict['confidence']
            result['method'] = 'hybrid-ensemble'
            result['analysis'] = self._generate_analysis(rule_report, ml_results)
        
        return result
    
    def _analyze_with_ml(self, frames, face_bboxes):
        """Analyze frames using ML model"""
        ml_results = {
            'individual_predictions': [],
            'average_confidence': 0.0,
            'deepfake_votes': 0,
            'authentic_votes': 0,
            'verdict': 'NO_FACES_DETECTED'
        }
        
        if not self.ml_detector:
            return ml_results
        
        face_count = 0
        
        for frame, bbox in zip(frames, face_bboxes):
            if bbox is None:
                continue
            
            # Convert PIL Image to numpy array if needed
            if hasattr(frame, 'mode'):  # PIL Image
                frame = np.array(frame)
                # Convert RGB to BGR for OpenCV compatibility
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Extract face crop
            x, y, w, h = bbox
            padding = int(w * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            # ML prediction
            try:
                pred = self.ml_detector.detect(face_crop)
                ml_results['individual_predictions'].append(pred)
                
                if pred['is_authentic']:
                    ml_results['authentic_votes'] += 1
                else:
                    ml_results['deepfake_votes'] += 1
                
                face_count += 1
            except Exception as e:
                print(f"⚠ ML detection error: {e}")
                continue
        
        if face_count > 0:
            # Average confidence
            confidences = [p['confidence'] for p in ml_results['individual_predictions']]
            ml_results['average_confidence'] = np.mean(confidences)
            
            # Majority vote (require strong consensus - 70%+ to call deepfake)
            total_votes = ml_results['deepfake_votes'] + ml_results['authentic_votes']
            deepfake_ratio = ml_results['deepfake_votes'] / total_votes if total_votes > 0 else 0
            
            if deepfake_ratio >= 0.7:  # 70% or more votes for deepfake
                ml_results['verdict'] = 'DEEPFAKE'
            elif deepfake_ratio <= 0.3:  # 30% or less votes for deepfake (70%+ authentic)
                ml_results['verdict'] = 'AUTHENTIC'
            else:
                ml_results['verdict'] = 'UNCERTAIN'  # Mixed results
        
        return ml_results
    
    def _ensemble_verdict(self, rule_report, ml_report):
        """
        Combine verdicts from rule-based and ML models
        Uses weighted voting and confidence scores
        """
        # Extract verdicts
        rule_verdict = rule_report['verdict']
        rule_confidence = rule_report['confidence']
        
        ml_verdict = ml_report.get('verdict', 'UNCERTAIN')
        ml_confidence = ml_report.get('average_confidence', 0.0)
        
        # Determine if models agree
        rule_is_fake = 'DEEPFAKE' in rule_verdict
        ml_is_fake = 'DEEPFAKE' in ml_verdict
        
        # Weighted voting (Rule-based gets more weight since ML not trained yet)
        weights = {'ml': 0.25, 'rule': 0.75}
        
        if ml_is_fake == rule_is_fake:
            # Models agree
            avg_confidence = (ml_confidence * weights['ml'] + 
                            rule_confidence * weights['rule'])
            verdict = 'DEEPFAKE' if ml_is_fake else 'AUTHENTIC'
            agreement = 'AGREEMENT (HIGH CONFIDENCE)'
        else:
            # Models disagree - take ML prediction with lower confidence
            avg_confidence = (ml_confidence * weights['ml'] + 
                            rule_confidence * weights['rule'])
            verdict = 'DEEPFAKE' if ml_is_fake else 'AUTHENTIC'
            agreement = 'DISAGREEMENT (REVIEW RECOMMENDED)'
        
        return {
            'verdict': f"{verdict} - {agreement}",
            'confidence': round(avg_confidence, 2),
            'agreement': agreement
        }
    
    def _generate_analysis(self, rule_report, ml_report):
        """Generate detailed analysis comparing both methods"""
        return {
            'rule_based_score': rule_report.get('confidence', 0),
            'rule_based_verdict': rule_report.get('verdict', 'UNKNOWN'),
            'rule_based_warnings': rule_report.get('warnings', []),
            
            'ml_based_score': ml_report.get('average_confidence', 0),
            'ml_based_verdict': ml_report.get('verdict', 'UNKNOWN'),
            'ml_deepfake_votes': ml_report.get('deepfake_votes', 0),
            'ml_authentic_votes': ml_report.get('authentic_votes', 0),
            'ml_predictions_count': len(ml_report.get('individual_predictions', [])),
            
            'note': 'ML model provides better accuracy on modern deepfakes'
        }


# Usage example
if __name__ == '__main__':
    # Initialize hybrid detector
    detector = HybridDeepfakeDetector(
        ml_model_path='Required/models/xception_ff.pth',
        use_ml=True
    )
    
    # Later... when analyzing video:
    result = detector.detect_deepfake(
        predictions_sequence=predictions,  # from BiSeNet
        frames=frames,
        face_bboxes=bboxes
    )
    
    print("\n" + "="*70)
    print("DEEPFAKE DETECTION RESULT")
    print("="*70)
    print(f"Method: {result['method']}")
    print(f"Verdict: {result['final_verdict']}")
    print(f"Confidence: {result['final_confidence']:.2f}%")
    print("\nDetailed Analysis:")
    for key, value in result['analysis'].items():
        print(f"  {key}: {value}")
    print("="*70)
