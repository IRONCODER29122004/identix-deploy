"""
Test Pipeline for Deepfake Detection Model v2 - FFA-MPDV

This module provides a separate inference pipeline specifically for the v2 model.
It processes video crops through the trained FFA-MPDV model and generates:
- Predictions (REAL vs FAKE)
- Confidence scores  
- Visual overlays with prediction heatmaps
- JSON results with per-frame detection metrics

DOES NOT modify existing pipelines (v1, segmentation models remain untouched)
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import warnings

# Import the v2 detector module
from deepfake_detector_v2 import DeepfakeDetectorV2, load_v2_model


class DeepfakeDetectionPipelineV2:
    """
    Standalone pipeline for deepfake detection v2.
    Processes crops and generates predictions with confidence scores.
    """
    
    def __init__(self, model_path, device='cuda', output_threshold=0.5):
        """
        Initialize the detection pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
            output_threshold: Decision threshold for binary classification (default 0.5)
        """
        self.model_path = model_path
        self.device = device
        self.threshold = output_threshold
        
        # Load model
        print(f"Loading FFA-MPDV v2 model from: {model_path}")
        self.detector = load_v2_model(model_path, device=device)
        
        # Log model info
        model_info = self.detector.get_model_info()
        print(f"\n=== Model Information ===")
        print(f"Model Name: {model_info['name']}")
        print(f"Device: {model_info['device']}")
        print(f"Config: {model_info['config']}")
        print(f"Training History Length: {model_info['training_history_length']} epochs")
        print(f"Final Metrics: {model_info['final_metrics']}")
        print()
        
    def process_single_image(self, image_path):
        """
        Detect deepfake in a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Prediction result with metadata
        """
        if not os.path.exists(image_path):
            return {
                'image': image_path,
                'success': False,
                'error': 'File not found'
            }
        
        try:
            prediction = self.detector.predict(
                image_path,
                return_proba=True,
                threshold=self.threshold
            )
            
            # Enrich result with file info
            result = {
                'image': image_path,
                'success': True,
                'label': prediction['label_name'],
                'probability': prediction['proba'],
                'logit': prediction['logit'],
                'confidence': prediction['confidence'],
            }
            
            return result
            
        except Exception as e:
            return {
                'image': image_path,
                'success': False,
                'error': str(e)
            }
    
    def process_crops_directory(self, crops_dir, output_overlays_dir=None, 
                               output_results_json=None, verbose=True):
        """
        Process all crop images in a directory.
        
        Args:
            crops_dir: Directory containing crop images
            output_overlays_dir: Where to save overlayed predictions (optional)
            output_results_json: Where to save JSON results (optional)
            verbose: Print progress information
            
        Returns:
            dict: Summary results and per-image predictions
        """
        crops_dir = Path(crops_dir)
        
        if not crops_dir.exists():
            raise ValueError(f"Crops directory not found: {crops_dir}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            f for f in crops_dir.glob('*')
            if f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            print(f"No image files found in {crops_dir}")
            return {'count': 0, 'predictions': []}
        
        predictions = []
        fake_count = 0
        real_count = 0
        confidence_scores = []
        
        # Process each image
        iterator = tqdm(image_files, desc="Processing crops") if verbose else image_files
        
        for img_path in iterator:
            result = self.process_single_image(str(img_path))
            predictions.append(result)
            
            if result['success']:
                if result['label'] == 'FAKE':
                    fake_count += 1
                else:
                    real_count += 1
                confidence_scores.append(result['confidence'])
        
        # Generate summary
        summary = {
            'total_images': len(predictions),
            'success_count': sum(1 for p in predictions if p['success']),
            'error_count': sum(1 for p in predictions if not p['success']),
            'fake_count': fake_count,
            'real_count': real_count,
            'fake_percentage': (fake_count / len(predictions) * 100) if predictions else 0,
            'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'threshold': self.threshold,
            'model_path': str(self.model_path),
            'predictions': predictions
        }
        
        # Save JSON results if requested
        if output_results_json:
            output_path = Path(output_results_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            if verbose:
                print(f"Results saved to: {output_path}")
        
        # Create visualization overlays if requested
        if output_overlays_dir:
            self._create_visualization_overlays(
                image_files,
                predictions,
                output_overlays_dir,
                verbose=verbose
            )
        
        return summary
    
    def _create_visualization_overlays(self, image_files, predictions, 
                                       output_dir, verbose=True):
        """
        Create overlayed images with prediction labels.
        
        Args:
            image_files: List of image file paths
            predictions: List of prediction results
            output_dir: Where to save overlayed images
            verbose: Print progress
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for img_file, pred in zip(image_files, predictions):
            if not pred['success']:
                continue
            
            try:
                # Load image
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                # Determine color based on prediction (red for fake, green for real)
                if pred['label'] == 'FAKE':
                    color = (0, 0, 255)  # Red in BGR
                    label_text = f"FAKE ({pred['probability']:.2%})"
                else:
                    color = (0, 255, 0)  # Green in BGR
                    label_text = f"REAL ({1-pred['probability']:.2%})"
                
                # Add border and text
                h, w = img.shape[:2]
                border_thickness = 8
                cv2.rectangle(img, (0, 0), (w, h), color, border_thickness)
                
                # Add text label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                text_x = (w - text_size[0]) // 2
                text_y = 50
                
                # Text background for readability
                cv2.rectangle(img, 
                            (text_x - 10, text_y - text_size[1] - 10),
                            (text_x + text_size[0] + 10, text_y + 10),
                            color, -1)
                cv2.putText(img, label_text, (text_x, text_y),
                           font, font_scale, (255, 255, 255), thickness)
                
                # Save overlay
                overlay_filename = f"v2_overlay_{img_file.stem}.jpg"
                overlay_path = output_path / overlay_filename
                cv2.imwrite(str(overlay_path), img)
                
            except Exception as e:
                if verbose:
                    print(f"Failed to create overlay for {img_file}: {e}")
                continue
    
    def generate_report(self, summary):
        """
        Generate a human-readable report from results.
        
        Args:
            summary: Results dict from process_crops_directory
            
        Returns:
            str: Formatted report
        """
        report = f"""
{'='*60}
DEEPFAKE DETECTION v2 PIPELINE REPORT
{'='*60}

Model: {self.detector.get_model_info()['name']}
Decision Threshold: {self.threshold}

RESULTS SUMMARY:
  Total Images:       {summary['total_images']}
  Successful:         {summary['success_count']}
  Errors:             {summary['error_count']}
  
PREDICTIONS:
  Fake Videos:        {summary['fake_count']} ({summary['fake_percentage']:.1f}%)
  Real Videos:        {summary['real_count']} ({100-summary['fake_percentage']:.1f}%)
  Mean Confidence:    {summary['mean_confidence']:.3f}

{'='*60}
"""
        return report


def run_v2_inference_on_crops(crops_dir, output_dir=None, model_path=None, 
                              threshold=0.5, device='cuda'):
    """
    Convenience function to run v2 inference on crops.
    
    Args:
        crops_dir: Directory with crop images
        output_dir: Base output directory (will create overlays/ and results/ subdirs)
        model_path: Path to v2 model checkpoint
        threshold: Decision threshold
        device: 'cuda' or 'cpu'
        
    Returns:
        dict: Summary results
    """
    # Determine model path if not provided
    if model_path is None:
        # Look for v2 model in standard location
        default_path = Path('models/deepfake_detector_v2_ffa_mpdv.pth')
        if default_path.exists():
            model_path = str(default_path)
        else:
            raise FileNotFoundError(
                "Model path not provided and default not found at: models/deepfake_detector_v2_ffa_mpdv.pth"
            )
    
    # Setup output directories
    if output_dir is None:
        output_dir = 'data/v2_detection_results'
    
    output_dir = Path(output_dir)
    overlays_dir = output_dir / 'overlays'
    results_json = output_dir / 'results.json'
    
    # Run pipeline
    pipeline = DeepfakeDetectionPipelineV2(model_path, device=device, output_threshold=threshold)
    
    summary = pipeline.process_crops_directory(
        crops_dir,
        output_overlays_dir=str(overlays_dir),
        output_results_json=str(results_json),
        verbose=True
    )
    
    # Generate and print report
    report = pipeline.generate_report(summary)
    print(report)
    
    # Save report to file
    report_path = output_dir / 'REPORT.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    return summary


if __name__ == '__main__':
    """
    Example usage:
    python test_pipeline_v2.py <crops_directory> [<output_directory>] [<model_path>]
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline_v2.py <crops_dir> [output_dir] [model_path]")
        sys.exit(1)
    
    crops_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    model_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    run_v2_inference_on_crops(crops_dir, output_dir=output_dir, model_path=model_path)
