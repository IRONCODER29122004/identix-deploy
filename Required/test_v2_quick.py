"""
Quick Test Script for Deepfake Detection v2 Model

This script verifies that the v2 model loads correctly and can make predictions.
Useful for sanity-checking the model integration.

Usage:
    python test_v2_quick.py                              # Test with dummy data
    python test_v2_quick.py path/to/image.jpg            # Test with actual image
    python test_v2_quick.py path/to/crops/folder         # Batch test directory
"""

import sys
import numpy as np
from pathlib import Path
import torch
from PIL import Image

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from deepfake_detector_v2 import load_v2_model


def create_dummy_image(size=256):
    """Create a realistic dummy image for testing"""
    # Create noise-based pseudo-face
    np.random.seed(42)
    img_array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    
    # Add some patterns to make it more realistic
    center_x, center_y = size // 2, size // 2
    radius = size // 4
    
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if dist < radius:
                img_array[i, j] = np.clip(img_array[i, j] + 50, 0, 255)
    
    return Image.fromarray(img_array)


def test_model_loading():
    """Test that model loads correctly"""
    print("="*60)
    print("TEST 1: Model Loading")
    print("="*60)
    
    model_path = Path('models/deepfake_detector_v2_ffa_mpdv.pth')
    
    if not model_path.exists():
        print(f"❌ FAILED: Model not found at {model_path}")
        return False
    
    try:
        detector = load_v2_model(str(model_path), device='cpu')
        print(f"✅ PASSED: Model loaded successfully")
        
        # Display model info
        info = detector.get_model_info()
        print(f"\nModel Information:")
        print(f"  Name: {info['name']}")
        print(f"  Device: {info['device']}")
        print(f"  Training epochs: {info['training_history_length']}")
        print(f"  Final metrics: {info['final_metrics']}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_single_inference(detector, image=None):
    """Test inference on a single image"""
    print("\n" + "="*60)
    print("TEST 2: Single Image Inference")
    print("="*60)
    
    try:
        if image is None:
            print("Creating dummy image...")
            image = create_dummy_image()
        else:
            print(f"Loading image: {image}")
            image = Image.open(image).convert('RGB')
        
        print(f"Image size: {image.size}")
        print("Running inference...")
        
        prediction = detector.predict(image, return_proba=True, threshold=0.5)
        
        print(f"\n✅ PASSED: Inference successful")
        print(f"Result:")
        print(f"  Label: {prediction['label_name']}")
        print(f"  Probability: {prediction['proba']:.4f}")
        print(f"  Confidence: {prediction['confidence']:.4f}")
        print(f"  Logit: {prediction['logit']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference(detector, image_list):
    """Test batch inference"""
    print("\n" + "="*60)
    print("TEST 3: Batch Inference")
    print("="*60)
    
    try:
        print(f"Processing {len(image_list)} images...")
        
        results = detector.predict_batch(image_list, return_proba=True)
        
        print(f"\n✅ PASSED: Batch inference successful")
        print(f"Results summary:")
        
        fake_count = sum(1 for r in results if r['label'] == 1)
        real_count = sum(1 for r in results if r['label'] == 0)
        mean_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"  FAKE: {fake_count}/{len(results)}")
        print(f"  REAL: {real_count}/{len(results)}")
        print(f"  Mean confidence: {mean_confidence:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_processing(detector, crop_dir):
    """Test processing a directory of crops"""
    print("\n" + "="*60)
    print("TEST 4: Directory Processing")
    print("="*60)
    
    crop_path = Path(crop_dir)
    
    if not crop_path.exists():
        print(f"❌ SKIPPED: Directory not found: {crop_dir}")
        return None
    
    try:
        from test_pipeline_v2 import DeepfakeDetectionPipelineV2
        
        print(f"Processing crops from: {crop_dir}")
        
        pipeline = DeepfakeDetectionPipelineV2(
            detector.model_path,
            device='cpu'
        )
        
        summary = pipeline.process_crops_directory(
            crop_dir,
            output_overlays_dir=None,
            output_results_json=None,
            verbose=False
        )
        
        print(f"\n✅ PASSED: Directory processing successful")
        print(f"Results:")
        print(f"  Total images: {summary['total_images']}")
        print(f"  Successful: {summary['success_count']}")
        print(f"  FAKE: {summary['fake_count']} ({summary['fake_percentage']:.1f}%)")
        print(f"  REAL: {summary['real_count']}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION v2 - QUICK TEST SUITE")
    print("="*60 + "\n")
    
    results = []
    
    # Test 1: Model loading
    success = test_model_loading()
    results.append(("Model Loading", success))
    
    if not success:
        print("\n⚠️  Model loading failed. Cannot continue with other tests.")
        return results
    
    # Load model for remaining tests
    model_path = Path('models/deepfake_detector_v2_ffa_mpdv.pth')
    detector = load_v2_model(str(model_path), device='cpu')
    
    # Test 2: Single image inference
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        # User provided an image file
        success = test_single_inference(detector, sys.argv[1])
    else:
        # Use dummy image
        success = test_single_inference(detector)
    results.append(("Single Inference", success))
    
    # Test 3: Batch inference (with 3 images)
    dummy_images = [create_dummy_image() for _ in range(3)]
    success = test_batch_inference(detector, dummy_images)
    results.append(("Batch Inference", success))
    
    # Test 4: Directory processing (if provided)
    if len(sys.argv) > 1:
        crop_dir = sys.argv[1]
        if Path(crop_dir).is_dir():
            success = test_directory_processing(detector, crop_dir)
            results.append(("Directory Processing", success))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✅ PASSED" if result is True else ("⚠️  SKIPPED" if result is None else "❌ FAILED")
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, r in results if r is True)
    total = len([r for r in results if r is not None])
    
    print(f"\nTotal: {passed}/{total} passed")
    print("="*60 + "\n")
    
    return results


if __name__ == '__main__':
    main()
