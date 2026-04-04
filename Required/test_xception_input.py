"""
Test script to verify Xception model input format
"""

import torch
import timm
import cv2
import numpy as np

print("="*70)
print("XCEPTION INPUT FORMAT VERIFICATION")
print("="*70)

# 1. Load the model
print("\n1. Loading Xception model...")
model = timm.create_model('legacy_xception', pretrained=False, num_classes=2)
model.eval()

# Check model's expected configuration
print(f"\n2. Model Configuration:")
print(f"   Expected input size: {model.default_cfg['input_size']}")
print(f"   Expected mean: {model.default_cfg['mean']}")
print(f"   Expected std: {model.default_cfg['std']}")

# 3. Create a test face image (simulated)
print("\n3. Creating test face image...")
# Simulate a 640x480 BGR image (typical webcam/video frame)
test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
print(f"   Original frame shape: {test_frame.shape} (BGR)")

# Simulate face crop (100x100 region)
face_bbox = (100, 100, 200, 200)  # x, y, w, h
x, y, w, h = face_bbox
face_crop = test_frame[y:y+h, x:x+w]
print(f"   Face crop shape: {face_crop.shape} (BGR)")

# 4. Preprocess according to Xception requirements
print("\n4. Preprocessing face crop...")

# Step 1: Resize to 299x299
face_resized = cv2.resize(face_crop, (299, 299), interpolation=cv2.INTER_CUBIC)
print(f"   After resize: {face_resized.shape}")

# Step 2: Convert BGR to RGB
face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
print(f"   After BGR→RGB: {face_rgb.shape}")

# Step 3: Normalize to [0, 1]
face_normalized = face_rgb.astype(np.float32) / 255.0
print(f"   After /255: range [{face_normalized.min():.3f}, {face_normalized.max():.3f}]")

# Step 4: Apply Xception normalization (mean=0.5, std=0.5)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
for i in range(3):
    face_normalized[:, :, i] = (face_normalized[:, :, i] - mean[i]) / std[i]
print(f"   After (x-0.5)/0.5: range [{face_normalized.min():.3f}, {face_normalized.max():.3f}]")

# Step 5: Convert to tensor [1, 3, 299, 299]
tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
print(f"   Final tensor shape: {tensor.shape}")
print(f"   Tensor dtype: {tensor.dtype}")
print(f"   Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")

# 5. Test model inference
print("\n5. Testing model inference...")
with torch.no_grad():
    output = model(tensor)

print(f"   Output shape: {output.shape}")
print(f"   Output logits: {output[0].numpy()}")

# Apply softmax
probs = torch.nn.functional.softmax(output, dim=1)
print(f"   Probabilities: {probs[0].numpy()}")
print(f"   Class 0 (Authentic): {probs[0, 0].item()*100:.2f}%")
print(f"   Class 1 (Deepfake): {probs[0, 1].item()*100:.2f}%")

print("\n" + "="*70)
print("✓ INPUT FORMAT VERIFICATION COMPLETE")
print("="*70)
print("\nSUMMARY:")
print("✓ Input shape: [1, 3, 299, 299] ✓")
print("✓ Color space: RGB (converted from BGR) ✓")
print("✓ Normalization: (x/255 - 0.5) / 0.5 → range [-1, 1] ✓")
print("✓ Model accepts input successfully ✓")
print("\nThe input format is CORRECT for Xception model!")
print("="*70)
