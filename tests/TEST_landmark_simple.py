"""
Simple Facial Landmark Segmentation - Written from scratch for new.ipynb model
Matches training exactly: IMG_SIZE=256, ImageNet normalization, BiSeNet-ResNet50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import os

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== BiSeNet Model (from new.ipynb) ====================
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=3)
        self.conv_atten = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid(atten)
        return feat * atten

class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.arm16 = AttentionRefinementModule(1024, 256)
        self.arm32 = AttentionRefinementModule(2048, 256)
        self.conv_head32 = ConvBNReLU(256, 256, kernel_size=3)
        self.conv_head16 = ConvBNReLU(256, 256, kernel_size=3)
        self.conv_avg = ConvBNReLU(2048, 256, kernel_size=1, padding=0)
    def forward(self, x):
        feat2 = self.relu(self.bn1(self.conv1(x)))
        feat4 = self.maxpool(feat2)
        feat8 = self.layer1(feat4)
        feat16 = self.layer2(feat8)
        feat32 = self.layer3(feat16)
        feat64 = self.layer4(feat32)
        avg = self.conv_avg(F.adaptive_avg_pool2d(feat64, 1))
        avg_up = F.interpolate(avg, size=feat64.size()[2:], mode='nearest')
        feat32_arm = self.arm32(feat64)
        feat32_sum = feat32_arm + avg_up
        cp16 = self.conv_head32(feat32_sum)
        feat16_arm = self.arm16(feat32)
        cp16_up = F.interpolate(cp16, size=feat32.size()[2:], mode='nearest')
        cp8 = self.conv_head16(feat16_arm + cp16_up)
        return feat2, feat8, cp8, cp16

class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 256, kernel_size=1, padding=0)
    def forward(self, x):
        return self.conv_out(self.conv3(self.conv2(self.conv1(x))))

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, fsp, fcp):
        if fsp.size()[2:] != fcp.size()[2:]:
            fcp = F.interpolate(fcp, size=fsp.size()[2:], mode='bilinear', align_corners=True)
        feat = self.convblk(torch.cat([fsp, fcp], dim=1))
        atten = self.sigmoid(self.conv2(self.relu(self.conv1(F.adaptive_avg_pool2d(feat, 1)))))
        return feat * atten + feat

class BiSeNetOutput(nn.Module):
    def __init__(self, in_channels, mid_channels, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_channels, mid_channels, kernel_size=3)
        self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=False)
    def forward(self, x):
        return self.conv_out(self.conv(x))

class BiSeNet(nn.Module):
    def __init__(self, n_classes=11):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(512, 512)
        self.conv_out = BiSeNetOutput(512, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(256, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(256, 64, n_classes)
    def forward(self, x):
        H, W = x.size()[2:]
        feat_res2, feat_res8, cp8, cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, cp8)
        out = F.interpolate(self.conv_out(feat_fuse), size=(H, W), mode='bilinear', align_corners=True)
        if self.training:
            aux16 = F.interpolate(self.conv_out16(cp8), size=(H, W), mode='bilinear', align_corners=True)
            aux32 = F.interpolate(self.conv_out32(cp16), size=(H, W), mode='bilinear', align_corners=True)
            return out, aux16, aux32
        return out

# ==================== Load Model ====================
print("Loading BiSeNet model...")
model = BiSeNet(n_classes=11).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
print("[OK] Model loaded successfully")

# ==================== Transform (EXACTLY from new.ipynb) ====================
# From new.ipynb line 178-187: Resize(256,256), ToTensor, Normalize with ImageNet
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== Face Detection ====================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    """Simple Haar Cascade face detection"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces

# ==================== Color Mapping ====================
COLORMAP = np.array([
    [0, 0, 0],        # 0 background
    [255, 0, 0],      # 1 skin
    [0, 255, 0],      # 2 nose
    [0, 0, 255],      # 3 glasses
    [255, 255, 0],    # 4 eyes
    [255, 0, 255],    # 5 brows
    [0, 255, 255],    # 6 ears
    [128, 0, 128],    # 7 mouth
    [255, 165, 0],    # 8 upper_lip
    [0, 128, 128],    # 9 lower_lip
    [128, 128, 0],    # 10 hair
], dtype=np.uint8)

def label_to_color(mask):
    """Convert class labels to RGB colors"""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(11):
        colored[mask == cls_id] = COLORMAP[cls_id]
    return colored

# ==================== Prediction Functions ====================
def predict_image(image_path):
    """Predict landmarks for a single image"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Detect face
    faces = detect_faces(image)
    
    if len(faces) > 0:
        # Use largest face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        
        # Crop with 20% padding
        padding = int(max(w, h) * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(original_size[0], x + w + padding)
        y2 = min(original_size[1], y + h + padding)
        
        face_crop = image.crop((x1, y1, x2, y2))
        crop_size = face_crop.size
        
        # Transform and predict
        img_tensor = transform(face_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
        
        # Resize back to crop size
        prediction = cv2.resize(prediction.astype(np.uint8), crop_size, interpolation=cv2.INTER_NEAREST)
        
        # Create full-size mask
        full_mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = prediction
        
        colored = label_to_color(full_mask)
        return full_mask, colored
    else:
        # No face detected - process whole image
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
        
        prediction = cv2.resize(prediction.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
        colored = label_to_color(prediction)
        return prediction, colored

# ==================== Test ====================
if __name__ == '__main__':
    # Test on a sample image
    test_images = ['test/images/' + f for f in os.listdir('test/images')[:3] if f.endswith(('.jpg', '.png'))]
    
    for img_path in test_images:
        print(f"\nProcessing: {img_path}")
        mask, colored = predict_image(img_path)
        
        # Show stats
        unique, counts = np.unique(mask, return_counts=True)
        print("Class distribution:")
        for cls, cnt in zip(unique, counts):
            pct = 100 * cnt / mask.size
            print(f"  Class {cls}: {pct:.1f}%")
        
        # Save result
        output_path = img_path.replace('test/images/', 'test_output_').replace('.jpg', '_result.png')
        Image.fromarray(colored).save(output_path)
        print(f"[OK] Saved: {output_path}")
