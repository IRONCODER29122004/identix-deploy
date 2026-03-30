"""
Test ALL combinations to find what matches morning's accuracy
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== BiSeNet Model ====================
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
        atten = self.sigmoid(self.bn_atten(self.conv_atten(atten)))
        return feat * atten

class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.conv1, self.bn1, self.relu, self.maxpool = resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
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
        cp16 = self.conv_head32(feat32_arm + avg_up)
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

# Load model
model = BiSeNet(n_classes=11).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# ==================== Test Combinations ====================
test_image_path = 'test/images/10001868414_0.jpg'
image = Image.open(test_image_path).convert('RGB')

COLORMAP = np.array([[0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255], [128,0,128], [255,165,0], [0,128,128], [128,128,0]], dtype=np.uint8)

def test_combination(name, transform_fn):
    """Test a specific transform combination"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    img_tensor = transform_fn(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
    
    # Resize back
    prediction = cv2.resize(prediction.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)
    
    # Stats
    unique, counts = np.unique(prediction, return_counts=True)
    total = prediction.size
    print("Class distribution:")
    for cls, cnt in zip(unique, counts):
        pct = 100 * cnt / total
        if pct > 0.5:  # Only show classes > 0.5%
            print(f"  Class {cls}: {pct:.1f}%")
    
    # Color and save
    colored = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for cls_id in range(11):
        colored[prediction == cls_id] = COLORMAP[cls_id]
    
    output_path = f'combo_test_{name.replace(" ", "_").replace("/", "_")}.png'
    Image.fromarray(colored).save(output_path)
    print(f"Saved: {output_path}")
    
    return prediction

# ==================== COMBINATIONS TO TEST ====================

print("\n" + "="*70)
print("TESTING ALL COMBINATIONS")
print("="*70)

# 1. ImageNet normalization + 256
transform1 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
pred1 = test_combination("256 + ImageNet norm", transform1)

# 2. No normalization + 256
transform2 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
pred2 = test_combination("256 + No norm", transform2)

# 3. ImageNet normalization + 512
transform3 = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
pred3 = test_combination("512 + ImageNet norm", transform3)

# 4. No normalization + 512
transform4 = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
pred4 = test_combination("512 + No norm", transform4)

# 5. ImageNet normalization + 384
transform5 = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
pred5 = test_combination("384 + ImageNet norm", transform5)

# 6. Original size + ImageNet norm
def transform6_fn(img):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t(img.resize((256, 256)))  # Must resize for model
pred6 = test_combination("Original -> 256 + ImageNet", transform6_fn)

print("\n" + "="*70)
print("TESTING COMPLETE - Check output images")
print("="*70)
print("\nCompare the outputs with your morning results!")
print("The one that matches best is the correct combination.")
