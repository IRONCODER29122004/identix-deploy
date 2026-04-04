import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2

# BiSeNet architecture (same as training)
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
        out = feat * atten
        return out

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
        feat2 = self.conv1(x)
        feat2 = self.bn1(feat2)
        feat2 = self.relu(feat2)
        feat4 = self.maxpool(feat2)
        feat8 = self.layer1(feat4)
        feat16 = self.layer2(feat8)
        feat32 = self.layer3(feat16)
        feat64 = self.layer4(feat32)
        avg = F.adaptive_avg_pool2d(feat64, 1)
        avg = self.conv_avg(avg)
        avg_up32 = F.interpolate(avg, size=feat64.size()[2:], mode='nearest')
        feat32_arm = self.arm32(feat64)
        feat32_sum = feat32_arm + avg_up32
        cp16 = self.conv_head32(feat32_sum)
        feat16_arm = self.arm16(feat32)
        cp16_up_to_16 = F.interpolate(cp16, size=feat32.size()[2:], mode='nearest')
        cp8 = self.conv_head16(feat16_arm + cp16_up_to_16)
        return feat2, feat8, cp8, cp16

class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 256, kernel_size=1, padding=0)
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

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
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = feat * atten
        feat_out = feat_atten + feat
        return feat_out

class BiSeNetOutput(nn.Module):
    def __init__(self, in_channels, mid_channels, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_channels, mid_channels, kernel_size=3)
        self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=False)
    def forward(self, x):
        feat = self.conv(x)
        out = self.conv_out(feat)
        return out

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
        out = self.conv_out(feat_fuse)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        if self.training:
            aux16 = self.conv_out16(cp8)
            aux16 = F.interpolate(aux16, size=(H, W), mode='bilinear', align_corners=True)
            aux32 = self.conv_out32(cp16)
            aux32 = F.interpolate(aux32, size=(H, W), mode='bilinear', align_corners=True)
            return out, aux16, aux32
        else:
            return out

# Load model
device = torch.device('cpu')
model = BiSeNet(n_classes=11).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
print("✓ Model loaded")

# Test on an image
test_img = list(Path('test/images').glob('*.jpg'))[0]
print(f"Testing on: {test_img}")

image = Image.open(test_img).convert('RGB')
original_size = image.size

# Transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(img_tensor)
    prediction = torch.argmax(output, dim=1)[0].cpu().numpy()

# Print class distribution
unique, counts = np.unique(prediction, return_counts=True)
total_pixels = prediction.size
print("\nPrediction class distribution:")
for cls, count in zip(unique, counts):
    pct = (count / total_pixels) * 100
    print(f"  Class {cls}: {count} pixels ({pct:.2f}%)")

print(f"\n⚠️ PROBLEM: Class 1 (skin) only has {counts[1] if 1 in unique else 0} pixels!")
print("The model is predicting mostly background instead of facial skin.")
