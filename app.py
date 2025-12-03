"""
Facial Landmark Generation Web Application
Flask server for landmark prediction using trained BiSeNet model
BiSeNet-only mode (no external landmark API)
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageOps
import numpy as np
import base64
from io import BytesIO
import os
import cv2
import tempfile
from collections import defaultdict, Counter
from deepfake_detector import DeepfakeDetector
from datetime import datetime
import hashlib
from functools import wraps
import secrets
import re
from mongodb_utils import get_db

# MediaPipe/third-party landmark detector removed by user request.
MEDIAPIPE_AVAILABLE = False
mediapipe_detector = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size for videos
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))  # Secret key for sessions

user_history = {}  # Store upload history per user

# Initialize MongoDB collections
try:
    db = get_db()
    auths_collection = db['auths']
    # Ensure unique email index
    auths_collection.create_index('email', unique=True)
except Exception as e:
    auths_collection = None
    print(f"⚠ MongoDB not available: {e}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Model Architecture ====================
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
        # Use ResNet-50 to match the trained model
        resnet = models.resnet50(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1   # 1/4
        self.layer2 = resnet.layer2   # 1/8
        self.layer3 = resnet.layer3   # 1/16
        self.layer4 = resnet.layer4   # 1/32
        
        # ResNet50 channel sizes: layer3=1024, layer4=2048
        self.arm16 = AttentionRefinementModule(1024, 256)
        self.arm32 = AttentionRefinementModule(2048, 256)
        self.conv_head32 = ConvBNReLU(256, 256, kernel_size=3)
        self.conv_head16 = ConvBNReLU(256, 256, kernel_size=3)
        self.conv_avg = ConvBNReLU(2048, 256, kernel_size=1, padding=0)
    
    def forward(self, x):
        # ResNet feature hierarchy
        feat2 = self.conv1(x)
        feat2 = self.bn1(feat2)
        feat2 = self.relu(feat2)
        feat4 = self.maxpool(feat2)
        feat8 = self.layer1(feat4)    # 1/4
        feat16 = self.layer2(feat8)   # 1/8
        feat32 = self.layer3(feat16)  # 1/16
        feat64 = self.layer4(feat32)  # 1/32
        
        # Global context
        avg = F.adaptive_avg_pool2d(feat64, 1)
        avg = self.conv_avg(avg)
        avg_up32 = F.interpolate(avg, size=feat64.size()[2:], mode='nearest')
        
        # Refine 1/32 path
        feat32_arm = self.arm32(feat64)
        feat32_sum = feat32_arm + avg_up32
        cp16 = self.conv_head32(feat32_sum)  # 1/32
        
        # Refine 1/16 path
        feat16_arm = self.arm16(feat32)
        cp16_up_to_16 = F.interpolate(cp16, size=feat32.size()[2:], mode='nearest')
        cp8 = self.conv_head16(feat16_arm + cp16_up_to_16)  # 1/16
        
        # Return features: low-level (1/2, 1/4) and refined cp8 (1/8), cp16 (1/16)
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
        # Ensure spatial sizes match before concatenation
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
        # Fuse at 1/8 resolution
        feat_fuse = self.ffm(feat_sp, cp8)
        
        out = self.conv_out(feat_fuse)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        
        if self.training:
            aux16 = self.conv_out16(cp8)    # 1/8
            aux16 = F.interpolate(aux16, size=(H, W), mode='bilinear', align_corners=True)
            aux32 = self.conv_out32(cp16)   # 1/16
            aux32 = F.interpolate(aux32, size=(H, W), mode='bilinear', align_corners=True)
            return out, aux16, aux32
        else:
            return out

# ==================== Load Model ====================
LAZY_LOAD = os.environ.get('LAZY_LOAD', 'true').lower() == 'true'
DEMO_MODE = os.environ.get('DEMO_MODE', 'false').lower() == 'true'

model = None
model_loaded = False

def _load_model_if_needed():
    global model, model_loaded
    if model is not None:
        return
    print("Loading BiSeNet model...")
    model = BiSeNet(n_classes=11)
    # Load trained weights - prefer 512x512 fine-tuned model
    model_paths = ['best_model_512.pth', 'best_model.pth']
    if DEMO_MODE:
        # Try a smaller checkpoint first if present in demo mode
        model_paths = ['best_model_demo.pth'] + model_paths
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                # Load with weights_only=True for security and memory efficiency
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict)
                print(f"✓ Model loaded from {model_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"⚠ Failed to load {model_path}: {e}")
    if not model_loaded:
        print("⚠ Warning: No model file found. Using untrained model.")
    model.eval()
    # Move to device after loading to minimize memory spike
    model = model.to(device)

# Load face detection model (Haar Cascade)
print("Loading face detection model...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✓ Face detection model loaded")

# Load ear detection for better landmark classification
print("Loading ear detection model...")
try:
    # Try to load ear cascade (may not be available in all OpenCV installations)
    ear_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_leftear.xml')
    if ear_cascade.empty():
        ear_cascade = None
        print("⚠ Ear detection not available, will use geometric heuristics")
    else:
        print("✓ Ear detection model loaded")
except:
    ear_cascade = None
    print("⚠ Ear detection not available, will use geometric heuristics")

# Initialize deepfake detector
print("Initializing deepfake detector...")
deepfake_detector = DeepfakeDetector()
print("✓ Deepfake detector initialized")

# MediaPipe/third-party detector removed by user request.
# mediapipe_detector remains None so code paths do not attempt to use it.
mediapipe_detector = None

# ==================== Image Preprocessing ====================
# IMPORTANT: Must match training image size (256x256)

def preprocess_image_advanced(image):
    """
    Advanced preprocessing for poor quality images
    Handles dark, bleached, low contrast, and noisy images
    
    Args:
        image: PIL Image
    Returns:
        Enhanced PIL Image
    """
    import cv2
    from PIL import ImageEnhance
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
    try:
        # Convert RGB to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        img_array = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    except Exception as e:
        print(f"CLAHE preprocessing failed: {e}")
    
    # 2. Denoise while preserving edges
    try:
        img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    except Exception:
        pass
    
    # Convert back to PIL
    enhanced_image = Image.fromarray(img_array)
    
    # 3. Auto-adjust brightness for very dark images
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(img_gray)
    
    if mean_brightness < 80:  # Very dark image
        enhancer = ImageEnhance.Brightness(enhanced_image)
        brightness_factor = 1.0 + (80 - mean_brightness) / 100.0
        enhanced_image = enhancer.enhance(min(brightness_factor, 1.8))
    elif mean_brightness > 200:  # Very bright/bleached image
        enhancer = ImageEnhance.Brightness(enhanced_image)
        brightness_factor = 200.0 / mean_brightness
        enhanced_image = enhancer.enhance(max(brightness_factor, 0.6))
    
    # 4. Enhance contrast for flat images
    enhancer = ImageEnhance.Contrast(enhanced_image)
    enhanced_image = enhancer.enhance(1.2)
    
    # 5. Slight sharpening for blurry images
    enhancer = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = enhancer.enhance(1.1)
    
    return enhanced_image

# Transform aligned to training configuration from new.ipynb (256 + ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def transform_with_preprocessing(image, use_advanced=False):
    """
    Transform image with optional advanced preprocessing
    
    Args:
        image: PIL Image
        use_advanced: Whether to apply advanced preprocessing (DISABLED by default)
    Returns:
        Transformed tensor
    """
    # DISABLED: Advanced preprocessing ruins model predictions
    # The model was trained on normal images, not preprocessed ones
    # if use_advanced:
    #     image = preprocess_image_advanced(image)
    return transform(image)

# ==================== Colormap for Visualization ====================
def create_label_colormap(custom_colors=None):
    """Creates a colormap for visualizing facial landmarks
    
    Args:
        custom_colors: Optional dict mapping class indices to RGB tuples
                      e.g., {1: [255, 0, 0], 2: [0, 255, 0]}
    """
    colormap = np.zeros((11, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]           # Background - Black
    # Use non-skin hue for skin to avoid using actual skin tones
    colormap[1] = [173, 216, 230]     # Skin - Light Blue
    colormap[2] = [101, 67, 33]       # Left eyebrow - Brown
    colormap[3] = [101, 67, 33]       # Right eyebrow - Brown
    colormap[4] = [255, 255, 255]     # Left eye - White
    colormap[5] = [255, 255, 255]     # Right eye - White
    colormap[6] = [200, 200, 255]     # Nose - Pale Lavender for smoother look
    colormap[7] = [220, 120, 120]     # Upper lip - Pink
    colormap[8] = [180, 80, 80]       # Inner mouth - Dark pink
    colormap[9] = [220, 120, 120]     # Lower lip - Pink
    colormap[10] = [70, 50, 30]       # Hair - Dark brown
    
    # Override with custom colors if provided
    if custom_colors:
        for idx, color in custom_colors.items():
            if 0 <= idx < 11:
                colormap[idx] = color
    
    return colormap

def label_to_color(label, custom_colors=None):
    """Convert label mask to RGB color image
    
    Args:
        label: Segmentation mask array
        custom_colors: Optional dict of custom colors for classes
    """
    colormap = create_label_colormap(custom_colors)
    h, w = label.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(11):
        colored[label == i] = colormap[i]
    return colored

# ==================== Optional Post-Processing (Toggle) ====================
# Enables mild, safe smoothing to produce cleaner masks without over-altering classes
SMOOTH_MASK = True
SMALL_COMPONENT_AREA_RATIO = 0.0005  # remove blobs < 0.05% of face-crop area
SKIN_CLOSE_KERNEL = 3                # kernel size for skin closing (3 is conservative)

# Edge refinement settings (NEW)
REFINE_EDGES = True                  # Enable boundary refinement for sharper landmark edges
BILATERAL_D = 9                      # Diameter for bilateral filter (preserves edges)
BILATERAL_SIGMA_COLOR = 50           # Color space sigma (LOWERED for sharper edges)
BILATERAL_SIGMA_SPACE = 50           # Coordinate space sigma (LOWERED for sharper edges)
BOUNDARY_KERNEL = 5                  # Morphological gradient kernel (INCREASED for wider edge zones)

# (Reverted) Ear suppression advanced flags from Changes 25–27 removed.
# Using strengthened ear heuristic from Change 24 only (width 20%, vertical band 15%-85%).


def _remove_small_components(label_map, min_area):
    """Remove tiny components for each non-background class by setting them to background."""
    h, w = label_map.shape
    out = label_map.copy()
    for cls_id in range(1, 11):  # skip background=0
        mask = (out == cls_id).astype(np.uint8)
        if mask.sum() == 0:
            continue
        num_labels, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            continue
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                out[comp == i] = 0  # set to background
    return out

def _fill_skin_holes(label_map, kernel_size=3):
    """Fill skin (class 1) holes and lightly close gaps without overwriting other landmarks."""
    out = label_map.copy()
    skin = (out == 1).astype(np.uint8)
    if skin.sum() == 0:
        return out
    # Fill holes by drawing filled contours
    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(skin)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    # Light morphological closing to remove tiny gaps
    ksize = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Only add filled skin where background currently exists to avoid overwriting other classes
    add_mask = (filled == 1) & (out == 0)
    out[add_mask] = 1
    return out

def _refine_edges_guided(pred_map, original_crop):
    """Refine landmark boundaries using guided filtering with the original face crop.
    Sharpens edges by aligning prediction boundaries with actual image edges.
    """
    # Convert prediction to multi-channel for processing
    h, w = pred_map.shape
    pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(11):
        pred_colored[pred_map == cls] = cls * 23  # spread classes across 0-255
    
    # Ensure original crop matches prediction size
    if original_crop.shape[:2] != (h, w):
        original_resized = cv2.resize(original_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        original_resized = original_crop
    
    # Apply bilateral filter to prediction guided by original image
    # This preserves edges present in the original image
    smoothed = cv2.bilateralFilter(
        pred_colored, 
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR, 
        sigmaSpace=BILATERAL_SIGMA_SPACE
    )
    
    # Convert back to class labels (take max channel value / 23)
    refined = (smoothed[:, :, 0] / 23).astype(np.uint8)
    
    # Detect boundaries in original image using Canny (balanced for clean edges)
    gray = cv2.cvtColor(original_resized, cv2.COLOR_RGB2GRAY)
    # Apply slight Gaussian blur first for cleaner edge detection
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, threshold1=40, threshold2=120)
    
    # Dilate edges slightly to create boundary zones
    kernel = np.ones((BOUNDARY_KERNEL, BOUNDARY_KERNEL), np.uint8)
    edge_zones = cv2.dilate(edges, kernel, iterations=1)
    
    # In edge zones, prefer the original prediction (sharper)
    # Outside edge zones, use bilateral-smoothed result
    out = refined.copy()
    out[edge_zones > 0] = pred_map[edge_zones > 0]
    
    return out

def _fix_ear_classification(pred_map, original_crop, face_bbox_in_crop=None):
    """(Reverted to Change 24 heuristic)
    Detect ear regions in side zones and reclassify them from skin (1) to background (0).
    Heuristic (Change 24):
      - Horizontal ear zones: outer 20% on each side
      - Vertical band: 15% to 85% of height
      - Contour area: 50 < area < 0.12 * (h*w)
      - Aspect ratio: ch / cw > 0.5 (ears taller than wide)
    """
    if original_crop is None or pred_map is None:
        return pred_map
    h, w = pred_map.shape
    out = pred_map.copy()
    ear_width = int(w * 0.20)
    ear_height_start = int(h * 0.15)
    ear_height_end = int(h * 0.85)
    left_ear_zone = (slice(ear_height_start, ear_height_end), slice(0, ear_width))
    right_ear_zone = (slice(ear_height_start, ear_height_end), slice(w - ear_width, w))
    for ear_zone in [left_ear_zone, right_ear_zone]:
        zone = out[ear_zone]
        zone_skin_mask = (zone == 1).astype(np.uint8)
        if zone_skin_mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(zone_skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < (h * w * 0.12):
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect_ratio = ch / (cw + 1e-6)
                if aspect_ratio > 0.5:
                    # Remove this contour region (likely ear)
                    contour_mask = np.zeros_like(zone_skin_mask)
                    cv2.drawContours(contour_mask, [contour], -1, 1, -1)
                    zone[contour_mask == 1] = 0
        out[ear_zone] = zone
    return out
    # (Removed ellipse and dynamic margin suppression logic in revert.)

def smooth_prediction_map(pred_map, original_crop=None, face_bbox=None):
    """Apply conservative smoothing steps to produce a cleaner mask.
    - Remove tiny speckles for all classes
    - Fill small holes and gaps in the skin region without touching other classes
    - Fix ear misclassification (ears → background)
    - Refine edges using guided filtering if original image provided
    """
    h, w = pred_map.shape
    min_area = max(30, int(SMALL_COMPONENT_AREA_RATIO * h * w))
    out = _remove_small_components(pred_map, min_area)
    out = _fill_skin_holes(out, kernel_size=SKIN_CLOSE_KERNEL)
    
    # Fix ear classification before edge refinement
    if original_crop is not None:
        out = _fix_ear_classification(out, original_crop, face_bbox)
    
    # Edge refinement (if original crop available and enabled)
    if REFINE_EDGES and original_crop is not None:
        out = _refine_edges_guided(out, original_crop)
    
    return out

# ==================== Enhanced Prediction Functions ====================
def predict_landmarks_mediapipe(image):
    """
    ENHANCED: Use MediaPipe for high-accuracy landmark detection (95%+)
    Falls back to BiSeNet if MediaPipe not available
    
    Args:
        image: PIL Image
    Returns:
        tuple: (prediction_mask, colored_viz, landmark_image, stats, method_used)
    """
    # Convert PIL to OpenCV format
    image_np = np.array(image)
    if len(image_np.shape) == 2:  # Grayscale
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:  # RGBA
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    elif image_np.shape[2] == 3:  # RGB
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_cv = image_np
    
    # Ensure model is loaded lazily
    if not model_loaded:
        _load_model_if_needed()
    # Try MediaPipe first
    if mediapipe_detector is not None:
        try:
            mask, vis_image, stats = mediapipe_detector.hybrid_prediction(image_cv)

            if mask is not None:
                # Convert visualization back to RGB
                vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

                # Create colored mask using internal colormap
                colored_mask = label_to_color(mask)

                # Add method info to stats (was MediaPipe previously)
                stats['method'] = 'External API (removed)'
                stats['accuracy'] = 'unknown'

                return mask, colored_mask, vis_image_rgb, stats, 'mediapipe'
        except Exception as e:
            print(f"MediaPipe failed, falling back to BiSeNet: {e}")
    
    # Fallback to BiSeNet
    return predict_landmarks_bisenet(image)


def predict_landmarks_bisenet(image):
    """
    BiSeNet facial landmark segmentation with FACE-FOCUSED approach
    
    Pipeline:
    1. Detect face using Haar Cascade (permissive settings)
    2. Draw bounding box around detected face
    3. Crop face region with 20% padding
    4. Resize crop to 256x256 (matches training input size)
    5. Apply transform: ToTensor → Normalize(ImageNet mean/std)
    6. Run BiSeNet model on 256x256 face crop
    7. Resize prediction back to original crop size using NEAREST interpolation
    8. Place segmented face back on full image (rest = background class 0)
    9. Create colored visualization for display
    
    Notes:
    - Aligning inference transform to training (256 + ImageNet norm) restores the previously perfect segmentation.
    - Face-only segmentation prevents mixing with background people.
    - NEAREST interpolation preserves class boundaries.
    
    Args:
        image: PIL Image (RGB)
    Returns:
        tuple: (prediction_mask, colored_viz, landmark_image, stats, method_used)
            - prediction_mask: Full-size mask with face segmented, rest background
            - colored_viz: Full-size colored visualization
            - landmark_image: Cropped face colored visualization
            - stats: Dict with method, accuracy, face info
            - method_used: 'bisenet'
    """
    original_size = image.size
    if not model_loaded:
        _load_model_if_needed()
    
    # STEP 1: Detect face using Haar Cascade
    faces = detect_faces(image)
    
    if len(faces) > 0:
        # Light filtering to reduce obvious false positives
        img_area = float(original_size[0] * original_size[1])
        filtered = []
        for (fx, fy, fw, fh) in faces:
            area_ratio = (fw * fh) / img_area
            aspect = fw / float(fh + 1e-6)
            if area_ratio < 0.0015:  # discard extremely tiny boxes
                continue
            if not (0.6 <= aspect <= 1.6):
                continue
            filtered.append((fx, fy, fw, fh))
        faces_sorted = sorted(filtered if filtered else faces, key=lambda b: b[2] * b[3], reverse=True)

        best = None
        best_score = -1.0
        best_stats = None

        def seg_score(pred_resized, bbox):
            # Score by plausibility: coverage within expected range, centrality, and presence of face features
            h_, w_ = pred_resized.shape
            area = float(h_ * w_)
            non_bg = (pred_resized > 0).sum() / area
            # Clamp raw coverage contribution
            cover = max(0.0, min(0.4, float(non_bg)))
            # Feature bonus: any of nose or mouth classes present
            has_feat = any([(pred_resized == k).any() for k in (6, 7, 8, 9)])
            feat_bonus = 0.03 if has_feat else 0.0
            # Centrality penalty (prefer center of image)
            x1,y1,x2,y2 = bbox
            bx = (x1 + x2) / 2.0
            by = (y1 + y2) / 2.0
            cx = original_size[0] / 2.0
            cy = original_size[1] / 2.0
            dx = abs(bx - cx) / max(1.0, original_size[0])
            dy = abs(by - cy) / max(1.0, original_size[1])
            center_penalty = 0.3 * (dx + dy)  # 0..~0.3
            return cover + feat_bonus - center_penalty

        for face_bbox in faces_sorted[:5]:
            x, y, w, h = face_bbox
            padding = int(max(w, h) * 0.4)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(original_size[0], x + w + padding)
            y2 = min(original_size[1], y + h + padding)
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = image.crop((x1, y1, x2, y2))
            face_crop_size = face_crop.size

            face_tensor = transform(face_crop).unsqueeze(0).to(device)
            if not model_loaded:
                _load_model_if_needed()
            with torch.no_grad():
                face_output = model(face_tensor)
                face_prediction = torch.argmax(face_output, dim=1)[0].cpu().numpy()

            face_prediction_resized = cv2.resize(
                face_prediction.astype(np.uint8), 
                face_crop_size,
                interpolation=cv2.INTER_NEAREST
            )
            # Gating: reject clearly implausible crops before scoring
            area_pix = float(face_prediction_resized.shape[0] * face_prediction_resized.shape[1])
            non_bg_ratio = (face_prediction_resized > 0).sum() / max(1.0, area_pix)
            # Expect some content but not overwhelming; skip tiny specks or oversized patches
            if non_bg_ratio < 0.02 or non_bg_ratio > 0.55:
                continue
            # Require some facial feature pixels (eyes/nose/mouth) to be present
            feature_px = 0
            for k in (4, 5, 6, 7, 8, 9):
                feature_px += int((face_prediction_resized == k).sum())
            if (feature_px / area_pix) < 0.003:
                continue
            if SMOOTH_MASK:
                face_crop_np = np.array(face_crop)
                face_prediction_resized = smooth_prediction_map(
                    face_prediction_resized,
                    original_crop=face_crop_np,
                    face_bbox=(x, y, w, h)
                )

            score = seg_score(face_prediction_resized, (x1, y1, x2, y2))

            if score > best_score:
                full_prediction = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
                full_prediction[y1:y2, x1:x2] = face_prediction_resized
                full_colored = label_to_color(full_prediction)
                face_colored = label_to_color(face_prediction_resized)
                best = (full_prediction, full_colored, face_colored)
                best_score = score
                best_stats = {
                    'method': 'BiSeNet (ResNet50) Face-Focused',
                    'accuracy': '91.6%',
                    'input_size': '256x256',
                    'num_faces': len(faces),
                    'face_bbox': (x1, y1, x2 - x1, y2 - y1),
                    'crop_size': face_crop_size,
                    'num_landmarks': 11,
                    'candidate_score': round(float(score), 4)
                }

            # Early accept if clearly plausible
            if best_score >= 0.06:
                break

        if best is not None:
            full_prediction, full_colored, face_colored = best
            vis_img = face_colored
            return full_prediction, full_colored, vis_img, best_stats, 'bisenet'

        # Fallback pass: retry with larger padding to catch missed hair/ears silhouettes
        for face_bbox in faces_sorted[:3]:
            x, y, w, h = face_bbox
            padding = int(max(w, h) * 0.6)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(original_size[0], x + w + padding)
            y2 = min(original_size[1], y + h + padding)
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = image.crop((x1, y1, x2, y2))
            face_crop_size = face_crop.size
            face_tensor = transform(face_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                face_output = model(face_tensor)
                face_prediction = torch.argmax(face_output, dim=1)[0].cpu().numpy()
            face_prediction_resized = cv2.resize(
                face_prediction.astype(np.uint8), face_crop_size, interpolation=cv2.INTER_NEAREST
            )
            area_pix = float(face_prediction_resized.shape[0] * face_prediction_resized.shape[1])
            non_bg_ratio = (face_prediction_resized > 0).sum() / max(1.0, area_pix)
            if non_bg_ratio < 0.02 or non_bg_ratio > 0.55:
                continue
            feature_px = 0
            for k in (4, 5, 6, 7, 8, 9):
                feature_px += int((face_prediction_resized == k).sum())
            if (feature_px / area_pix) < 0.003:
                continue
            if SMOOTH_MASK:
                face_crop_np = np.array(face_crop)
                face_prediction_resized = smooth_prediction_map(
                    face_prediction_resized, original_crop=face_crop_np, face_bbox=(x, y, w, h)
                )
            score = seg_score(face_prediction_resized, (x1, y1, x2, y2))
            if score > best_score:
                full_prediction = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
                full_prediction[y1:y2, x1:x2] = face_prediction_resized
                full_colored = label_to_color(full_prediction)
                face_colored = label_to_color(face_prediction_resized)
                best_stats = {
                    'method': 'BiSeNet (ResNet50) Face-Focused',
                    'accuracy': '91.6%',
                    'input_size': '256x256',
                    'num_faces': len(faces),
                    'face_bbox': (x1, y1, x2 - x1, y2 - y1),
                    'crop_size': face_crop_size,
                    'num_landmarks': 11,
                    'candidate_score': round(float(score), 4)
                }
                vis_img = face_colored
                return full_prediction, full_colored, vis_img, best_stats, 'bisenet'

        # Ultimate fallback: use largest candidate (no special scoring)
        if faces_sorted:
            x, y, w, h = faces_sorted[0]
            padding = int(max(w, h) * 0.4)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(original_size[0], x + w + padding)
            y2 = min(original_size[1], y + h + padding)
            face_crop = image.crop((x1, y1, x2, y2))
            face_tensor = transform(face_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                face_output = model(face_tensor)
                face_prediction = torch.argmax(face_output, dim=1)[0].cpu().numpy()
            face_prediction_resized = cv2.resize(
                face_prediction.astype(np.uint8), face_crop.size, interpolation=cv2.INTER_NEAREST
            )
            if SMOOTH_MASK:
                face_prediction_resized = smooth_prediction_map(
                    face_prediction_resized, original_crop=np.array(face_crop), face_bbox=(x, y, w, h)
                )
            full_prediction = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
            full_prediction[y1:y2, x1:x2] = face_prediction_resized
            full_colored = label_to_color(full_prediction)
            vis_img = label_to_color(face_prediction_resized)
            stats = {
                'method': 'BiSeNet (ResNet50) Face-Focused',
                'accuracy': '91.6%',
                'input_size': '256x256',
                'num_faces': len(faces),
                'face_bbox': (x1, y1, x2 - x1, y2 - y1),
                'crop_size': face_crop.size,
                'num_landmarks': 11,
                'candidate_score': 0.0
            }
            return full_prediction, full_colored, vis_img, stats, 'bisenet'
    
    else:
        # No face detected - fall back to whole-image segmentation
        print("⚠️ No face detected, processing whole image (not recommended)")
        img_tensor = transform(image).unsqueeze(0).to(device)
        if not model_loaded:
            _load_model_if_needed()
        
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
        
        # Resize back to original size using cv2 for consistency
        prediction_resized = cv2.resize(
            prediction.astype(np.uint8), 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Create colored visualization
        colored = label_to_color(prediction_resized)
        
        stats = {
            'method': 'BiSeNet (ResNet50) Full-Image',
            'accuracy': '91.6%',
            'input_size': '256x256',
            'num_faces': 0,
            'num_landmarks': 11
        }

        vis_img = colored

        return prediction_resized, colored, vis_img, stats, 'bisenet'


# Wrapper function for backward compatibility
def predict_landmarks(image):
    """
    Main prediction function - uses MediaPipe if available, else BiSeNet
    
    Args:
        image: PIL Image
    Returns:
        tuple: (prediction_mask, colored_viz, face_only_original, face_only_mask, face_bbox)
    """
    mask, colored, vis_img, stats, method = predict_landmarks_mediapipe(image)
    
    # For backward compatibility, extract face bbox if available
    face_bbox = None
    if 'faces' in stats and len(stats.get('faces', [])) > 0:
        # Extract first face bbox from MediaPipe
        face_bbox = (0, 0, image.size[0], image.size[1])  # Placeholder
    
    return mask, colored, None, None, face_bbox


# ==================== Video Processing Functions ====================
def is_valid_face_region(frame, bbox):
    """
    DEPRECATED: Overly strict validation was rejecting valid faces
    Kept for backward compatibility but always returns True
    """
    return True  # Trust Haar Cascade detection

def detect_faces(frame):
    """
    Detect faces using Haar Cascade with cascading fallback attempts.
    Uses multiple detection passes with different parameters for better reliability.
    Returns: List of (x, y, w, h) bounding boxes
    """
    gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization for better detection
    gray = cv2.equalizeHist(gray)
    
    # First attempt: balanced parameters
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # If no faces found, try more lenient settings
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
    # Final fallback: very lenient (may have false positives but better than missing faces)
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(15, 15),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
    # ULTRA aggressive last resort - catch ANY possible face
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.03,
            minNeighbors=1,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
    print(f"[DEBUG] Haar Cascade detected {len(faces)} face(s)")
    
    faces_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    
    if len(faces_list) > 0:
        print(f"[DEBUG] Returning {len(faces_list)} face(s): {faces_list}")
    else:
        print(f"[DEBUG] No faces detected by Haar Cascade")
    
    return faces_list



def get_face_id(bbox, existing_faces, iou_threshold=0.3):
    """
    Match face bounding box to existing tracked face or create new ID
    Uses IoU (Intersection over Union) for matching
    """
    x1, y1, w1, h1 = bbox
    
    best_iou = 0
    best_id = None
    
    for face_id, tracked_bbox in existing_faces.items():
        x2, y2, w2, h2 = tracked_bbox
        
        # Calculate IoU
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        
        if iou > best_iou:
            best_iou = iou
            best_id = face_id
    
    if best_iou >= iou_threshold:
        return best_id
    else:
        # New face
        return max(existing_faces.keys(), default=-1) + 1

def predict_landmarks_for_face(image, bbox):
    """
    Predict facial landmarks for a specific face bounding box (for video frames)
    Face-focused segmentation:
    1. Crop face with 20% padding
    2. Resize crop to 512x512 for model
    3. Predict landmarks
    4. Resize prediction back to original crop size
    
    Args:
        image: PIL Image (full frame)
        bbox: (x, y, w, h) face bounding box
    Returns:
        prediction mask, colored visualization, crop coordinates (x1,y1,x2,y2)
    """
    x, y, w, h = bbox
    
    # STEP 1: Add padding around face (40% of face size for full hair coverage)
    padding = int(max(w, h) * 0.4)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.size[0], x + w + padding)
    y2 = min(image.size[1], y + h + padding)
    
    # Crop face region from frame
    face_crop = image.crop((x1, y1, x2, y2))
    original_crop_size = face_crop.size
    
    # STEP 2: Transform to 512x512 and predict
    # Transform: Resize(512,512) → ToTensor → Normalize(ImageNet)
    img_tensor = transform(face_crop).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1)[0].cpu().numpy()
    
    # STEP 3: Resize prediction back to original crop size
    prediction_resized = cv2.resize(
        prediction.astype(np.uint8), 
        original_crop_size, 
        interpolation=cv2.INTER_NEAREST
    )
    # Optional smoothing with edge refinement and ear fix
    if SMOOTH_MASK:
        face_crop_np = np.array(face_crop)
        prediction_resized = smooth_prediction_map(
            prediction_resized, 
            original_crop=face_crop_np,
            face_bbox=bbox
        )
    
    # Create colored visualization
    colored = label_to_color(prediction_resized)
    
    return prediction_resized, colored, (x1, y1, x2, y2)

def extract_frames(video_path, max_frames=100):
    """
    Extract frames from video
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
    Returns:
        List of PIL Images
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    
    frames = []
    frame_numbers = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            frame_numbers.append(idx)
    
    cap.release()
    return frames, frame_numbers, fps, total_frames

def calculate_landmark_quality(prediction, landmark_id):
    """
    Calculate quality score for a specific landmark in a frame
    Returns normalized percentage score (0-100)
    Higher score = better quality
    Metrics: pixel count, compactness, centrality, coverage
    """
    mask = (prediction == landmark_id).astype(np.uint8)
    pixel_count = np.sum(mask)
    total_pixels = prediction.shape[0] * prediction.shape[1]
    
    if pixel_count == 0:
        return 0.0
    
    # 1. Coverage score (what % of image does this landmark occupy)
    coverage = pixel_count / total_pixels
    # Expected coverage ranges for different landmarks (rough estimates)
    expected_coverage = {
        1: 0.35,  # Skin - largest
        2: 0.03, 3: 0.03,  # Eyebrows
        4: 0.015, 5: 0.015,  # Eyes
        6: 0.05,  # Nose
        7: 0.02, 9: 0.02,  # Lips
        8: 0.01,  # Inner mouth
        10: 0.25  # Hair
    }
    expected = expected_coverage.get(landmark_id, 0.05)
    coverage_score = min(100, (coverage / expected) * 50)  # Max 50 points
    
    # 2. Find contours for compactness
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return coverage_score  # Return coverage score only
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # 3. Compactness score (circularity) - how well-formed is the shape
    compactness = (4 * np.pi * area / (perimeter * perimeter + 1e-6)) if perimeter > 0 else 0
    compactness_score = compactness * 30  # Max 30 points
    
    # 4. Fragmentation penalty - single blob is better than scattered pixels
    num_contours = len(contours)
    fragmentation_penalty = min(10, (num_contours - 1) * 2)  # -2 points per extra contour
    
    # 5. Size consistency - is the largest contour most of the pixels?
    largest_ratio = area / pixel_count if pixel_count > 0 else 0
    consistency_score = largest_ratio * 20  # Max 20 points
    
    # Combine scores (max 100)
    quality_score = coverage_score + compactness_score + consistency_score - fragmentation_penalty
    quality_score = max(0, min(100, quality_score))  # Clamp to 0-100
    
    return quality_score

def validate_face_detection(frame, bbox):
    """
    Validate a face detection by checking if segmentation produces valid landmarks.
    Filters false positives from Haar Cascade.
    
    Returns: (is_valid, quality_score)
    """
    try:
        # Quick segmentation test
        prediction, colored, crop_coords = predict_landmarks_for_face(frame, bbox)
        
        # Check if we have meaningful facial features (eyes, nose, mouth)
        feature_classes = {4, 5, 6, 7, 8, 9}  # eyes, nose, lips, mouth
        feature_pixels = sum(np.sum(prediction == cls) for cls in feature_classes)
        total_pixels = prediction.shape[0] * prediction.shape[1]
        feature_ratio = feature_pixels / total_pixels if total_pixels > 0 else 0
        
        # Check skin coverage
        skin_pixels = np.sum(prediction == 1)
        skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
        
        # Must have features (at least 0.5%) and reasonable skin (5-50%)
        is_valid = (feature_ratio >= 0.005 and 0.05 <= skin_ratio <= 0.5)
        
        # Quality score based on feature presence and skin coverage
        quality = feature_ratio * 100 + (skin_ratio * 50 if skin_ratio <= 0.4 else 0)
        
        return is_valid, quality
    except:
        return False, 0.0

def process_video_landmarks(video_path, max_frames=100):
    """
    NEW SIMPLE IMPLEMENTATION - From scratch for reliability
    
    Simple multi-person video segmentation:
    1. Detect faces per frame (Haar Cascade)
    2. Validate detections with segmentation quality
    3. Track persons across frames (simple IoU matching)
    4. Filter false positives (low screen time, poor quality)
    5. Identify main person (screen_time × quality)
    6. Select best frame per landmark + overall best frame
    
    Returns: (main_person_result, others_summary, processed_frames_count, total_frames)
    """
    frames, frame_numbers, fps, total_frames = extract_frames(video_path, max_frames)
    if not frames:
        return None, [], 0, total_frames

    print(f"\n{'='*70}")
    print(f"VIDEO SEGMENTATION - Processing {len(frames)} frames")
    print(f"{'='*70}\n")
    
    # Track faces across frames
    face_tracker = {}  # face_id -> last known bbox
    face_appearances = defaultdict(list)  # face_id -> list of (frame_idx, bbox, quality)
    
    print(f"Processing {len(frames)} frames from video...")
    print("Step 1: Detecting and validating faces...")
    
    # Detect and validate faces in all frames
    for frame_idx, frame in enumerate(frames):
        faces = detect_faces(frame)
        
        # Validate each detection before tracking
        for bbox in faces:
            is_valid, quality = validate_face_detection(frame, bbox)
            
            if is_valid:
                face_id = get_face_id(bbox, face_tracker)
                face_tracker[face_id] = bbox
                face_appearances[face_id].append((frame_idx, bbox, quality))
    
    # Filter faces: minimum 3 appearances AND average quality > threshold
    MIN_SCREEN_TIME = 3
    MIN_AVG_QUALITY = 1.0
    
    filtered_faces = {}
    for face_id, appearances in face_appearances.items():
        if len(appearances) >= MIN_SCREEN_TIME:
            avg_quality = sum(q for _, _, q in appearances) / len(appearances)
            if avg_quality >= MIN_AVG_QUALITY:
                filtered_faces[face_id] = appearances
    
    if not filtered_faces:
        print("No valid faces detected in video!")
        return None, [], len(frames), total_frames
    
    print(f"Validated {len(filtered_faces)} person(s) after filtering (removed {len(face_appearances) - len(filtered_faces)} false positives)")
    
    # Find main character (highest screen time × quality)
    face_scores = {}
    for face_id, appearances in filtered_faces.items():
        screen_time = len(appearances)
        avg_quality = sum(q for _, _, q in appearances) / len(appearances)
        face_scores[face_id] = screen_time * avg_quality
    
    main_face_id = max(face_scores.items(), key=lambda x: x[1])[0]
    main_screen_time = len(filtered_faces[main_face_id])
    
    print(f"Main character: Person {main_face_id + 1} with {main_screen_time} frames ({100*main_screen_time/len(frames):.1f}%)")
    
    # Process MAIN CHARACTER with detailed analysis
    print(f"Processing main character in detail...")
    
    person_predictions = []
    landmark_scores = defaultdict(list)
    overall_frame_scores = []  # Track overall quality of each frame
    
    # Process frames where main character appears
    for frame_idx, bbox, _ in filtered_faces[main_face_id]:
        frame = frames[frame_idx]
        
        # Get landmarks for this face
        prediction, colored, crop_coords = predict_landmarks_for_face(frame, bbox)
        person_predictions.append((prediction, colored, frame, crop_coords, frame_idx))
        
        # Calculate quality score for each landmark
        total_landmark_score = 0
        for landmark_id in range(1, 11):  # All 10 landmarks
            score = calculate_landmark_quality(prediction, landmark_id)
            landmark_scores[landmark_id].append((len(person_predictions) - 1, frame_numbers[frame_idx], score))
            total_landmark_score += score
        
        # Overall frame quality (average of all landmarks)
        avg_frame_quality = total_landmark_score / 10
        overall_frame_scores.append((len(person_predictions) - 1, frame_numbers[frame_idx], avg_frame_quality))
    
    # Find best frame for each landmark for main character
    # Find best frame for each landmark
    best_frames = {}
    for landmark_id in range(1, 11):
        scores = landmark_scores[landmark_id]
        if scores:
            best_idx, best_frame_num, best_score = max(scores, key=lambda x: x[2])
            pred_data = person_predictions[best_idx]
            
            best_frames[landmark_id] = {
                'frame_index': pred_data[4],
                'frame_number': best_frame_num,
                'score': float(best_score),
                'quality_score': float(best_score),
                'prediction': pred_data[0],
                'colored': pred_data[1],
                'image': pred_data[2],
                'crop_coords': pred_data[3]
            }
    
    # Find overall best frame (highest average quality across all landmarks)
    best_overall_idx, best_overall_frame_num, best_overall_score = max(overall_frame_scores, key=lambda x: x[2])
    best_overall_data = person_predictions[best_overall_idx]
    
    main_character_result = {
        'face_id': int(main_face_id),
        'screen_time': main_screen_time,
        'total_frames': len(frames),
        'best_frames': best_frames,
        'best_overall_frame': {
            'frame_index': best_overall_data[4],
            'frame_number': best_overall_frame_num,
            'quality_score': float(best_overall_score),
            'prediction': best_overall_data[0],
            'colored': best_overall_data[1],
            'image': best_overall_data[2],
            'crop_coords': best_overall_data[3]
        }
    }
    
    # Process OTHER FACES - only significant ones (>10% screen time of main character)
    MIN_OTHER_SCREEN_TIME_RATIO = 0.10
    other_faces = []
    
    for face_id, appearances in filtered_faces.items():
        if face_id == main_face_id:
            continue  # Skip main character
        
        # Filter: must have at least 10% of main character's screen time
        if len(appearances) < main_screen_time * MIN_OTHER_SCREEN_TIME_RATIO:
            continue
        
        # Get the best quality frame for this person
        best_appearance = max(appearances, key=lambda x: x[2])
        frame_idx, bbox, quality = best_appearance
        frame = frames[frame_idx]
        
        # Predict landmarks for this sample frame
        prediction, colored, crop_coords = predict_landmarks_for_face(frame, bbox)
        
        other_faces.append({
            'face_id': int(face_id),
            'screen_time': len(appearances),
            'screen_time_percent': 100 * len(appearances) / len(frames),
            'frame_number': frame_numbers[frame_idx],
            'image': frame,
            'colored': colored,
            'crop_coords': crop_coords
        })
    
    print(f"Showing {len(other_faces)} other person(s) (filtered by screen time threshold)")
    
    return main_character_result, other_faces, len(frames), total_frames

# ==================== User Authentication System ====================
def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            return jsonify({'error': 'Authentication required', 'redirect': '/login'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name', 'User').strip()

        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Basic email validation
        email = email.strip().lower()
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Password strength check
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        # Sanitize name (prevent XSS)
        name = re.sub(r'[<>\"\'&]', '', name)[:100]
        if not name:
            name = 'User'

        if auths_collection is None:
            return jsonify({'error': 'Database not ready'}), 500

        hashed = hashlib.sha256(password.encode()).hexdigest()
        doc = {
            'email': email,
            'password': hashed,
            'name': name,
            'created_at': datetime.utcnow()
        }
        try:
            auths_collection.insert_one(doc)
        except Exception as e:
            if 'E11000' in str(e):
                return jsonify({'error': 'User already exists'}), 409
            raise

        session['user_email'] = email
        session['user_name'] = name
        return jsonify({'success': True, 'message': 'Registration successful'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """Login user"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400

        if auths_collection is None:
            return jsonify({'error': 'Database not ready'}), 500

        hashed = hashlib.sha256(password.encode()).hexdigest()
        user = auths_collection.find_one({'email': email})
        if not user or user.get('password') != hashed:
            return jsonify({'error': 'Invalid credentials'}), 401

        session['user_email'] = email
        session['user_name'] = user.get('name', 'User')
        return jsonify({'success': True, 'message': 'Login successful', 'name': session['user_name']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logout', methods=['POST'])
def logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/check-auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if 'user_email' in session:
        return jsonify({
            'authenticated': True,
            'email': session['user_email'],
            'name': session['user_name']
        })
    return jsonify({'authenticated': False})

@app.route('/history', methods=['GET'])
@login_required
def get_history():
    """Get user's upload history"""
    try:
        user_email = session.get('user_email')
        history = user_history.get(user_email, [])
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== Routes ====================
@app.route('/')
def index():
    """Render landing page"""
    return render_template('index.html')

@app.route('/facial-landmarks')
def facial_landmarks():
    """Render dedicated image analysis page"""
    return render_template('image_analysis.html')

@app.route('/video-analysis')
def video_analysis():
    """Render dedicated video analysis page"""
    return render_template('video_analysis.html')

@app.route('/deepfake-detection')
def deepfake_detection():
    """Render dedicated deepfake detection page"""
    return render_template('deepfake_detection.html')

@app.route('/about')
def about():
    """Render about page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Render contact page"""
    return render_template('contact.html')

@app.route('/profile')
def profile():
    """Render user profile page"""
    return render_template('profile.html')

@app.route('/settings')
def settings():
    """Render settings page"""
    return render_template('settings.html')

@app.route('/blog')
def blog():
    """Render blog page"""
    return render_template('blog.html')

@app.route('/careers')
def careers():
    """Render careers page"""
    return render_template('careers.html')

@app.route('/privacy-policy')
def privacy_policy():
    """Render privacy policy page"""
    return render_template('privacy-policy.html')

@app.route('/terms-of-service')
def terms_of_service():
    """Render terms of service page"""
    return render_template('terms-of-service.html')

@app.route('/cookie-policy')
def cookie_policy():
    """Render cookie policy page"""
    return render_template('cookie-policy.html')

@app.route('/gdpr')
def gdpr():
    """Render GDPR page"""
    return render_template('gdpr.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image (respect EXIF orientation from phones)
        image = Image.open(file.stream)
        image = ImageOps.exif_transpose(image).convert('RGB')
        original_size = image.size
        
        # Use BiSeNet-only prediction (MediaPipe/API removed)
        mask, colored, vis_img, stats, method = predict_landmarks_bisenet(image)
        
        # Convert images to base64 for sending to frontend
        # Original image
        buffered_orig = BytesIO()
        image.save(buffered_orig, format="PNG")
        img_orig_str = base64.b64encode(buffered_orig.getvalue()).decode()
        
        # Colored prediction (full image with face mask)
        buffered_pred = BytesIO()
        Image.fromarray(colored).save(buffered_pred, format="PNG")
        img_pred_str = base64.b64encode(buffered_pred.getvalue()).decode()
        
    # Visualization image
        buffered_vis = BytesIO()
        Image.fromarray(vis_img).save(buffered_vis, format="PNG")
        img_vis_str = base64.b64encode(buffered_vis.getvalue()).decode()
        
        # Overlay (blend original with prediction)
        overlay = np.array(image)
        alpha = 0.6
        overlay = (alpha * overlay + (1 - alpha) * colored).astype(np.uint8)
        buffered_overlay = BytesIO()
        Image.fromarray(overlay).save(buffered_overlay, format="PNG")
        img_overlay_str = base64.b64encode(buffered_overlay.getvalue()).decode()
        
        # Raw mask (grayscale segmentation mask for recoloring)
        buffered_mask = BytesIO()
        Image.fromarray(mask.astype(np.uint8)).save(buffered_mask, format="PNG")
        img_mask_str = base64.b64encode(buffered_mask.getvalue()).decode()
        
        # Response with enhanced information
        response_data = {
            'success': True,
            'original': f'data:image/png;base64,{img_orig_str}',
            'prediction': f'data:image/png;base64,{img_pred_str}',
            'visualization': f'data:image/png;base64,{img_vis_str}',  # NEW: Landmark visualization
            'overlay': f'data:image/png;base64,{img_overlay_str}',
            'mask': f'data:image/png;base64,{img_mask_str}',  # Raw mask for recoloring
            'image_size': original_size,
            'face_detected': stats.get('num_faces', 0) > 0,
            'detection_method': stats.get('method', 'BiSeNet'),  # NEW: Show which method was used
            'accuracy': stats.get('accuracy', '91.6%'),  # NEW: Accuracy info
            'num_landmarks': stats.get('num_landmarks', 0),  # NEW: Number of landmarks detected
            'num_faces': stats.get('num_faces', 0)  # NEW: Number of faces
        }
        
        # Count landmarks from mask
        unique_labels = np.unique(mask)
        landmark_counts = {int(label): int(np.sum(mask == label)) for label in unique_labels if label > 0}
        
        # Landmark names mapping
        landmark_names = {
            1: "Skin",
            2: "Left Eyebrow",
            3: "Right Eyebrow",
            4: "Left Eye",
            5: "Right Eye",
            6: "Nose",
            7: "Upper Lip",
            8: "Inner Mouth",
            9: "Lower Lip",
            10: "Hair"
        }
        
        response_data['landmark_counts'] = landmark_counts
        response_data['landmark_names'] = landmark_names
        
        # Save to history if user is logged in
        if 'user_email' in session:
            user_email = session['user_email']
            if user_email not in user_history:
                user_history[user_email] = []
            
            history_item = {
                'file_type': 'image',
                'uploaded_at': datetime.utcnow().isoformat(),
                'timestamp': datetime.utcnow().isoformat(),
                'image_data': img_orig_str[:1000],  # Store small preview
                'size': len(img_orig_str),
                'face_detected': stats.get('num_faces', 0) > 0,
                'detection_method': stats.get('method', 'BiSeNet')
            }
            user_history[user_email].insert(0, history_item)
            # Keep only last 50 items
            user_history[user_email] = user_history[user_email][:50]
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recolor', methods=['POST'])
def recolor():
    """Recolor an existing segmentation mask with custom colors"""
    try:
        data = request.get_json()
        
        if not data or 'mask_data' not in data or 'original_image' not in data or 'colors' not in data:
            return jsonify({'error': 'Missing required data'}), 400
        
        # Decode the mask image (grayscale segmentation)
        mask_data = data['mask_data'].split(',')[1]  # Remove data:image/png;base64, prefix
        mask_bytes = base64.b64decode(mask_data)
        mask_image = Image.open(BytesIO(mask_bytes)).convert('L')
        mask = np.array(mask_image)
        
        # Decode the original image
        orig_data = data['original_image'].split(',')[1]
        orig_bytes = base64.b64decode(orig_data)
        original_image = Image.open(BytesIO(orig_bytes)).convert('RGB')
        original_array = np.array(original_image)
        
        # Parse custom colors (convert from JavaScript arrays to tuples)
        custom_colors = {}
        for key, value in data['colors'].items():
            custom_colors[int(key)] = value  # value is already a list [r, g, b]
        
        # Apply custom colormap to mask
        colored = label_to_color(mask, custom_colors=custom_colors)
        
        # Create overlay
        alpha = 0.6
        overlay = (alpha * original_array + (1 - alpha) * colored).astype(np.uint8)
        
        # Convert to base64
        buffered_pred = BytesIO()
        Image.fromarray(colored).save(buffered_pred, format="PNG")
        img_pred_str = base64.b64encode(buffered_pred.getvalue()).decode()
        
        buffered_overlay = BytesIO()
        Image.fromarray(overlay).save(buffered_overlay, format="PNG")
        img_overlay_str = base64.b64encode(buffered_overlay.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'prediction': f'data:image/png;base64,{img_pred_str}',
            'overlay': f'data:image/png;base64,{img_overlay_str}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Handle video upload and find best frames for main character + show other faces"""
    try:
        # Check if video was uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No video selected'}), 400
        
        # Get max_frames parameter (optional)
        max_frames = int(request.form.get('max_frames', 100))
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_path = tmp_file.name
            file.save(video_path)
        
        try:
            # Process video
            main_character, other_faces, processed_frames, total_frames = process_video_landmarks(video_path, max_frames)
            
            if main_character is None:
                return jsonify({'error': 'No faces detected in video'}), 400
            
            # Prepare response data
            landmark_names = {
                1: "Skin", 2: "Left Eyebrow", 3: "Right Eyebrow", 
                4: "Left Eye", 5: "Right Eye", 6: "Nose",
                7: "Upper Lip", 8: "Inner Mouth", 9: "Lower Lip", 10: "Hair"
            }
            
            # Process MAIN CHARACTER results
            main_landmarks = {}
            
            for landmark_id, data in main_character['best_frames'].items():
                # Get face crop from the frame
                x1, y1, x2, y2 = data['crop_coords']
                face_region = data['image'].crop((x1, y1, x2, y2))
                
                # Original face crop
                buffered_orig = BytesIO()
                face_region.save(buffered_orig, format="PNG")
                img_orig_str = base64.b64encode(buffered_orig.getvalue()).decode()
                
                # Colored prediction
                buffered_pred = BytesIO()
                Image.fromarray(data['colored']).save(buffered_pred, format="PNG")
                img_pred_str = base64.b64encode(buffered_pred.getvalue()).decode()
                
                # Overlay (blend face crop with prediction)
                overlay = np.array(face_region)
                # Resize colored to match face_region size if needed
                if overlay.shape[:2] != data['colored'].shape[:2]:
                    colored_resized = cv2.resize(data['colored'], (overlay.shape[1], overlay.shape[0]))
                else:
                    colored_resized = data['colored']
                
                alpha = 0.6
                overlay = (alpha * overlay + (1 - alpha) * colored_resized).astype(np.uint8)
                buffered_overlay = BytesIO()
                Image.fromarray(overlay).save(buffered_overlay, format="PNG")
                img_overlay_str = base64.b64encode(buffered_overlay.getvalue()).decode()
                
                # Create visualization showing only this landmark
                highlight_mask = np.zeros_like(data['colored'])
                landmark_pixels = data['prediction'] == landmark_id
                highlight_mask[landmark_pixels] = data['colored'][landmark_pixels]
                
                buffered_highlight = BytesIO()
                Image.fromarray(highlight_mask).save(buffered_highlight, format="PNG")
                img_highlight_str = base64.b64encode(buffered_highlight.getvalue()).decode()
                
                # Handle quality score gracefully (support legacy 'score')
                q_val = float(data.get('quality_score', data.get('score', 0.0)))
                
                main_landmarks[str(landmark_id)] = {
                    'landmark_name': landmark_names.get(landmark_id, f"Landmark {landmark_id}"),
                    'frame_number': int(data['frame_number']),
                    'quality_score': q_val,
                    'original': f'data:image/png;base64,{img_orig_str}',
                    'prediction': f'data:image/png;base64,{img_pred_str}',
                    'overlay': f'data:image/png;base64,{img_overlay_str}',
                    'highlight': f'data:image/png;base64,{img_highlight_str}'
                }
            
            # Process BEST OVERALL FRAME for main character
            best_overall = main_character['best_overall_frame']
            x1, y1, x2, y2 = best_overall['crop_coords']
            best_face_region = best_overall['image'].crop((x1, y1, x2, y2))
            
            # Original face crop
            buffered_best_orig = BytesIO()
            best_face_region.save(buffered_best_orig, format="PNG")
            best_img_orig_str = base64.b64encode(buffered_best_orig.getvalue()).decode()
            
            # Colored prediction
            buffered_best_pred = BytesIO()
            Image.fromarray(best_overall['colored']).save(buffered_best_pred, format="PNG")
            best_img_pred_str = base64.b64encode(buffered_best_pred.getvalue()).decode()
            
            # Overlay
            best_overlay = np.array(best_face_region)
            if best_overlay.shape[:2] != best_overall['colored'].shape[:2]:
                best_colored_resized = cv2.resize(best_overall['colored'], (best_overlay.shape[1], best_overlay.shape[0]))
            else:
                best_colored_resized = best_overall['colored']
            
            alpha = 0.6
            best_overlay = (alpha * best_overlay + (1 - alpha) * best_colored_resized).astype(np.uint8)
            buffered_best_overlay = BytesIO()
            Image.fromarray(best_overlay).save(buffered_best_overlay, format="PNG")
            best_img_overlay_str = base64.b64encode(buffered_best_overlay.getvalue()).decode()
            
            main_character_data = {
                'screen_time': int(main_character['screen_time']),
                'screen_time_percent': float(round(100 * main_character['screen_time'] / processed_frames, 1)),
                'landmarks': main_landmarks,
                'best_overall': {
                    'frame_number': int(best_overall['frame_number']),
                    'quality_score': float(best_overall['quality_score']),
                    'original': f'data:image/png;base64,{best_img_orig_str}',
                    'prediction': f'data:image/png;base64,{best_img_pred_str}',
                    'overlay': f'data:image/png;base64,{best_img_overlay_str}'
                }
            }
            
            # Process OTHER FACES - just simple visualization
            other_faces_data = []
            
            for face_data in other_faces:
                # Get face crop
                x1, y1, x2, y2 = face_data['crop_coords']
                face_region = face_data['image'].crop((x1, y1, x2, y2))
                
                # Original face
                buffered_orig = BytesIO()
                face_region.save(buffered_orig, format="PNG")
                img_orig_str = base64.b64encode(buffered_orig.getvalue()).decode()
                
                # Landmark overlay
                overlay = np.array(face_region)
                if overlay.shape[:2] != face_data['colored'].shape[:2]:
                    colored_resized = cv2.resize(face_data['colored'], (overlay.shape[1], overlay.shape[0]))
                else:
                    colored_resized = face_data['colored']
                
                alpha = 0.6
                overlay = (alpha * overlay + (1 - alpha) * colored_resized).astype(np.uint8)
                buffered_overlay = BytesIO()
                Image.fromarray(overlay).save(buffered_overlay, format="PNG")
                img_overlay_str = base64.b64encode(buffered_overlay.getvalue()).decode()
                
                other_faces_data.append({
                    'face_id': int(face_data['face_id']),
                    'screen_time': int(face_data['screen_time']),
                    'screen_time_percent': float(face_data['screen_time_percent']),
                    'frame_number': int(face_data['frame_number']),
                    'original': f'data:image/png;base64,{img_orig_str}',
                    'overlay': f'data:image/png;base64,{img_overlay_str}'
                })
            
            return jsonify({
                'success': True,
                'main_character': main_character_data,
                'other_faces': other_faces_data,
                'total_people': int(1 + len(other_faces_data)),
                'processed_frames': int(processed_frames),
                'total_frames': int(total_frames)
            })
        
        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/detect_deepfake', methods=['POST'])
def detect_deepfake():
    """Analyze video for deepfake detection"""
    try:
        # Check if video was uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No video selected'}), 400
        
        # Get max_frames parameter
        max_frames = int(request.form.get('max_frames', 100))
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_path = tmp_file.name
            file.save(video_path)
        
        try:
            # Extract frames
            frames, frame_numbers, fps, total_frames = extract_frames(video_path, max_frames)
            
            # Detect face in first frame to get main subject
            first_faces = detect_faces(frames[0])
            if len(first_faces) == 0:
                return jsonify({'error': 'No face detected in video'}), 400
            
            # Use the largest face as main subject
            main_bbox = max(first_faces, key=lambda b: b[2] * b[3])
            
            # Track this face across frames
            face_tracker = {0: main_bbox}
            tracked_bboxes = []
            predictions_sequence = []
            frames_with_face = []
            
            print(f"Analyzing {len(frames)} frames for deepfake detection...")
            
            for frame_idx, frame in enumerate(frames):
                faces = detect_faces(frame)
                
                if len(faces) > 0:
                    # Match to main face
                    face_id = get_face_id(faces[0], face_tracker)
                    if face_id == 0:
                        bbox = faces[0]
                    else:
                        bbox = main_bbox  # Use previous if no match
                    
                    face_tracker[0] = bbox
                else:
                    bbox = main_bbox  # Use previous bbox
                
                # Get landmarks for this face
                prediction, colored, crop_coords = predict_landmarks_for_face(frame, bbox)
                
                predictions_sequence.append(prediction)
                tracked_bboxes.append(bbox)
                frames_with_face.append(frame)
            
            # Run deepfake detection
            deepfake_report = deepfake_detector.detect_deepfake(
                predictions_sequence, 
                frames_with_face, 
                tracked_bboxes
            )
            
            # Add metadata
            deepfake_report['video_info'] = {
                'total_frames': int(total_frames),
                'analyzed_frames': len(predictions_sequence),
                'fps': float(fps) if fps else None
            }
            
            return jsonify({
                'success': True,
                'report': deepfake_report
            })
        
        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': os.path.exists('best_model.pth'),
        'device': str(device)
    })

# ==================== Main ====================
if __name__ == '__main__':
    print("="*70)
    print("Facial Landmark Generation Web Application - IDENTIX")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: BiSeNet with 11 landmark classes")
    print(f"Model Loaded: {model_loaded}")
    print(f"MongoDB Connected: {auths_collection is not None}")
    
    # Get port from environment (required for Render)
    port = int(os.environ.get('PORT', 5000))
    # Disable debug in production
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Server starting on port {port} (debug={debug_mode})")
    print("="*70)
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

