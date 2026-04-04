"""
SegFormer B1 with Edge-Aware Refinement for Facial Landmark Segmentation

This model combines region-based and edge-based segmentation for precise facial landmark detection.
Architecture: SegFormer B1 backbone + Edge Detection Head + Refinement Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

# ==================== Model Architecture ====================

class SegformerEdgeAware(nn.Module):
    """
    SegFormer B1 with advanced edge-aware refinement for facial landmark segmentation.
    
    Architecture Overview:
    - SegFormer B1 backbone (18M params, pretrained on ADE20K)
    - Edge detection head (identifies boundaries between facial regions)
    - Refinement module (fuses region + edge information)
    
    Outputs:
    - region_logits: Raw 11-class segmentation predictions
    - edge_logits: Binary edge confidence map
    - refined_logits: Final predictions (USE THIS for inference)
    
    Expected Performance:
    - mI</s>: 0.80-0.85 (facial landmark accuracy)
    - Edge F1: 0.90+ (boundary detection quality)
    - Training input: 256×256 images
    """
    
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        
        # BACKBONE: Pretrained SegFormer B1
        if pretrained:
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                'nvidia/segformer-b1-finetuned-ade-512-512',
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = SegformerConfig(
                num_labels=num_classes,
                num_channels=3,
                image_size=256
            )
            self.segformer = SegformerForSemanticSegmentation(config)
        
        # EDGE DETECTION HEAD: Learns boundaries between facial regions
        self.edge_head = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        # REFINEMENT MODULE: Fuses region + edge for final predictions
        self.refinement = nn.Sequential(
            nn.Conv2d(num_classes * 2 + 1, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Forward pass: Image → Region predictions → Edge detection → Refinement
        
        Args:
            x: Input tensor (B, 3, 256, 256)
        
        Returns:
            region_logits: (B, 11, 256, 256) - Raw predictions
            edge_logits: (B, 1, 256, 256) - Edge map
            refined_logits: (B, 11, 256, 256) - Final output (USE THIS)
        """
        # Step 1: Extract region predictions from SegFormer backbone
        outputs = self.segformer(x)
        region_logits = outputs.logits
        
        # Step 2: Upscale if needed (match input size)
        if region_logits.shape[-2:] != x.shape[-2:]:
            region_logits = F.interpolate(
                region_logits,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Step 3: Detect edges from region predictions
        edge_logits = self.edge_head(region_logits)
        
        # Step 4: Create edge-weighted regions (amplify at boundaries)
        edge_prob = torch.sigmoid(edge_logits)
        edge_attention = edge_prob.repeat(1, region_logits.shape[1], 1, 1)
        edge_weighted_region = region_logits * (1.0 + 0.5 * edge_attention)
        
        # Step 5: Concatenate features for refinement
        # Combined: original regions (11) + edge-weighted regions (11) + edges (1) = 23 channels
        combined = torch.cat([region_logits, edge_weighted_region, edge_logits], dim=1)
        
        # Step 6: Refine predictions using deep fusion module
        refined_logits = self.refinement(combined)
        
        return region_logits, edge_logits, refined_logits


# ==================== Face Detection ====================

class FaceDetector:
    """Face detection using Haar Cascade for cropping before segmentation."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, image):
        """
        Detect faces in RGB image.
        
        Args:
            image: RGB numpy array
        
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def crop_face(self, image, face_bbox, padding=0.2):
        """
        Crop face region with padding.
        
        Args:
            image: RGB numpy array
            face_bbox: (x, y, w, h) tuple
            padding: Percentage of face size to add as padding
        
        Returns:
            face_crop: Cropped face region (RGB)
            bbox: Adjusted (x1, y1, x2, y2) coordinates
        """
        x, y, w, h = face_bbox
        h_img, w_img = image.shape[:2]
        
        # Add padding
        pad_h = int(h * padding)
        pad_w = int(w * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w_img, x + w + pad_w)
        y2 = min(h_img, y + h + pad_h)
        
        face_crop = image[y1:y2, x1:x2]
        
        return face_crop, (x1, y1, x2, y2)


# ==================== Inference Pipeline ====================

class SegFormerPipeline:
    """
    Complete pipeline for facial landmark segmentation using SegFormer model.
    
    Usage:
        pipeline = SegFormerPipeline(model_path='path/to/model.pth', device='cuda')
        result = pipeline.segment(image_path)
        # Returns: dict with 'original', 'face_crop', 'region_mask', 'edge_map', 'bbox'
    """
    
    def __init__(self, model_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.face_detector = FaceDetector()
        self.model = SegformerEdgeAware(num_classes=11)
        
        # Load trained weights if provided
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ SegFormer model loaded from: {model_path}")
        else:
            print("⚠️ No checkpoint loaded - using pretrained backbone only")
        
        self.model.to(self.device).eval()
        
        # Image preprocessing (ImageNet normalization for SegFormer)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def load_image(self, image_input):
        """Load image from path or numpy array."""
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Cannot load image: {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            image = image_input
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError("Input must be file path or numpy array")
        return image
    
    def segment(self, image_input):
        """
        Segment facial landmarks in image.
        
        Args:
            image_input: Image path (str) or numpy array (RGB)
        
        Returns:
            dict with keys:
                - 'original': Original image (RGB)
                - 'face_crop': Detected face crop
                - 'region_mask': Segmentation mask (H, W) with class indices 0-10
                - 'edge_map': Binary edge map (H, W)
                - 'bbox': Face bounding box (x1, y1, x2, y2)
        """
        # Load image
        original_image = self.load_image(image_input)
        h_orig, w_orig = original_image.shape[:2]
        
        # Detect face
        faces = self.face_detector.detect_faces(original_image)
        
        if len(faces) == 0:
            print("⚠️ No face detected, using full image")
            face_crop = original_image
            bbox = (0, 0, w_orig, h_orig)
        else:
            # Use largest face
            face_bbox = max(faces, key=lambda f: f[2] * f[3])
            face_crop, bbox = self.face_detector.crop_face(original_image, face_bbox)
        
        # Preprocess and predict
        input_tensor = self.transform(face_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, edge_logits, refined_logits = self.model(input_tensor)
        
        # Convert to numpy
        region_prob = F.softmax(refined_logits, dim=1)[0]
        region_mask = torch.argmax(region_prob, dim=0).cpu().numpy().astype(np.uint8)
        region_mask = np.clip(region_mask, 0, 10)
        
        edge_prob = torch.sigmoid(edge_logits)[0, 0].cpu().numpy()
        edge_map = (edge_prob > 0.5).astype(np.uint8)
        
        # Resize to original face crop size
        h_crop, w_crop = face_crop.shape[:2]
        region_mask = cv2.resize(region_mask, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)
        edge_map = cv2.resize(edge_map, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)
        
        return {
            'original': original_image,
            'face_crop': face_crop,
            'region_mask': region_mask,
            'edge_map': edge_map,
            'bbox': bbox
        }


# ==================== Utility Functions ====================

def create_colored_mask(mask, color_map=None):
    """
    Convert grayscale mask to colored visualization.
    
    Args:
        mask: (H, W) array with class indices 0-10 or 0-11
        color_map: Optional custom color map (11 or 12 colors)
    
    Returns:
        colored_mask: (H, W, 3) RGB array
    """
    if color_map is None:
        # Default LaPa-style colors (11 classes)
        color_map = np.array([
            [0, 0, 0],         # 0: Background
            [255, 224, 189],   # 1: Skin (peach)
            [139, 69, 19],     # 2: Left Eyebrow (brown)
            [101, 67, 33],     # 3: Right Eyebrow (dark brown)
            [30, 144, 255],    # 4: Left Eye (bright blue)
            [0, 191, 255],     # 5: Right Eye (cyan)
            [255, 160, 122],   # 6: Nose (light coral)
            [220, 20, 60],     # 7: Upper Lip (crimson)
            [178, 34, 34],     # 8: Lower Lip (firebrick)
            [255, 215, 0],     # 9: Left Ear (gold)
            [218, 165, 32],    # 10: Right Ear (goldenrod)
        ], dtype=np.uint8)
    
    # Ensure mask values are within range
    mask = np.clip(mask, 0, len(color_map) - 1)
    colored = color_map[mask]
    
    return colored


def overlay_mask_on_image(image, mask, alpha=0.5, color_map=None):
    """
    Create overlay of colored mask on original image.
    
    Args:
        image: Original RGB image (H, W, 3)
        mask: Segmentation mask (H, W) with class indices
        alpha: Transparency (0=only image, 1=only mask)
        color_map: Optional custom colors
    
    Returns:
        overlay: Blended RGB image
    """
    colored_mask = create_colored_mask(mask, color_map)
    
    # Resize if needed
    if image.shape[:2] != mask.shape:
        colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
    
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay


# ==================== Export ====================

__all__ = [
    'SegformerEdgeAware',
    'FaceDetector',
    'SegFormerPipeline',
    'create_colored_mask',
    'overlay_mask_on_image'
]
