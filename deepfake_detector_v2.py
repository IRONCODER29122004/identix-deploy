"""
Deepfake Detection Model v2 - FFA-MPDV (Paper-Aligned Baseline)

This module implements the trained FFA-MPDV deepfake detector from the notebook.
Combines Meso4 backbone + FPN fusion + Capsule routing + Spatial attention.
Optimized for paper-baseline reproducibility (MSE loss, Adam optimizer, no scheduler).

Architecture:
- Meso4 backbone: 4-layer lightweight CNN (3→8→8→16→16 channels)
- FPN fusion: Multi-scale feature pyramid with spatial attention  
- Capsule layer: 8 capsules with 16 dims each, 3 routing iterations
- Classifier: 2-layer MLP on concatenated features

Model weights loaded from trained checkpoint with full training metadata.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import warnings

# Optional: try to import timm for SegFormer support (disabled by default)
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU block"""
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Meso4Backbone(nn.Module):
    """4-layer Meso4 backbone for deepfake detection"""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(ConvBlock(3, 8, 3, 1), nn.MaxPool2d(2))
        self.c2 = nn.Sequential(ConvBlock(8, 8, 5, 2), nn.MaxPool2d(2))
        self.c3 = nn.Sequential(ConvBlock(8, 16, 5, 2), nn.MaxPool2d(2))
        self.c4 = nn.Sequential(ConvBlock(16, 16, 5, 2), nn.MaxPool2d(4))

    def forward(self, x):
        f1 = self.c1(x)
        f2 = self.c2(f1)
        f3 = self.c3(f2)
        f4 = self.c4(f3)
        return f2, f3, f4


class FPNFusion(nn.Module):
    """Feature Pyramid Network for multi-scale fusion"""
    def __init__(self, channels=(8, 16, 16), out_ch=32):
        super().__init__()
        self.l2 = nn.Conv2d(channels[0], out_ch, 1)
        self.l3 = nn.Conv2d(channels[1], out_ch, 1)
        self.l4 = nn.Conv2d(channels[2], out_ch, 1)
        self.smooth = ConvBlock(out_ch, out_ch, 3, 1)

    def forward(self, f2, f3, f4):
        p4 = self.l4(f4)
        p3 = self.l3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode='nearest')
        p2 = self.l2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode='nearest')
        return self.smooth(p2)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CapsuleLayer(nn.Module):
    """Capsule routing layer with squashing"""
    def __init__(self, in_dim, num_caps=8, cap_dim=16, routing_iters=3):
        super().__init__()
        self.num_caps = num_caps
        self.cap_dim = cap_dim
        self.routing_iters = routing_iters
        self.proj = nn.Linear(in_dim, num_caps * cap_dim)

    @staticmethod
    def squash(s, dim=-1, eps=1e-8):
        """Squashing function for capsule activations"""
        sq_norm = (s ** 2).sum(dim=dim, keepdim=True)
        scale = sq_norm / (1.0 + sq_norm)
        return scale * s / torch.sqrt(sq_norm + eps)

    def forward(self, x):
        u = self.proj(x).view(x.size(0), self.num_caps, self.cap_dim)
        b = torch.zeros(x.size(0), self.num_caps, 1, device=x.device)

        for _ in range(self.routing_iters):
            c = torch.softmax(b, dim=1)
            s = (c * u).sum(dim=1, keepdim=True)
            v = self.squash(s, dim=-1)
            b = b + (u * v).sum(dim=-1, keepdim=True)

        v = v.squeeze(1)
        return v


class SegformerFeatureExtractor(nn.Module):
    """Optional SegFormer feature extractor (disabled by default for v2)"""
    def __init__(self, model_name='mit_b0', pretrained=True, trainable=False, out_dim=64):
        super().__init__()
        self.using_timm = False
        self.fallback_dim = 32

        if TIMM_AVAILABLE:
            candidate_specs = []
            for name, family in [
                (model_name, 'segformer'),
                ('mit_b0', 'segformer'),
                ('mobilevitv2_100', 'alternate'),
                ('convnext_tiny', 'alternate'),
                ('efficientnet_b0', 'alternate'),
            ]:
                if name not in [n for n, _ in candidate_specs]:
                    candidate_specs.append((name, family))

            last_err = None
            for cand, family in candidate_specs:
                for use_pretrained in ([True, False] if pretrained else [False]):
                    try:
                        self.backbone = timm.create_model(
                            cand,
                            pretrained=use_pretrained,
                            num_classes=0,
                        )
                        self.using_timm = True
                        feat_dim = self.backbone.num_features
                        if family == 'segformer':
                            print(f'Using SegFormer/MiT backbone: {cand} (pretrained={use_pretrained})')
                        else:
                            print(f'Using alternate timm backbone: {cand} (pretrained={use_pretrained})')
                        break
                    except Exception as e:
                        last_err = e
                if self.using_timm:
                    break

            if not self.using_timm:
                warnings.warn(f'SegFormer init failed ({last_err}); using lightweight fallback branch.')
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                feat_dim = self.fallback_dim
        else:
            warnings.warn('timm not available; using lightweight fallback branch instead of SegFormer.')
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            feat_dim = self.fallback_dim

        if self.using_timm and not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        if self.using_timm:
            f = self.backbone(x)
        else:
            f = self.backbone(x).flatten(1)
        return self.proj(f)


class FFAMPDVNet(nn.Module):
    """Main FFA-MPDV network: Meso4 + FPN + Capsule + Spatial Attention + Optional SegFormer"""
    def __init__(self, use_segformer=False, segformer_variant='mit_b0', 
                 segformer_pretrained=True, segformer_trainable=False):
        super().__init__()
        
        self.backbone = Meso4Backbone()
        self.fpn = FPNFusion(channels=(8, 16, 16), out_ch=32)
        self.sattn = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.meso_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.caps = CapsuleLayer(in_dim=32, num_caps=8, cap_dim=16, routing_iters=3)

        self.segformer_branch = None
        segformer_dim = 0
        if use_segformer:
            self.segformer_branch = SegformerFeatureExtractor(
                model_name=segformer_variant,
                pretrained=segformer_pretrained,
                trainable=segformer_trainable,
                out_dim=64,
            )
            segformer_dim = 64

        # Classifier: combines Meso head (64) + Capsule (16) + [optional SegFormer (64)]
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16 + segformer_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        f2, f3, f4 = self.backbone(x)
        fused = self.fpn(f2, f3, f4)
        fused = self.sattn(fused)

        g = self.pool(fused).flatten(1)
        meso_feat = self.meso_head(g)
        cap_feat = self.caps(g)

        feats = [meso_feat, cap_feat]
        if self.segformer_branch is not None:
            seg_feat = self.segformer_branch(x)
            feats.append(seg_feat)

        feat = torch.cat(feats, dim=1)
        logit = self.classifier(feat).squeeze(1)
        return logit


class DeepfakeDetectorV2:
    """
    Wrapper for FFA-MPDV deepfake detection model v2.
    
    Handles model loading, preprocessing, and inference.
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            device: torch device ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.checkpoint = torch.load(self.model_path, map_location=self.device)
        self.config = self.checkpoint.get('config', {})
        self.model_name = self.checkpoint.get('model_name', 'Unknown')
        self.state_dict = self.checkpoint.get('state_dict', self.checkpoint)
        
        # Build and load model
        self._build_model()
        
    def _build_model(self):
        """Reconstruct model from checkpoint metadata"""
        use_segformer = self.config.get('use_segformer', False)
        segformer_variant = self.config.get('segformer_variant', 'mit_b0')
        segformer_pretrained = self.config.get('segformer_pretrained', True)
        segformer_trainable = self.config.get('segformer_trainable', False)
        
        self.model = FFAMPDVNet(
            use_segformer=use_segformer,
            segformer_variant=segformer_variant,
            segformer_pretrained=segformer_pretrained,
            segformer_trainable=segformer_trainable
        ).to(self.device)
        
        # Load weights
        try:
            self.model.load_state_dict(self.state_dict)
        except RuntimeError as e:
            # Try without strict mode if keys don't match exactly
            print(f"Strict loading failed: {e}. Attempting non-strict load...")
            self.model.load_state_dict(self.state_dict, strict=False)
        
        self.model.eval()
    
    def preprocess(self, image_input, img_size=256):
        """
        Preprocess image for model inference.
        
        Args:
            image_input: PIL Image, numpy array, or file path string
            img_size: Target image size (default 256x256)
            
        Returns:
            torch.Tensor: Preprocessed image (1, 3, H, W)
        """
        # Load image if path
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
        
        # Resize
        image = image.resize((img_size, img_size), Image.BILINEAR)
        
        # Convert to tensor and normalize
        img_array = np.array(image) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        # Paper normalization: mean=0.5, std=0.5
        img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image_input, return_proba=True, threshold=0.5):
        """
        Make prediction on a single image.
        
        Args:
            image_input: PIL Image, numpy array, or file path string
            return_proba: If True, return probability; if False, return binary label
            threshold: Decision threshold for binary classification
            
        Returns:
            dict: {
                'logit': raw model output,
                'proba': sigmoid(logit) if return_proba else None,
                'label': 0 (real) or 1 (fake),
                'confidence': absolute distance from 0.5
            }
        """
        with torch.no_grad():
            img_tensor = self.preprocess(image_input)
            logit = self.model(img_tensor).item()
            proba = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
            label = 1 if proba >= threshold else 0
            confidence = abs(proba - 0.5) * 2  # 0 to 1 scale
        
        return {
            'logit': logit,
            'proba': proba if return_proba else None,
            'label': label,
            'confidence': confidence,
            'label_name': 'FAKE' if label == 1 else 'REAL'
        }
    
    def predict_batch(self, image_list, return_proba=True, threshold=0.5):
        """
        Make predictions on multiple images.
        
        Args:
            image_list: List of PIL Images, file paths, or numpy arrays
            return_proba: If True, return probabilities
            threshold: Decision threshold
            
        Returns:
            list of dict: Predictions for each image
        """
        results = []
        for img in image_list:
            results.append(self.predict(img, return_proba=return_proba, threshold=threshold))
        return results
    
    def get_model_info(self):
        """Return model metadata"""
        return {
            'name': self.model_name,
            'config': self.config,
            'device': str(self.device),
            'final_metrics': self.checkpoint.get('final_metrics', {}),
            'training_history_length': len(self.checkpoint.get('history', [])),
            'notes': self.checkpoint.get('notes', 'No notes'),
        }


def load_v2_model(model_path, device='cpu'):
    """
    Convenience function to load a v2 model.
    
    Args:
        model_path: Path to trained model checkpoint
        device: torch device
        
    Returns:
        DeepfakeDetectorV2: Loaded model wrapper
    """
    return DeepfakeDetectorV2(model_path, device=device)
