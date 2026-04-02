"""
Deepfake Detector v4 - SelimSeferbekov DFDC EfficientNet-B7 checkpoint.

Reference project:
https://github.com/selimsef/dfdc_deepfake_challenge
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image


class DeepFakeClassifierB7(nn.Module):
    """Classifier head used by selimsef's DFDC solution."""

    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.encoder = timm.create_model(
            'tf_efficientnet_b7.ns_jft_in1k',
            pretrained=False,
            drop_path_rate=0.2,
            num_classes=0,
            global_pool='',
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(2560, 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DeepfakeDetectorV4Selim:
    """Inference wrapper for one Selim B7 DFDC checkpoint."""

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        ckpt = torch.load(self.model_path, map_location='cpu')
        if not isinstance(ckpt, dict):
            raise RuntimeError('Unexpected checkpoint format for Selim v4 model')

        state_dict = ckpt.get('state_dict', ckpt)
        cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model = DeepFakeClassifierB7(dropout_rate=0.0).to(self.device)
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        allowed_unexpected = {'encoder.classifier.weight', 'encoder.classifier.bias'}
        hard_unexpected = [k for k in unexpected if k not in allowed_unexpected]
        if hard_unexpected:
            raise RuntimeError(f"Unexpected keys while loading v4 model: {hard_unexpected[:20]}")
        if missing:
            critical = {'encoder.conv_stem.weight', 'fc.weight', 'fc.bias'}
            if any(k in critical for k in missing):
                raise RuntimeError(f"Missing critical keys while loading v4 model: {missing[:20]}")

        self.model.eval()

    @staticmethod
    def _isotropically_resize_image(img, size):
        h, w = img.shape[:2]
        if max(w, h) == size:
            return img
        if w > h:
            scale = size / w
            h = h * scale
            w = size
        else:
            scale = size / h
            w = w * scale
            h = size
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        return cv2.resize(img, (int(w), int(h)), interpolation=interp)

    @staticmethod
    def _put_to_center(img, input_size):
        img = img[:input_size, :input_size]
        canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        start_w = (input_size - img.shape[1]) // 2
        start_h = (input_size - img.shape[0]) // 2
        canvas[start_h : start_h + img.shape[0], start_w : start_w + img.shape[1], :] = img
        return canvas

    def preprocess(self, image_input, input_size=380):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
            arr = np.asarray(image)
        elif isinstance(image_input, np.ndarray):
            arr = image_input
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
        else:
            arr = np.asarray(image_input.convert('RGB'))

        arr = self._isotropically_resize_image(arr, input_size)
        arr = self._put_to_center(arr, input_size)

        x = torch.tensor(arr, device=self.device).float() / 255.0
        x = x.permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        x = (x - mean) / std
        return x.unsqueeze(0)

    def predict(self, image_input, return_proba=True, threshold=0.5):
        with torch.no_grad():
            x = self.preprocess(image_input)
            logit = self.model(x).squeeze().float()
            proba = float(torch.sigmoid(logit).item())
            label = 1 if proba >= threshold else 0
            confidence = abs(proba - 0.5) * 2.0

        return {
            'logit': float(logit.item()),
            'proba': proba if return_proba else None,
            'label': label,
            'confidence': float(confidence),
            'label_name': 'FAKE' if label == 1 else 'REAL',
            'model_name': 'Selim DFDC EfficientNet-B7',
        }


def load_v4_model(model_path, device='cpu'):
    return DeepfakeDetectorV4Selim(model_path=model_path, device=device)
