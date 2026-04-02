"""
Deepfake Detector v7/v8 - Kaggle EfficientNet-B0 binary checkpoints.

Primary source dataset:
https://www.kaggle.com/datasets/cyrinegraf/deepfake-efficientnet-trained-model
"""

from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image


class KaggleEfficientNetB0Binary(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extractor that matches common EfficientNet-B0 checkpoint naming.
        self.backbone = timm.create_model(
            "tf_efficientnet_b0_ns",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class DeepfakeDetectorV7KaggleEffNet:
    """Inference wrapper for Kaggle EfficientNet-B0 binary checkpoint."""

    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        payload = torch.load(self.model_path, map_location="cpu")
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise RuntimeError("Unexpected checkpoint format for Kaggle EfficientNet model")

        cleaned = {}
        for key, val in state_dict.items():
            key = key.replace("module.", "")
            cleaned[key] = val

        self.model = KaggleEfficientNetB0Binary().to(self.device)
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)

        # Fail only if critical heads are absent; allow minor buffer differences.
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading v7/v8 model: {unexpected[:20]}")
        required = {"backbone.conv_stem.weight", "classifier.1.weight", "classifier.4.weight"}
        if any(k in required for k in missing):
            raise RuntimeError(f"Missing critical keys while loading v7/v8 model: {missing[:20]}")

        self.model.eval()

    def preprocess(self, image_input, img_size=224):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        image = image.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        # ImageNet normalization expected by EfficientNet-B0 backbones.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image_input, return_proba=True, threshold=0.5):
        with torch.no_grad():
            x = self.preprocess(image_input)
            logits = self.model(x).squeeze().float()
            if logits.ndim == 0:
                logit_val = logits
            else:
                logit_val = logits[0]

            fake_proba = float(torch.sigmoid(logit_val).item())
            real_proba = float(1.0 - fake_proba)
            label = 1 if fake_proba >= threshold else 0
            confidence = abs(fake_proba - 0.5) * 2.0

        return {
            "logits": [float(logit_val.item())],
            "proba": fake_proba if return_proba else None,
            "real_proba": real_proba if return_proba else None,
            "label": label,
            "confidence": float(confidence),
            "label_name": "FAKE" if label == 1 else "REAL",
            "model_name": "Kaggle EfficientNet-B0",
        }


def load_v7_model(model_path, device="cpu"):
    return DeepfakeDetectorV7KaggleEffNet(model_path=model_path, device=device)


def load_v8_model(model_path, device="cpu"):
    return DeepfakeDetectorV7KaggleEffNet(model_path=model_path, device=device)
