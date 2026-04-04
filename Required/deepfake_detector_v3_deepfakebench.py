"""
Deepfake Detector v3 - DeepfakeBench Meso4 (pretrained release checkpoint).

Reference project:
https://github.com/SCLBD/DeepfakeBench
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class Meso4(nn.Module):
    """Meso4 backbone adapted from DeepfakeBench release architecture."""

    def __init__(self, num_classes=2, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool1(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeepfakeDetectorV3DeepfakeBench:
    """Inference wrapper for DeepfakeBench Meso4 checkpoint."""

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = Meso4(num_classes=2, in_channels=3).to(self.device)
        state = torch.load(self.model_path, map_location='cpu')

        if not isinstance(state, dict):
            raise RuntimeError('Unexpected checkpoint format for DeepfakeBench Meso4 model')

        cleaned = {}
        for key, val in state.items():
            key = key.replace('module.', '')
            if key.startswith('backbone.'):
                key = key[len('backbone.') :]
            cleaned[key] = val

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading v3 model: {unexpected}")
        if missing:
            # Allow small metadata mismatches but require core layers.
            required = {'conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'fc2.weight'}
            if any(k in required for k in missing):
                raise RuntimeError(f"Missing critical keys while loading v3 model: {missing}")

        self.model.eval()

    def preprocess(self, image_input, img_size=256):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')

        image = image.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        # DeepfakeBench detector configs use mean/std = 0.5.
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image_input, return_proba=True, threshold=0.5):
        with torch.no_grad():
            x = self.preprocess(image_input)
            logits = self.model(x)[0]
            probs = torch.softmax(logits, dim=0)

            fake_proba = float(probs[1].item())
            real_proba = float(probs[0].item())
            label = 1 if fake_proba >= threshold else 0
            confidence = abs(fake_proba - 0.5) * 2.0

        return {
            'logits': [float(logits[0].item()), float(logits[1].item())],
            'proba': fake_proba if return_proba else None,
            'real_proba': real_proba if return_proba else None,
            'label': label,
            'confidence': float(confidence),
            'label_name': 'FAKE' if label == 1 else 'REAL',
            'model_name': 'DeepfakeBench Meso4',
        }


def load_v3_model(model_path, device='cpu'):
    return DeepfakeDetectorV3DeepfakeBench(model_path=model_path, device=device)
