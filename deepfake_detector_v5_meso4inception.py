"""
Deepfake Detector v5 - DeepfakeBench Meso4Inception checkpoint.

Reference projects:
https://github.com/SCLBD/DeepfakeBench
https://github.com/HongguLiu/MesoNet-Pytorch
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class Meso4Inception(nn.Module):
    """Meso4Inception backbone adapted from DeepfakeBench mesonet implementation."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.inception1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
        self.inception1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.inception1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.inception1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.inception1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.inception1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
        self.inception1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.inception1_bn = nn.BatchNorm2d(11)

        self.inception2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.inception2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.inception2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.inception2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.inception2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.inception2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.inception2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.inception2_bn = nn.BatchNorm2d(12)

        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def _inception_layer1(self, x):
        x1 = self.inception1_conv1(x)
        x2 = self.inception1_conv2_2(self.inception1_conv2_1(x))
        x3 = self.inception1_conv3_2(self.inception1_conv3_1(x))
        x4 = self.inception1_conv4_2(self.inception1_conv4_1(x))
        y = torch.cat((x1, x2, x3, x4), dim=1)
        y = self.inception1_bn(y)
        y = self.maxpool1(y)
        return y

    def _inception_layer2(self, x):
        x1 = self.inception2_conv1(x)
        x2 = self.inception2_conv2_2(self.inception2_conv2_1(x))
        x3 = self.inception2_conv3_2(self.inception2_conv3_1(x))
        x4 = self.inception2_conv4_2(self.inception2_conv4_1(x))
        y = torch.cat((x1, x2, x3, x4), dim=1)
        y = self.inception2_bn(y)
        y = self.maxpool1(y)
        return y

    def forward(self, x):
        x = self._inception_layer1(x)
        x = self._inception_layer2(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool2(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeepfakeDetectorV5Meso4Inception:
    """Inference wrapper for DeepfakeBench Meso4Inception checkpoint."""

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        state = torch.load(self.model_path, map_location='cpu')
        if not isinstance(state, dict):
            raise RuntimeError('Unexpected checkpoint format for DeepfakeBench Meso4Inception model')

        self.model = Meso4Inception(num_classes=2).to(self.device)
        cleaned = {}
        for key, val in state.items():
            key = key.replace('module.', '')
            if key.startswith('backbone.'):
                key = key[len('backbone.') :]
            key = key.replace('Incption', 'inception')
            cleaned[key] = val

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading v5 model: {unexpected[:20]}")
        if missing:
            required = {'inception1_conv1.weight', 'conv1.weight', 'conv2.weight', 'fc2.weight'}
            if any(k in required for k in missing):
                raise RuntimeError(f"Missing critical keys while loading v5 model: {missing[:20]}")

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
            'model_name': 'DeepfakeBench Meso4Inception',
        }


def load_v5_model(model_path, device='cpu'):
    return DeepfakeDetectorV5Meso4Inception(model_path=model_path, device=device)
