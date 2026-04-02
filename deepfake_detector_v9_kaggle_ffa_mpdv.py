"""
Deepfake Detector v9 - Kaggle FFA-MPDV paper-baseline checkpoint.

This module intentionally uses a separate architecture path so existing v2-v8
code remains untouched.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from numpy_compat import ensure_numpy_pickle_compat

try:
    from deepfake_detector_v2 import SegformerFeatureExtractor
except Exception:
    SegformerFeatureExtractor = None


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FeaturePyramid(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv_align1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv_align2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_align3 = nn.Conv2d(256, 64, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.relu(self.bn2(self.conv2(c1)))
        c3 = self.relu(self.bn3(self.conv3(c2)))

        c2_up = F.interpolate(c2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        c3_up = F.interpolate(c3, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        return self.conv_align1(c1) + self.conv_align2(c2_up) + self.conv_align3(c3_up)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class Meso4ProfessorBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1a = ConvBNReLU(3, 8, k=3, stride=1, padding=1)
        self.b1b = ConvBNReLU(8, 8, k=3, stride=2, padding=1)

        self.b2a = ConvBNReLU(8, 16, k=3, stride=1, padding=1)
        self.b2b = ConvBNReLU(16, 16, k=3, stride=2, padding=1)

        self.b3a = ConvBNReLU(16, 32, k=3, stride=1, padding=1)
        self.b3b = ConvBNReLU(32, 32, k=3, stride=2, padding=1)

        self.b4a = ConvBNReLU(32, 64, k=3, stride=1, padding=1)
        self.b4b = ConvBNReLU(64, 64, k=3, stride=2, padding=1)

        self.fpn = FeaturePyramid(in_ch=64)
        self.sattn = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.b1a(x)
        x = self.b1b(x)
        x = self.b2a(x)
        x = self.b2b(x)
        x = self.b3a(x)
        x = self.b3b(x)
        x = self.b4a(x)
        x = self.b4b(x)
        x = self.fpn(x)
        x = self.sattn(x)
        return x


class CapsuleLayer(nn.Module):
    def __init__(self, input_num_capsules, input_dim_capsules, num_capsules=10, dim_capsules=16, routings=3):
        super().__init__()
        self.input_num_capsules = input_num_capsules
        self.input_dim_capsules = input_dim_capsules
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routings = routings

        self.W = nn.Parameter(
            torch.empty(input_num_capsules, num_capsules, input_dim_capsules, dim_capsules)
        )
        nn.init.xavier_uniform_(self.W)

    @staticmethod
    def squash(s, eps=1e-8):
        s_norm = torch.sum(s * s, dim=-1, keepdim=True)
        return (s_norm / (1.0 + s_norm)) * (s / torch.sqrt(s_norm + eps))

    def forward(self, inputs):
        if inputs.size(1) != self.input_num_capsules:
            raise ValueError(
                f"Capsule input token count mismatch: expected {self.input_num_capsules}, got {inputs.size(1)}"
            )

        inputs_hat = torch.einsum("bid,ijdf->bijf", inputs, self.W)

        b = torch.zeros(
            inputs_hat.size(0),
            self.input_num_capsules,
            self.num_capsules,
            device=inputs_hat.device,
            dtype=inputs_hat.dtype,
        )

        for i in range(self.routings):
            c = torch.softmax(b, dim=2)
            s = torch.sum(c.unsqueeze(-1) * inputs_hat, dim=1)
            v = self.squash(s)
            if i < self.routings - 1:
                b = b + torch.sum(inputs_hat * v.unsqueeze(1), dim=-1)

        return v


class FFAMPDVNetV9(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Meso4ProfessorBackbone()

        token_hw = int(cfg.img_size) // 16
        self.input_num_capsules = token_hw * token_hw

        self.caps = CapsuleLayer(
            input_num_capsules=self.input_num_capsules,
            input_dim_capsules=64,
            num_capsules=10,
            dim_capsules=16,
            routings=3,
        )

        self.segformer_branch = None
        segformer_dim = 0
        if bool(getattr(cfg, "use_segformer", False)):
            if SegformerFeatureExtractor is None:
                raise RuntimeError("SegFormer branch requested but SegformerFeatureExtractor is unavailable")
            self.segformer_branch = SegformerFeatureExtractor(
                model_name=getattr(cfg, "segformer_variant", "mit_b0"),
                pretrained=bool(getattr(cfg, "segformer_pretrained", True)),
                trainable=bool(getattr(cfg, "segformer_trainable", False)),
                out_dim=64,
            )
            segformer_dim = 64

        self.classifier = nn.Sequential(nn.Linear(10 * 16 + segformer_dim, 1))

    def forward(self, x):
        fm = self.backbone(x)
        tokens = fm.flatten(2).transpose(1, 2)
        cap_feat = self.caps(tokens).flatten(1)

        feats = [cap_feat]
        if self.segformer_branch is not None:
            feats.append(self.segformer_branch(x))

        feat = torch.cat(feats, dim=1)
        logit = self.classifier(feat).squeeze(1)
        return logit


class DeepfakeDetectorV9KaggleFFAMPDV:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        ensure_numpy_pickle_compat()
        payload = torch.load(self.model_path, map_location=self.device)
        if isinstance(payload, dict):
            self.config = payload.get("config", {})
            self.state_dict = payload.get("state_dict", payload.get("model_state_dict", payload))
            self.model_name = payload.get("model_name", "FFA-MPDV v9")
        else:
            self.config = {}
            self.state_dict = payload
            self.model_name = "FFA-MPDV v9"

        self.cfg = self._build_cfg(self.config)
        self.model = FFAMPDVNetV9(self.cfg).to(self.device)
        self.model.load_state_dict(self.state_dict, strict=False)
        self.model.eval()

    @staticmethod
    def _build_cfg(cfg_dict):
        return SimpleNamespace(
            img_size=int(cfg_dict.get("img_size", 256)),
            use_segformer=bool(cfg_dict.get("use_segformer", False)),
            segformer_variant=cfg_dict.get("segformer_variant", "mit_b0"),
            segformer_pretrained=bool(cfg_dict.get("segformer_pretrained", True)),
            segformer_trainable=bool(cfg_dict.get("segformer_trainable", False)),
        )

    @staticmethod
    def _as_pil(image_input):
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        return image_input.convert("RGB")

    def preprocess(self, image_input):
        image = self._as_pil(image_input)
        image = image.resize((self.cfg.img_size, self.cfg.img_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0).to(self.device)

    def _predict_proba_tta(self, image_input):
        image = self._as_pil(image_input)
        views = [image, ImageOps.mirror(image)]

        logits = []
        with torch.no_grad():
            for v in views:
                x = self.preprocess(v)
                logits.append(float(self.model(x).item()))

        mean_logit = float(np.mean(logits))
        proba = float(1.0 / (1.0 + np.exp(-mean_logit)))
        return proba, mean_logit, float(np.std(logits)), len(logits)

    def predict(self, image_input, return_proba=True, threshold=0.5):
        proba, logit, logit_std, views = self._predict_proba_tta(image_input)
        label = 1 if proba >= threshold else 0
        confidence = abs(proba - 0.5) * 2.0

        return {
            "logit": logit,
            "logit_std": logit_std,
            "tta_views": views,
            "proba": proba if return_proba else None,
            "label": label,
            "confidence": float(confidence),
            "label_name": "FAKE" if label == 1 else "REAL",
            "model_name": self.model_name,
        }


def load_v9_model(model_path, device="cpu"):
    return DeepfakeDetectorV9KaggleFFAMPDV(model_path=model_path, device=device)
