"""
Deepfake Detector v6 - Selim DFDC EfficientNet-B7 ensemble.

Reference project:
https://github.com/selimsef/dfdc_deepfake_challenge
"""

from pathlib import Path

import numpy as np

from deepfake_detector_v4_selimsef import load_v4_model


class DeepfakeDetectorV6SelimEnsemble:
    """Averages probabilities across multiple Selim B7 checkpoints."""

    def __init__(self, model_paths, device='cpu'):
        if not model_paths:
            raise ValueError('No model paths provided for v6 ensemble')

        self.detectors = []
        self.model_paths = []
        for path in model_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                continue
            self.detectors.append(load_v4_model(str(path_obj), device=device))
            self.model_paths.append(str(path_obj))

        if not self.detectors:
            raise FileNotFoundError('No valid Selim checkpoints found for v6 ensemble')

    def predict(self, image_input, return_proba=True, threshold=0.5):
        probs = []
        confidences = []
        for det in self.detectors:
            pred = det.predict(image_input, return_proba=True, threshold=threshold)
            probs.append(float(pred.get('proba', 0.0)))
            confidences.append(float(pred.get('confidence', 0.0)))

        mean_proba = float(np.mean(probs))
        std_proba = float(np.std(probs)) if len(probs) > 1 else 0.0
        label = 1 if mean_proba >= threshold else 0
        confidence = float(max(0.0, min(1.0, (abs(mean_proba - 0.5) * 2.0) * (1.0 - min(0.35, std_proba)))))

        return {
            'proba': mean_proba if return_proba else None,
            'label': label,
            'confidence': confidence,
            'label_name': 'FAKE' if label == 1 else 'REAL',
            'model_name': 'Selim DFDC EfficientNet-B7 Ensemble',
            'member_count': len(self.detectors),
            'member_std': std_proba,
            'member_confidence_mean': float(np.mean(confidences)) if confidences else 0.0,
        }


def load_v6_model(model_paths, device='cpu'):
    return DeepfakeDetectorV6SelimEnsemble(model_paths=model_paths, device=device)
