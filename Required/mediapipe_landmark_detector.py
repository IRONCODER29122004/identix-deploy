"""
MediaPipe integration removed — safe stub module.

This module provides a minimal, import-safe stub so the rest of the project
can run in BiSeNet-only mode without raising ImportError or NameError when an
older import is present. It intentionally does not perform any landmark
detection.
"""

from typing import Tuple, Dict
import numpy as np


class MediaPipeEnhancedDetector:
    """Stub detector: methods return a clear 'removed' response."""

    def __init__(self, custom_model=None, device='cpu'):
        self.custom_model = custom_model
        self.device = device

    def hybrid_prediction(self, image: np.ndarray, use_model_refinement: bool = True) -> Tuple:
        """Return (None, None, stats) indicating API was removed."""
        return None, None, {'error': 'MediaPipe integration removed'}

    def close(self):
        return


def create_colored_mask(mask: np.ndarray, color_map: Dict[int, Tuple[int, int, int]] = None) -> np.ndarray:
    """Simple fallback to convert a label mask into an RGB image."""
    if color_map is None:
        color_map = {
            0: (0, 0, 0),
            1: (173, 216, 230),
            2: (101, 67, 33),
            3: (101, 67, 33),
            4: (255, 255, 255),
            5: (255, 255, 255),
            6: (200, 200, 255),
            7: (220, 120, 120),
            8: (180, 80, 80),
            9: (220, 120, 120),
            10: (70, 50, 30),
        }

    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, col in color_map.items():
        out[mask == k] = col
    return out
