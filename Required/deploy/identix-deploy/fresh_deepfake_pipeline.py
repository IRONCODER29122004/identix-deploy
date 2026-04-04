import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
REQUIRED_ROOT = CURRENT_DIR.parents[1]
if str(REQUIRED_ROOT) not in sys.path:
    sys.path.insert(0, str(REQUIRED_ROOT))

try:
    from ml_deepfake_detector import XceptionDeepfakeDetector
except Exception:
    XceptionDeepfakeDetector = None


class FreshDeepfakePipeline:
    """Fresh deepfake pipeline isolated from the legacy detector stack."""

    def __init__(self):
        self.model_version = "Version 1 Identix Deepfake Detector"
        self.model_path = None
        self.model = None

        model_candidates = [
            Path(r"D:\link2\Capstone 4-1\Code_try_1\Required\models\premium_deepfakemodel.pth"),
            REQUIRED_ROOT / "models" / "premium_deepfakemodel.pth",
            CURRENT_DIR / "models" / "premium_deepfakemodel.pth",
        ]

        for candidate in model_candidates:
            if candidate.exists():
                self.model_path = str(candidate)
                break

        if XceptionDeepfakeDetector is not None and self.model_path:
            try:
                self.model = XceptionDeepfakeDetector(model_path=self.model_path)
            except Exception:
                self.model = None

    def _extract_face_crop(self, frame_bgr, bbox, pad_ratio=0.2):
        if frame_bgr is None or bbox is None:
            return None

        # Accept PIL frames from video sampling and normalize to OpenCV BGR arrays.
        if isinstance(frame_bgr, Image.Image):
            frame_rgb = np.array(frame_bgr.convert("RGB"))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        elif not isinstance(frame_bgr, np.ndarray):
            frame_bgr = np.array(frame_bgr)

        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

        x, y, w, h = bbox
        pad = int(max(w, h) * pad_ratio)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return crop

    def _heuristic_report(self, face_crops):
        if len(face_crops) < 2:
            return {
                "available": False,
                "model_version": self.model_version,
                "model_path": self.model_path,
                "confidence": 0.0,
                "verdict": "INSUFFICIENT_DATA",
                "reason": "Need at least two face crops to run heuristic fallback",
            }

        diffs = []
        prev_gray = None

        for crop in face_crops:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                diffs.append(float(np.mean(diff)))
            prev_gray = gray

        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        confidence = float(np.clip((mean_diff / 24.0) * 100.0, 0.0, 100.0))
        verdict = "LIKELY_DEEPFAKE" if confidence >= 55.0 else "LIKELY_AUTHENTIC"

        return {
            "available": False,
            "model_version": self.model_version,
            "model_path": self.model_path,
            "confidence": round(confidence, 2),
            "verdict": verdict,
            "reason": "ML model unavailable, used temporal consistency fallback",
            "mean_frame_diff": round(mean_diff, 4),
        }

    def analyze_video_frames(self, frames, tracked_bboxes):
        face_crops = []

        for frame, bbox in zip(frames, tracked_bboxes):
            crop = self._extract_face_crop(frame, bbox)
            if crop is not None:
                face_crops.append(crop)

        if not face_crops:
            return {
                "model_version": self.model_version,
                "model_path": self.model_path,
                "confidence": 0.0,
                "verdict": "NO_VALID_FACE_CROPS",
                "frames_used": 0,
            }

        if self.model is None:
            report = self._heuristic_report(face_crops)
            report["frames_used"] = len(face_crops)
            return report

        predictions = []
        for crop in face_crops:
            try:
                predictions.append(self.model.detect(crop))
            except Exception:
                continue

        if not predictions:
            report = self._heuristic_report(face_crops)
            report["frames_used"] = len(face_crops)
            return report

        deepfake_votes = sum(1 for item in predictions if item.get("class") == "DEEPFAKE")
        authentic_votes = sum(1 for item in predictions if item.get("class") == "AUTHENTIC")
        avg_conf = float(np.mean([float(item.get("confidence", 0.0)) for item in predictions]))

        verdict = "LIKELY_AUTHENTIC" if authentic_votes >= deepfake_votes else "LIKELY_DEEPFAKE"

        return {
            "available": True,
            "model_version": self.model_version,
            "model_path": self.model_path,
            "confidence": round(avg_conf, 2),
            "verdict": verdict,
            "frames_used": len(predictions),
            "authentic_votes": int(authentic_votes),
            "deepfake_votes": int(deepfake_votes),
        }

    def analyze_image(self, image_bgr, detect_faces_fn):
        if self.model is None:
            return None, "premium model not loaded"

        faces = detect_faces_fn(image_bgr)
        if len(faces) == 0:
            return None, "No face detected in image"

        best_bbox = max(faces, key=lambda box: box[2] * box[3])
        crop = self._extract_face_crop(image_bgr, best_bbox)
        if crop is None:
            return None, "Could not extract face crop"

        prediction = self.model.detect(crop)

        return {
            "model_version": self.model_version,
            "model_path": self.model_path,
            "faces_detected": int(len(faces)),
            "bbox": [int(v) for v in best_bbox],
            "prediction": prediction,
        }, None
