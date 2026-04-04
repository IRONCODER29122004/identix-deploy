"""
Standalone v9 FFA-MPDV inference + export utility.

Additive-only helper script:
- Runs the v9 detector on a single image or video input
- Writes full JSON output
- Appends one benchmark row to CSV log
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from deepfake_detector_v9_kaggle_ffa_mpdv import load_v9_model


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CSV_FIELDS = [
    "timestamp_utc",
    "tag",
    "input_type",
    "input_path",
    "model_path",
    "device",
    "threshold",
    "max_frames",
    "sampled_frames",
    "label",
    "score",
    "proba_fake",
    "confidence",
    "mean_p",
    "q75",
    "q25",
    "logit",
    "logit_std",
    "tta_views",
]


def uniform_sample_indices(total, count):
    if total <= 0:
        return []
    if total <= count:
        return list(range(total))
    return np.linspace(0, total - 1, count, dtype=int).tolist()


def infer_image(detector, image_path, threshold):
    img = Image.open(image_path).convert("RGB")
    out = detector.predict(img, return_proba=True, threshold=threshold)

    return {
        "input_type": "image",
        "sampled_frames": 1,
        "label": out["label_name"],
        "score": float(out["proba"]),
        "proba_fake": float(out["proba"]),
        "confidence": float(out["confidence"]),
        "mean_p": None,
        "q75": None,
        "q25": None,
        "logit": float(out["logit"]),
        "logit_std": float(out["logit_std"]),
        "tta_views": int(out["tta_views"]),
        "raw": out,
    }


def infer_video(detector, video_path, threshold, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(uniform_sample_indices(total, max_frames))

    probs = []
    sampled = 0
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_i in idxs:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            out = detector.predict(pil, return_proba=True, threshold=threshold)
            probs.append(float(out["proba"]))
            sampled += 1

        frame_i += 1

    cap.release()

    if sampled == 0:
        raise RuntimeError("No frames sampled from video input")

    mean_p = float(np.mean(probs))
    q75 = float(np.percentile(probs, 75))
    q25 = float(np.percentile(probs, 25))
    score = float(np.clip(0.7 * mean_p + 0.2 * q75 + 0.1 * q25, 0.0, 1.0))

    return {
        "input_type": "video",
        "sampled_frames": sampled,
        "label": "FAKE" if score >= threshold else "REAL",
        "score": score,
        "proba_fake": None,
        "confidence": float(abs(score - 0.5) * 2.0),
        "mean_p": mean_p,
        "q75": q75,
        "q25": q25,
        "logit": None,
        "logit_std": None,
        "tta_views": None,
        "raw": {
            "video_probabilities": probs,
            "aggregation": "score = 0.7*mean + 0.2*q75 + 0.1*q25",
        },
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Run v9 detector and export JSON/CSV")
    parser.add_argument("--model", required=True, help="Path to v9 checkpoint (.pth)")
    parser.add_argument("--input", required=True, help="Path to image or video")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--threshold", type=float, default=0.66, help="Deepfake threshold")
    parser.add_argument("--max-frames", type=int, default=32, help="Frames sampled for video")
    parser.add_argument("--out-json", default="v9_last_result.json", help="Output JSON path")
    parser.add_argument("--log-csv", default="v9_benchmark_log.csv", help="Append CSV log path")
    parser.add_argument("--tag", default="manual_run", help="Tag to identify this run in CSV")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    input_path = Path(args.input).resolve()
    out_json = Path(args.out_json).resolve()
    log_csv = Path(args.log_csv).resolve()

    detector = load_v9_model(str(model_path), device=args.device)

    if input_path.suffix.lower() in VIDEO_EXTS:
        result = infer_video(detector, input_path, threshold=args.threshold, max_frames=args.max_frames)
    else:
        result = infer_image(detector, input_path, threshold=args.threshold)

    timestamp_utc = datetime.now(timezone.utc).isoformat()

    payload = {
        "timestamp_utc": timestamp_utc,
        "tag": args.tag,
        "input_path": str(input_path),
        "model_path": str(model_path),
        "device": args.device,
        "threshold": args.threshold,
        "max_frames": args.max_frames,
        "result": result,
    }
    write_json(out_json, payload)

    row = {
        "timestamp_utc": timestamp_utc,
        "tag": args.tag,
        "input_type": result["input_type"],
        "input_path": str(input_path),
        "model_path": str(model_path),
        "device": args.device,
        "threshold": args.threshold,
        "max_frames": args.max_frames,
        "sampled_frames": result["sampled_frames"],
        "label": result["label"],
        "score": result["score"],
        "proba_fake": result["proba_fake"],
        "confidence": result["confidence"],
        "mean_p": result["mean_p"],
        "q75": result["q75"],
        "q25": result["q25"],
        "logit": result["logit"],
        "logit_std": result["logit_std"],
        "tta_views": result["tta_views"],
    }
    append_csv(log_csv, row)

    print("=== V9 Export Complete ===")
    print(f"Input: {input_path}")
    print(f"Label: {result['label']}")
    print(f"Score: {result['score']:.6f}")
    print(f"JSON: {out_json}")
    print(f"CSV:  {log_csv}")


if __name__ == "__main__":
    main()
