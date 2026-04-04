import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_offline_models import CASES, normalize_verdict
from landmark_app import (
    extract_frames,
    find_main_face_bbox,
    detect_faces,
    get_crop_coords_for_model,
    get_deepfake_detector_v9,
)

OUT_JSON = Path(__file__).resolve().parent / "v9_tuning_results.json"


def _sample(values, count):
    if not values:
        return []
    if len(values) <= count:
        return list(values)
    idx = np.linspace(0, len(values) - 1, count, dtype=int)
    return [values[i] for i in idx]


def _aggregate(probs, count, w_mean, w_q75, w_q25):
    sampled = _sample(probs, count)
    if not sampled:
        return 0.5
    score_mean = float(np.mean(sampled))
    score_q75 = float(np.percentile(sampled, 75))
    score_q25 = float(np.percentile(sampled, 25))
    return float(np.clip(w_mean * score_mean + w_q75 * score_q75 + w_q25 * score_q25, 0.0, 1.0))


def _predict_video_probs(video_path, max_frames=100):
    frames, _, _, _ = extract_frames(str(video_path), max_frames)
    main_bbox, _, _ = find_main_face_bbox(frames, search_limit=12)
    if main_bbox is None:
        return []

    detector = get_deepfake_detector_v9()
    probs = []
    last_bbox = main_bbox

    for frame in frames:
        faces = detect_faces(frame)
        bbox = faces[0] if faces else last_bbox
        last_bbox = bbox

        coords = get_crop_coords_for_model(frame.size, bbox, model_type="bisenet")
        if not coords:
            continue

        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            continue

        face_crop = frame.crop((x1, y1, x2, y2))
        pred = detector.predict(face_crop, return_proba=True, threshold=0.5)
        probs.append(float(pred.get("proba", 0.0)))

    return probs


def _outcome(truth, pred):
    if pred in {"real", "fake"}:
        if truth == "real" and pred == "fake":
            return "FP"
        if truth == "fake" and pred == "real":
            return "FN"
        return "OK"
    return "unknown"


def main():
    cached = []
    for name, path, truth in CASES:
        if not path.exists():
            continue
        probs = _predict_video_probs(path)
        cached.append((name, truth, probs))

    if not cached:
        raise RuntimeError("No benchmark videos available for v9 tuning")

    sampled_counts = [16, 24, 32, 40]
    weight_sets = [
        (0.70, 0.20, 0.10),
        (0.60, 0.30, 0.10),
        (0.55, 0.35, 0.10),
        (0.50, 0.35, 0.15),
    ]
    deepfake_thresholds = [round(v, 2) for v in np.arange(0.58, 0.76, 0.02)]
    authentic_thresholds = [round(v, 2) for v in np.arange(0.24, 0.42, 0.02)]

    best = None
    trials = []

    for sampled_count in sampled_counts:
        for w_mean, w_q75, w_q25 in weight_sets:
            for d_th in deepfake_thresholds:
                for a_th in authentic_thresholds:
                    if a_th >= d_th:
                        continue

                    rows = []
                    ok = fp = fn = unknown = 0
                    for name, truth, probs in cached:
                        score = _aggregate(probs, sampled_count, w_mean, w_q75, w_q25)
                        if score >= d_th:
                            verdict = "LIKELY DEEPFAKE"
                        elif score <= a_th:
                            verdict = "LIKELY AUTHENTIC"
                        else:
                            verdict = "LOW CONFIDENCE - NEEDS MANUAL REVIEW"

                        pred = normalize_verdict(verdict)
                        oc = _outcome(truth, pred)
                        ok += int(oc == "OK")
                        fp += int(oc == "FP")
                        fn += int(oc == "FN")
                        unknown += int(oc == "unknown")
                        rows.append(
                            {
                                "video": name,
                                "truth": truth,
                                "score": round(score, 6),
                                "verdict": verdict,
                                "prediction": pred,
                                "outcome": oc,
                            }
                        )

                    fitness = ok * 10.0 - fp * 6.0 - fn * 8.0 - unknown * 2.0
                    item = {
                        "sampled_count": sampled_count,
                        "weights": {"mean": w_mean, "q75": w_q75, "q25": w_q25},
                        "deepfake_threshold": d_th,
                        "authentic_threshold": a_th,
                        "fitness": round(fitness, 4),
                        "ok": ok,
                        "fp": fp,
                        "fn": fn,
                        "unknown": unknown,
                        "rows": rows,
                    }
                    trials.append(item)

                    if best is None or item["fitness"] > best["fitness"]:
                        best = item

    top = sorted(trials, key=lambda x: x["fitness"], reverse=True)[:20]
    output = {
        "best": best,
        "top20": top,
        "trials": len(trials),
    }
    OUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("Best v9 config:")
    print(json.dumps(best, indent=2))
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
