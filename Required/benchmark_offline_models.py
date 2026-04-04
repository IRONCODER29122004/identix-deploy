import json
import argparse
import io
import contextlib
from pathlib import Path

from landmark_app import (
    extract_frames,
    find_main_face_bbox,
    detect_faces,
    get_crop_coords_for_model,
    predict_landmarks_for_face,
    deepfake_detector,
    get_deepfake_detector_v2,
    get_deepfake_detector_v3,
    get_deepfake_detector_v4,
    get_deepfake_detector_v5,
    get_deepfake_detector_v6,
    get_deepfake_detector_v7,
    get_deepfake_detector_v8,
    get_deepfake_detector_v9,
    summarize_v2_predictions,
    summarize_v3_deepfakebench,
    summarize_v4_selim,
    summarize_kaggle_binary,
    V2_FAKE_THRESHOLD,
    V3_V4_THRESHOLD,
    V5_DEEPFAKE_THRESHOLD,
    V6_DEEPFAKE_THRESHOLD,
    V7_DEEPFAKE_THRESHOLD,
    V7_AUTHENTIC_THRESHOLD,
    V8_DEEPFAKE_THRESHOLD,
    V8_AUTHENTIC_THRESHOLD,
    V9_DEEPFAKE_THRESHOLD,
    V9_AUTHENTIC_THRESHOLD,
    V9_SAMPLED_FRAMES,
)

ROOT = Path(__file__).resolve().parent

CASES = [
    ("speaking_person.mp4", ROOT / "faceswap_samples" / "speaking_person.mp4", "real"),
    ("test_person_video.mp4", ROOT / "test_videos" / "test_person_video.mp4", "real"),
    ("output_deepfake.mp4", ROOT / "output_deepfake.mp4", "fake"),
]

MODELS = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"]


def normalize_verdict(verdict: str):
    s = (verdict or "").lower()
    if "authentic" in s or "real" in s:
        return "real"
    if "deepfake" in s or "fake" in s:
        return "fake"
    if "manual review" in s:
        return "uncertain"
    return "unknown"


def run_case(video_path: Path, model: str, max_frames: int = 100):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        frames, _, fps, total_frames = extract_frames(str(video_path), max_frames)
        main_bbox, _, _ = find_main_face_bbox(frames, search_limit=12)
    if main_bbox is None:
        return {"error": "Could not initialize tracking region"}

    face_tracker = {0: main_bbox}
    tracked_bboxes = []
    frames_with_face = []

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for frame in frames:
            faces = detect_faces(frame)
            if len(faces) > 0:
                bbox = faces[0]
                face_tracker[0] = bbox
            else:
                bbox = main_bbox
            tracked_bboxes.append(bbox)
            frames_with_face.append(frame)

    if model == "v1":
        predictions_sequence = []
        for frame, bbox in zip(frames_with_face, tracked_bboxes):
            prediction, _, _ = predict_landmarks_for_face(frame, bbox, "bisenet")
            predictions_sequence.append(prediction)
        rep = deepfake_detector.detect_deepfake(predictions_sequence, frames_with_face, tracked_bboxes)
        verdict = rep.get("verdict", "UNKNOWN")
        confidence = rep.get("confidence")
    elif model == "v2":
        detector = get_deepfake_detector_v2()
        frame_predictions = []
        for frame, bbox in zip(frames_with_face, tracked_bboxes):
            coords = get_crop_coords_for_model(frame.size, bbox, model_type="bisenet")
            if not coords:
                continue
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame.crop((x1, y1, x2, y2))
            pred = detector.predict(face_crop, return_proba=True, threshold=V2_FAKE_THRESHOLD)
            frame_predictions.append(
                {
                    "label": pred.get("label_name", "REAL"),
                    "probability_fake": float(pred.get("proba", 0.0)),
                    "confidence": float(pred.get("confidence", 0.0)),
                }
            )
        summary = summarize_v2_predictions(frame_predictions)
        verdict = summary["verdict"]
        confidence = summary["confidence_pct"]
    elif model == "v3":
        detector = get_deepfake_detector_v3()
        frame_predictions = []
        for frame, bbox in zip(frames_with_face, tracked_bboxes):
            coords = get_crop_coords_for_model(frame.size, bbox, model_type="bisenet")
            if not coords:
                continue
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame.crop((x1, y1, x2, y2))
            pred = detector.predict(face_crop, return_proba=True, threshold=V3_V4_THRESHOLD)
            frame_predictions.append(
                {
                    "label": pred.get("label_name", "REAL"),
                    "probability_fake": float(pred.get("proba", 0.0)),
                    "confidence": float(pred.get("confidence", 0.0)),
                }
            )
        summary = summarize_v3_deepfakebench(frame_predictions)
        verdict = summary["verdict"]
        confidence = summary["confidence_pct"]
    elif model == "v4":
        detector = get_deepfake_detector_v4()
        frame_predictions = []
        for frame, bbox in zip(frames_with_face, tracked_bboxes):
            coords = get_crop_coords_for_model(frame.size, bbox, model_type="bisenet")
            if not coords:
                continue
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame.crop((x1, y1, x2, y2))
            pred = detector.predict(face_crop, return_proba=True, threshold=V3_V4_THRESHOLD)
            frame_predictions.append(
                {
                    "label": pred.get("label_name", "REAL"),
                    "probability_fake": float(pred.get("proba", 0.0)),
                    "confidence": float(pred.get("confidence", 0.0)),
                }
            )
        summary = summarize_v4_selim(frame_predictions)
        verdict = summary["verdict"]
        confidence = summary["confidence_pct"]
    elif model == "v5":
        detector = get_deepfake_detector_v5()
        frame_predictions = []
        for frame, bbox in zip(frames_with_face, tracked_bboxes):
            coords = get_crop_coords_for_model(frame.size, bbox, model_type="bisenet")
            if not coords:
                continue
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame.crop((x1, y1, x2, y2))
            pred = detector.predict(face_crop, return_proba=True, threshold=V5_DEEPFAKE_THRESHOLD)
            frame_predictions.append(
                {
                    "label": pred.get("label_name", "REAL"),
                    "probability_fake": float(pred.get("proba", 0.0)),
                    "confidence": float(pred.get("confidence", 0.0)),
                }
            )
        summary = summarize_v3_deepfakebench(frame_predictions)
        verdict = summary["verdict"]
        confidence = summary["confidence_pct"]
    elif model == "v6":
        detector = get_deepfake_detector_v6()
        frame_predictions = []
        for frame, bbox in zip(frames_with_face, tracked_bboxes):
            coords = get_crop_coords_for_model(frame.size, bbox, model_type="bisenet")
            if not coords:
                continue
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame.crop((x1, y1, x2, y2))
            pred = detector.predict(face_crop, return_proba=True, threshold=V6_DEEPFAKE_THRESHOLD)
            frame_predictions.append(
                {
                    "label": pred.get("label_name", "REAL"),
                    "probability_fake": float(pred.get("proba", 0.0)),
                    "confidence": float(pred.get("confidence", 0.0)),
                }
            )
        summary = summarize_v4_selim(frame_predictions)
        verdict = summary["verdict"]
        confidence = summary["confidence_pct"]
    elif model == "v7":
        detector = get_deepfake_detector_v7()
        frame_predictions = []
        for frame, bbox in zip(frames_with_face, tracked_bboxes):
            coords = get_crop_coords_for_model(frame.size, bbox, model_type="bisenet")
            if not coords:
                continue
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame.crop((x1, y1, x2, y2))
            pred = detector.predict(face_crop, return_proba=True, threshold=V7_DEEPFAKE_THRESHOLD)
            frame_predictions.append(
                {
                    "label": pred.get("label_name", "REAL"),
                    "probability_fake": float(pred.get("proba", 0.0)),
                    "confidence": float(pred.get("confidence", 0.0)),
                }
            )
        summary = summarize_kaggle_binary(
            frame_predictions,
            deepfake_threshold=V7_DEEPFAKE_THRESHOLD,
            authentic_threshold=V7_AUTHENTIC_THRESHOLD,
            sampled_count=24,
        )
        verdict = summary["verdict"]
        confidence = summary["confidence_pct"]
    elif model == "v8":
        detector = get_deepfake_detector_v8()
        frame_predictions = []
        for frame, bbox in zip(frames_with_face, tracked_bboxes):
            coords = get_crop_coords_for_model(frame.size, bbox, model_type="bisenet")
            if not coords:
                continue
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame.crop((x1, y1, x2, y2))
            pred = detector.predict(face_crop, return_proba=True, threshold=V8_DEEPFAKE_THRESHOLD)
            frame_predictions.append(
                {
                    "label": pred.get("label_name", "REAL"),
                    "probability_fake": float(pred.get("proba", 0.0)),
                    "confidence": float(pred.get("confidence", 0.0)),
                }
            )
        summary = summarize_kaggle_binary(
            frame_predictions,
            deepfake_threshold=V8_DEEPFAKE_THRESHOLD,
            authentic_threshold=V8_AUTHENTIC_THRESHOLD,
            sampled_count=24,
        )
        verdict = summary["verdict"]
        confidence = summary["confidence_pct"]
    else:
        detector = get_deepfake_detector_v9()
        frame_predictions = []
        for frame, bbox in zip(frames_with_face, tracked_bboxes):
            coords = get_crop_coords_for_model(frame.size, bbox, model_type="bisenet")
            if not coords:
                continue
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame.crop((x1, y1, x2, y2))
            pred = detector.predict(face_crop, return_proba=True, threshold=V9_DEEPFAKE_THRESHOLD)
            frame_predictions.append(
                {
                    "label": pred.get("label_name", "REAL"),
                    "probability_fake": float(pred.get("proba", 0.0)),
                    "confidence": float(pred.get("confidence", 0.0)),
                }
            )
        summary = summarize_kaggle_binary(
            frame_predictions,
            deepfake_threshold=V9_DEEPFAKE_THRESHOLD,
            authentic_threshold=V9_AUTHENTIC_THRESHOLD,
            sampled_count=V9_SAMPLED_FRAMES,
        )
        verdict = summary["verdict"]
        confidence = summary["confidence_pct"]

    return {
        "verdict": verdict,
        "prediction": normalize_verdict(verdict),
        "confidence": confidence,
        "fps": fps,
        "total_frames": total_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline benchmark for deepfake models")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--video", help="Optional specific video filename from CASES")
    args = parser.parse_args()

    selected_cases = [c for c in CASES if not args.video or c[0] == args.video]

    rows = []
    for name, path, truth in selected_cases:
        if not path.exists():
            rows.append({"video": name, "truth": truth, "model": "-", "error": "missing file"})
            continue
        for model in args.models:
            print(f"Running {model} on {name} ...", flush=True)
            try:
                result = run_case(path, model, max_frames=args.max_frames)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                result = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "prediction": "error",
                    "verdict": "ERROR",
                    "confidence": None,
                }
            pred = result.get("prediction", "error")
            outcome = "error"
            if pred in {"real", "fake"}:
                if truth == "real" and pred == "fake":
                    outcome = "FP"
                elif truth == "fake" and pred == "real":
                    outcome = "FN"
                else:
                    outcome = "OK"
            elif pred in {"unknown", "uncertain"}:
                outcome = pred
            rows.append(
                {
                    "video": name,
                    "truth": truth,
                    "model": model,
                    "prediction": pred,
                    "verdict": result.get("verdict"),
                    "confidence": result.get("confidence"),
                    "outcome": outcome,
                    "error": result.get("error"),
                }
            )
            print(rows[-1], flush=True)

    out = ROOT / "benchmark_offline_results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
