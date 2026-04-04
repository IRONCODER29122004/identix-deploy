import json
import argparse
from pathlib import Path
import mimetypes
import uuid
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000/detect_deepfake"
ROOT = Path(__file__).resolve().parent

CASES = [
    ("speaking_person.mp4", ROOT / "faceswap_samples" / "speaking_person.mp4", "real"),
    ("test_person_video.mp4", ROOT / "test_videos" / "test_person_video.mp4", "real"),
    ("output_deepfake.mp4", ROOT / "output_deepfake.mp4", "fake"),
]

MODELS = [
    "bisenet",
    "ffa_mpdv_v2",
    "deepfakebench_meso4_v3",
    "selim_b7_v4",
    "deepfakebench_meso4inception_v5",
    "selim_b7_ensemble_v6",
    "ffa_mpdv_kaggle_v9",
    "auto_best",
]


def normalize_label(raw: str) -> str:
    s = (raw or "").strip().lower()
    if "real" in s:
        return "real"
    if "deepfake" in s or "fake" in s:
        return "fake"
    return "unknown"


def extract_prediction(payload: dict) -> str:
    pred_raw = payload.get("result") or payload.get("label") or ""
    pred = normalize_label(pred_raw)
    if pred != "unknown":
        return pred

    report = payload.get("report") or {}
    verdict = report.get("verdict") or ""
    pred = normalize_label(verdict)
    if pred != "unknown":
        return pred

    is_authentic = report.get("is_authentic")
    if isinstance(is_authentic, bool):
        return "real" if is_authentic else "fake"

    return "unknown"


def to_percent(conf):
    if conf is None:
        return None
    try:
        v = float(conf)
    except (TypeError, ValueError):
        return None
    if v <= 1.0:
        v *= 100.0
    return round(v, 2)


def build_multipart_form(video_path: Path, model: str):
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    crlf = "\r\n"
    parts = []

    parts.append(f"--{boundary}{crlf}".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="model"{crlf}{crlf}{model}{crlf}'
        ).encode("utf-8")
    )

    filename = video_path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    file_bytes = video_path.read_bytes()

    parts.append(f"--{boundary}{crlf}".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="video"; filename="{filename}"{crlf}'
            f"Content-Type: {content_type}{crlf}{crlf}"
        ).encode("utf-8")
    )
    parts.append(file_bytes)
    parts.append(crlf.encode("utf-8"))
    parts.append(f"--{boundary}--{crlf}".encode("utf-8"))

    body = b"".join(parts)
    content_type_header = f"multipart/form-data; boundary={boundary}"
    return body, content_type_header


def run_one(video_path: Path, model: str):
    body, content_type_header = build_multipart_form(video_path, model)
    req = urllib.request.Request(
        BASE_URL,
        data=body,
        method="POST",
        headers={"Content-Type": content_type_header},
    )

    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            status = resp.getcode()
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        status = e.code
        text = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
    except Exception as e:
        return 0, {"error": f"request failed: {type(e).__name__}: {e}"}

    try:
        payload = json.loads(text)
    except Exception:
        payload = {"error": f"non-json response: {text[:200]}"}

    return status, payload


def main():
    parser = argparse.ArgumentParser(description="Benchmark local deepfake API")
    parser.add_argument("--model", choices=MODELS, help="Run only one model")
    parser.add_argument("--video", help="Run only one video case by filename")
    args = parser.parse_args()

    selected_models = [args.model] if args.model else MODELS
    selected_cases = [c for c in CASES if (not args.video or c[0] == args.video)]

    print("Running local benchmark against /detect_deepfake ...")
    print(f"Endpoint: {BASE_URL}")

    missing = [name for name, path, _ in selected_cases if not path.exists()]
    if missing:
        print("Missing files:", ", ".join(missing))
        return

    rows = []
    for case_name, path, truth in selected_cases:
        for model in selected_models:
            print(f"REQUEST: video={case_name} model={model}", flush=True)
            status, payload = run_one(path, model)
            if status != 200 or payload.get("error"):
                rows.append(
                    {
                        "video": case_name,
                        "truth": truth,
                        "model": model,
                        "status": status,
                        "pred": "error",
                        "confidence": None,
                        "raw": payload,
                        "outcome": "error",
                    }
                )
                continue

            pred = extract_prediction(payload)
            conf = to_percent(payload.get("confidence"))

            if pred not in {"real", "fake"}:
                outcome = "unknown"
            elif truth == "real" and pred == "fake":
                outcome = "FP"
            elif truth == "fake" and pred == "real":
                outcome = "FN"
            else:
                outcome = "OK"

            rows.append(
                {
                    "video": case_name,
                    "truth": truth,
                    "model": model,
                    "status": status,
                    "pred": pred,
                    "confidence": conf,
                    "raw": payload,
                    "outcome": outcome,
                }
            )

    print("\nRESULTS")
    print("video\ttruth\tmodel\tpred\tconf(%)\toutcome")
    for row in rows:
        conf_text = "-" if row["confidence"] is None else str(row["confidence"])
        print(
            f"{row['video']}\t{row['truth']}\t{row['model']}\t{row['pred']}\t{conf_text}\t{row['outcome']}"
        )

    summary = {m: {"OK": 0, "FP": 0, "FN": 0, "error": 0, "unknown": 0} for m in selected_models}
    for row in rows:
        summary[row["model"]][row["outcome"]] = summary[row["model"]].get(row["outcome"], 0) + 1

    print("\nSUMMARY_BY_MODEL")
    for m in selected_models:
        s = summary[m]
        print(f"{m}: OK={s.get('OK', 0)} FP={s.get('FP', 0)} FN={s.get('FN', 0)} error={s.get('error', 0)} unknown={s.get('unknown', 0)}")

    out_path = ROOT / "benchmark_results_local_api.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved detailed rows to: {out_path}")


if __name__ == "__main__":
    main()
