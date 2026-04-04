import json
import sys
import io
import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark_offline_models as bom
from benchmark_offline_models import CASES, run_case

OUT = Path(__file__).resolve().parent / "v9_tuning_quick.json"


def normalize(verdict):
    s = (verdict or "").lower()
    if "authentic" in s or "real" in s:
        return "real"
    if "deepfake" in s or "fake" in s:
        return "fake"
    if "manual review" in s or "low confidence" in s:
        return "uncertain"
    return "unknown"


def score_case(truth, pred):
    if pred == "real" and truth == "real":
        return "OK"
    if pred == "fake" and truth == "fake":
        return "OK"
    if pred == "fake" and truth == "real":
        return "FP"
    if pred == "real" and truth == "fake":
        return "FN"
    if pred == "uncertain":
        return "uncertain"
    return "error"


configs = [
    (0.64, 0.30, 24),
    (0.66, 0.32, 24),
    (0.68, 0.34, 24),
]

rows = []
best = None
for d, a, s in configs:
    bom.V9_DEEPFAKE_THRESHOLD = d
    bom.V9_AUTHENTIC_THRESHOLD = a
    bom.V9_SAMPLED_FRAMES = s

    ok = fp = fn = uncertain = error = 0
    details = []

    for name, path, truth in CASES:
        if not path.exists():
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = run_case(path, "v9", max_frames=36)
            pred = normalize(result.get("verdict"))
            outcome = score_case(truth, pred)
        except Exception as exc:
            outcome = "error"
            result = {"error": f"{type(exc).__name__}: {exc}"}

        ok += int(outcome == "OK")
        fp += int(outcome == "FP")
        fn += int(outcome == "FN")
        uncertain += int(outcome == "uncertain")
        error += int(outcome == "error")
        details.append(
            {
                "video": name,
                "truth": truth,
                "verdict": result.get("verdict"),
                "confidence": result.get("confidence"),
                "outcome": outcome,
                "error": result.get("error"),
            }
        )

    fitness = ok * 10.0 - fp * 6.0 - fn * 8.0 - uncertain * 2.0 - error * 12.0
    item = {
        "deepfake_threshold": d,
        "authentic_threshold": a,
        "sampled_frames": s,
        "fitness": fitness,
        "OK": ok,
        "FP": fp,
        "FN": fn,
        "uncertain": uncertain,
        "error": error,
        "details": details,
    }
    rows.append(item)
    if best is None or item["fitness"] > best["fitness"]:
        best = item

rows.sort(key=lambda x: x["fitness"], reverse=True)
output = {"best": best, "top5": rows[:5], "all": rows}
OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps(best, indent=2))
print(f"Saved: {OUT}")
