import argparse
import json
from pathlib import Path

from benchmark_offline_models import run_case, CASES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "v9_only_results.json"


def main():
    parser = argparse.ArgumentParser(description="Run isolated v9 pipeline on benchmark videos")
    parser.add_argument("--max-frames", type=int, default=24)
    args = parser.parse_args()

    rows = []
    for name, path, truth in CASES:
        if not path.exists():
            rows.append({"video": name, "truth": truth, "error": "missing file"})
            continue

        result = run_case(path, "v9", max_frames=args.max_frames)
        rows.append(
            {
                "video": name,
                "truth": truth,
                "model": "v9",
                "verdict": result.get("verdict"),
                "prediction": result.get("prediction"),
                "confidence": result.get("confidence"),
                "error": result.get("error"),
            }
        )

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
