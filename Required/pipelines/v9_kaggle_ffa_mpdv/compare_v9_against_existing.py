import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmark_offline_results.json"
OUT = Path(__file__).resolve().parent / "v9_comparison_report.md"


def summarize(rows):
    by_model = {}
    for r in rows:
        m = r.get("model")
        if not m:
            continue
        by_model.setdefault(m, {"OK": 0, "FP": 0, "FN": 0, "uncertain": 0, "error": 0})
        outcome = r.get("outcome") or "error"
        if outcome not in by_model[m]:
            by_model[m][outcome] = 0
        by_model[m][outcome] += 1

    order = sorted(by_model.keys())
    lines = ["# v9 vs Existing Models (Offline Benchmark)", "", "| Model | OK | FP | FN | Uncertain | Error |", "|---|---:|---:|---:|---:|---:|"]
    for m in order:
        s = by_model[m]
        lines.append(
            f"| {m} | {s.get('OK', 0)} | {s.get('FP', 0)} | {s.get('FN', 0)} | {s.get('uncertain', 0)} | {s.get('error', 0)} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- Lower FP/FN is better.")
    lines.append("- Uncertain indicates manual-review outcomes.")

    return "\n".join(lines)


def main():
    if not RESULTS.exists():
        raise FileNotFoundError(f"Missing benchmark results: {RESULTS}")

    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    report = summarize(rows)
    OUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
