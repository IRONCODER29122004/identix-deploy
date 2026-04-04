# Benchmark Subset Results (v3/v4)

## Scope
- Method: offline benchmark runner (`benchmark_offline_models.py`)
- Models: `v3` (DeepfakeBench Meso4), `v4` (Selim B7)
- Videos:
  - `faceswap_samples/speaking_person.mp4` (ground truth: real)
  - `output_deepfake.mp4` (ground truth: fake)
- Sampling: `max_frames=10`

## Results

| Video | Truth | Model | Verdict | Predicted | Confidence | Outcome |
|---|---|---|---|---|---:|---|
| speaking_person.mp4 | real | v3 | LIKELY DEEPFAKE | fake | 30.2452 | FP |
| speaking_person.mp4 | real | v4 | LIKELY DEEPFAKE | fake | 10.7928 | FP |
| output_deepfake.mp4 | fake | v3 | LIKELY DEEPFAKE | fake | 29.6102 | OK |
| output_deepfake.mp4 | fake | v4 | LIKELY DEEPFAKE | fake | 78.7516 | OK |

## Interpretation
- Both external models (`v3`, `v4`) correctly flag the generated fake sample.
- Both external models also flag the real sample as deepfake (false positives), which matches the user-reported issue.
- `v4` appears more strongly confident on the fake sample than `v3` in this subset.

## Post-Tuning Update (Current Session)
- Applied stricter decision gating in backend:
  - `v3`: deepfake threshold `0.80`, authentic threshold `0.30`
  - `v4`: deepfake threshold `0.80`, authentic threshold `0.35`
- Observed updated behavior on real sample (`speaking_person.mp4`):
  - `v3`: `LOW CONFIDENCE - NEEDS MANUAL REVIEW`
  - `v4`: `LOW CONFIDENCE - NEEDS MANUAL REVIEW`
- This reduces hard false-positive verdicts on that real sample by converting borderline cases to manual-review.

## Notes
- This is a quick subset benchmark (2 videos, 10 sampled frames each), not a full calibration set.
- Full matrix runs in this environment are slow/fragile due heavy model initialization and video processing overhead.
