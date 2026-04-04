# v9 Integration and Comparison Summary

## New v9 model
- Model key: `ffa_mpdv_kaggle_v9`
- Checkpoint: `models/ffa_mpdv_v9_kaggle_paper_baseline.pth`
- Source notebook snapshot: `notebooks/kaggle_versions/kaggle_v9_source.ipynb`
- Runtime config: `pipelines/v9_kaggle_ffa_mpdv/v9_runtime_config.json`

## Notebook performance (Kaggle run output)
- ROC AUC: 0.95
- PR AP: 0.96

## Existing benchmark baseline (from current project JSONs)
Source files:
- `benchmark_matrix_latest.json`
- `benchmark_kaggle_v7_v8_results.json`

Outcome count by model:
- v1: OK=0, FP=1, FN=0, uncertain=1
- v2: OK=1, FP=1, FN=0, uncertain=0
- v3: OK=1, FP=0, FN=0, uncertain=1
- v4: OK=2, FP=0, FN=0, uncertain=0
- v5: OK=0, FP=0, FN=0, uncertain=2
- v6: OK=1, FP=0, FN=0, uncertain=1
- v7: OK=1, FP=0, FN=1, uncertain=0
- v8: OK=1, FP=0, FN=1, uncertain=0

## Practical read
- v9 has stronger curve-quality metrics (AUC/AP) than prior reported checkpoints.
- v4 is strongest on the current tiny local benchmark mix (2/2 OK among listed rows).
- v7/v8 show an FN on `output_deepfake.mp4`, so v9 is integrated as an additional option and auto-best candidate.

## Next run commands
```bash
cd Required
python benchmark_offline_models.py --models v4 v7 v8 v9 --max-frames 24
python pipelines/v9_kaggle_ffa_mpdv/compare_v9_against_existing.py
```
