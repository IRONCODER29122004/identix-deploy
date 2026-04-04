# v9 Kaggle FFA-MPDV Separate Pipeline

This folder contains an isolated pipeline for the Kaggle-produced v9 model.

## Assets
- Model checkpoint: `Required/models/ffa_mpdv_v9_kaggle_paper_baseline.pth`
- Notebook source snapshot: `Required/notebooks/kaggle_versions/kaggle_v9_source.ipynb`

## Scripts
- `tune_v9_thresholds.py`: grid-searches v9 aggregation thresholds on local benchmark videos.
- `tune_v9_quick.py`: lightweight CPU-safe threshold sweep for fast tuning.
- `run_v9_only.py`: runs only v9 on benchmark videos and saves isolated output.
- `compare_v9_against_existing.py`: compares v9 with v1-v8 using benchmark outputs.

## Config and Reports
- `v9_runtime_config.json`: v9 default runtime/threshold settings.
- `v9_vs_existing_summary.md`: baseline comparison summary against existing models.

## Quick usage
```bash
cd Required
python pipelines/v9_kaggle_ffa_mpdv/tune_v9_thresholds.py
python benchmark_offline_models.py --models v1 v2 v3 v4 v5 v6 v7 v8 v9
python pipelines/v9_kaggle_ffa_mpdv/compare_v9_against_existing.py
```
