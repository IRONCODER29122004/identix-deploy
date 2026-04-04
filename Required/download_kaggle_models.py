"""
Download selected pretrained deepfake checkpoints from Kaggle datasets.

Usage:
  python download_kaggle_models.py --list
  python download_kaggle_models.py --models cyrine_v7_weights,cyrine_v8_model
"""

import argparse
import os
import shutil

import kagglehub

MODEL_REGISTRY = {
    "cyrine_v7_weights": {
        "dataset": "cyrinegraf/deepfake-efficientnet-trained-model",
        "file": "best_weights.pth",
        "target": "models/kaggle_cyrine_best_weights.pth",
        "notes": "EfficientNet-B0 binary checkpoint (lightweight Kaggle source)",
    },
    "cyrine_v8_model": {
        "dataset": "cyrinegraf/deepfake-efficientnet-trained-model",
        "file": "best_model.pth",
        "target": "models/kaggle_cyrine_best_model.pth",
        "notes": "EfficientNet-B0 training checkpoint with model_state_dict",
    },
    "kuvalgarg_xception": {
        "dataset": "kuvalgarg/dfdc-ensemble-model-weights",
        "file": "xception-net.pth",
        "target": "models/kaggle_kuval_xception_net.pth",
        "notes": "Large Xception checkpoint; may take longer to download",
    },
    "kuvalgarg_effnet_b0": {
        "dataset": "kuvalgarg/dfdc-ensemble-model-weights",
        "file": "efficientnet-b0.pth",
        "target": "models/kaggle_kuval_efficientnet_b0.pth",
        "notes": "Alternate EfficientNet-B0 checkpoint",
    },
}


def download_one(key: str):
    meta = MODEL_REGISTRY[key]
    target = meta["target"]
    os.makedirs(os.path.dirname(target), exist_ok=True)

    print(f"Downloading {key}")
    print(f"  Dataset: {meta['dataset']}")
    print(f"  File: {meta['file']}")
    local_path = kagglehub.dataset_download(meta["dataset"], path=meta["file"])
    shutil.copy2(local_path, target)

    size_mb = os.path.getsize(target) / (1024 * 1024)
    print(f"  Saved: {target} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download deepfake checkpoints from Kaggle")
    parser.add_argument("--list", action="store_true", help="List available model keys")
    parser.add_argument(
        "--models",
        type=str,
        default="cyrine_v7_weights,cyrine_v8_model",
        help="Comma-separated model keys",
    )
    args = parser.parse_args()

    if args.list:
        print("Available model keys:")
        for key, meta in MODEL_REGISTRY.items():
            print(f"- {key}: {meta['dataset']}::{meta['file']} -> {meta['target']}")
        return 0

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODEL_REGISTRY]
    if unknown:
        print(f"Unknown model key(s): {', '.join(unknown)}")
        return 1

    for key in keys:
        try:
            download_one(key)
        except Exception as exc:
            print(f"Failed to download {key}: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
