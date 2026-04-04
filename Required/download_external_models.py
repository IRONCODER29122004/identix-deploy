"""
Download pretrained deepfake checkpoints from external GitHub projects.

Usage:
  python download_external_models.py --list
  python download_external_models.py --models deepfakebench_meso4
  python download_external_models.py --models deepfakebench_meso4,deepfakebench_xception
"""

import argparse
import os
import sys
import urllib.request

MODEL_REGISTRY = {
    "deepfakebench_meso4": {
        "url": "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/meso4_best.pth",
        "target": "models/deepfakebench_meso4_best.pth",
        "project": "SCLBD/DeepfakeBench",
    },
    "deepfakebench_xception": {
        "url": "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth",
        "target": "models/deepfakebench_xception_best.pth",
        "project": "SCLBD/DeepfakeBench",
    },
    "deepfakebench_effnb4": {
        "url": "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/effnb4_best.pth",
        "target": "models/deepfakebench_effnb4_best.pth",
        "project": "SCLBD/DeepfakeBench",
    },
    "deepfakebench_meso4inception": {
        "url": "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/meso4Incep_best.pth",
        "target": "models/deepfakebench_meso4Incep_best.pth",
        "project": "SCLBD/DeepfakeBench",
    },
    "selimsef_b7_111": {
        "url": "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
        "target": "models/selimsef_b7_final_111.pth",
        "project": "selimsef/dfdc_deepfake_challenge",
    },
    "selimsef_b7_555": {
        "url": "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
        "target": "models/selimsef_b7_final_555.pth",
        "project": "selimsef/dfdc_deepfake_challenge",
    },
    "selimsef_b7_777": {
        "url": "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29",
        "target": "models/selimsef_b7_final_777.pth",
        "project": "selimsef/dfdc_deepfake_challenge",
    },
    "selimsef_b7_777_alt": {
        "url": "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31",
        "target": "models/selimsef_b7_final_777_alt.pth",
        "project": "selimsef/dfdc_deepfake_challenge",
    },
    "selimsef_b7_888": {
        "url": "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37",
        "target": "models/selimsef_b7_final_888.pth",
        "project": "selimsef/dfdc_deepfake_challenge",
    },
    "selimsef_b7_999": {
        "url": "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
        "target": "models/selimsef_b7_final_999.pth",
        "project": "selimsef/dfdc_deepfake_challenge",
    },
}


def _download(url, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    urllib.request.urlretrieve(url, target)


def main():
    parser = argparse.ArgumentParser(description="Download external deepfake model checkpoints")
    parser.add_argument("--list", action="store_true", help="List available model keys")
    parser.add_argument(
        "--models",
        type=str,
        default="deepfakebench_meso4",
        help="Comma-separated model keys to download",
    )
    args = parser.parse_args()

    if args.list:
        print("Available model keys:")
        for key, meta in MODEL_REGISTRY.items():
            print(f"- {key}: {meta['project']} -> {meta['target']}")
        return 0

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODEL_REGISTRY]
    if unknown:
        print(f"Unknown model key(s): {', '.join(unknown)}")
        return 1

    for key in keys:
        meta = MODEL_REGISTRY[key]
        target = meta["target"]
        print(f"Downloading {key} from {meta['project']}")
        print(f"  URL: {meta['url']}")
        print(f"  Target: {target}")
        try:
            _download(meta["url"], target)
            size_mb = os.path.getsize(target) / (1024 * 1024)
            print(f"  Done ({size_mb:.2f} MB)")
        except Exception as exc:
            print(f"  Failed: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
