from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "paths": {
        "dataset_yaml": "dataset/data.yaml",
        "source_images": "dataset/train/images",
        "source_labels": "dataset/train/labels",
        "artifacts_dir": "artifacts",
        "runs_dir": "runs",
    },
    "split": {
        "val_ratio": 0.2,
        "validate_images": True,
        "auto_fix_labels": True,
        "oversample_minority": True,
        "imbalance_target_ratio": 0.6,
        "imbalance_max_repeat": 4,
    },
    "detection": {
        "model": "yolo12n.pt",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "device": "0",
        "workers": 8,
        "amp": True,
        "auto_safe_mode": True,
        "cpu_fallback": False,
        "print_epoch_metrics": True,
        "auto_resume": True,
        "resume_checkpoint": "",
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.1,
        "mosaic": 0.0,
        "erasing": 0.0,
        "auto_augment": None,
        "close_mosaic": 0,
        "project": "runs/detection",
        "name": "yolo12_detector",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_det": 100,
    },
    "sam": {
        "model_path": "sam3.pt",
        "conf_threshold": 0.25,
        "min_mask_area": 64,
    },
    "pseudo_segmentation": {
        "prompt_source": "gt",
        "copy_images": False,
        "save_debug_masks": False,
        "output_dir": "artifacts/pseudo_segmentation_dataset",
    },
    "segmentation": {
        "model": "yolo12n-seg.pt",
        "epochs": 60,
        "imgsz": 640,
        "batch": 16,
        "device": "0",
        "workers": 8,
        "amp": True,
        "auto_safe_mode": True,
        "cpu_fallback": False,
        "print_epoch_metrics": True,
        "auto_resume": True,
        "resume_checkpoint": "",
        "degrees": 5.0,
        "shear": 2.0,
        "perspective": 0.0005,
        "flipud": 0.1,
        "mixup": 0.1,
        "cutmix": 0.05,
        "copy_paste": 0.1,
        "project": "runs/segmentation",
        "name": "yolo12_segmentation",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
    },
    "inference": {
        "conf_threshold": 0.25,
        "min_leaf_area": 100.0,
        "leaf_area_mode": "bbox",
        "output_dir": "artifacts/inference",
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    if not isinstance(user_config, dict):
        raise ValueError("Top-level config must be a mapping.")

    return _deep_update(config, user_config)


def resolve_path(value: str | Path, cwd: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(cwd) if cwd is not None else Path.cwd()
    return (root / path).resolve()
