from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import resolve_path
from .data.detection_split import build_detection_split, extract_class_names
from .data.pseudo_segmentation_builder import build_pseudo_segmentation_dataset
from .models.detector import YOLOv12Detector, pick_best_weights
from .models.sam3_wrapper import SAM3MaskGenerator
from .models.segmenter import YOLOSegmenter
from .severity import compute_severity
from .utils.fs import ensure_dir, read_yaml, write_yaml
from .utils.geometry import binary_mask_from_bbox, box_iou_xyxy
from .utils.visualization import draw_inference_result


def _stage_start(name: str) -> float:
    print(f"[pipeline] >>> {name} started", flush=True)
    return time.perf_counter()


def _stage_end(name: str, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    print(f"[pipeline] <<< {name} completed in {elapsed:.1f}s", flush=True)


def _artifacts_file(cfg: dict[str, Any]) -> Path:
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    ensure_dir(artifacts_dir)
    return artifacts_dir / "pipeline_artifacts.yaml"


def load_artifacts(cfg: dict[str, Any]) -> dict[str, Any]:
    path = _artifacts_file(cfg)
    if path.exists():
        return read_yaml(path)
    return {}


def save_artifacts(cfg: dict[str, Any], artifacts: dict[str, Any]) -> Path:
    path = _artifacts_file(cfg)
    write_yaml(path, artifacts)
    return path


def update_artifacts(cfg: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    artifacts = load_artifacts(cfg)
    artifacts.update(updates)
    save_artifacts(cfg, artifacts)
    return artifacts


def get_class_names(cfg: dict[str, Any]) -> list[str]:
    dataset_yaml = resolve_path(cfg["paths"]["dataset_yaml"])
    return extract_class_names(dataset_yaml)


def prepare_detection_data(cfg: dict[str, Any]) -> dict[str, Any]:
    t0 = _stage_start("Prepare Detection Data")
    class_names = get_class_names(cfg)

    output_dir = resolve_path(cfg["paths"]["artifacts_dir"]) / "detection_split"
    split_info = build_detection_split(
        source_images_dir=resolve_path(cfg["paths"]["source_images"]),
        source_labels_dir=resolve_path(cfg["paths"]["source_labels"]),
        class_names=class_names,
        val_ratio=float(cfg["split"]["val_ratio"]),
        seed=int(cfg["seed"]),
        output_dir=output_dir,
        validate_images=bool(cfg["split"].get("validate_images", True)),
        auto_fix_labels=bool(cfg["split"].get("auto_fix_labels", True)),
        oversample_minority=bool(cfg["split"].get("oversample_minority", True)),
        imbalance_target_ratio=float(cfg["split"].get("imbalance_target_ratio", 0.6)),
        imbalance_max_repeat=int(cfg["split"].get("imbalance_max_repeat", 4)),
    )

    artifacts = update_artifacts(
        cfg,
        {
            "class_names": class_names,
            **split_info,
        },
    )
    print(f"[pipeline] Detection split file: {artifacts.get('detection_data_yaml')}", flush=True)
    _stage_end("Prepare Detection Data", t0)
    return artifacts


def train_detector(cfg: dict[str, Any], data_yaml: str | Path | None = None) -> dict[str, Any]:
    t0 = _stage_start("Train Detector")
    artifacts = load_artifacts(cfg)
    if data_yaml is None:
        if "detection_data_yaml" not in artifacts:
            artifacts = prepare_detection_data(cfg)
        data_yaml = artifacts["detection_data_yaml"]

    detector = YOLOv12Detector(cfg["detection"]["model"])
    train_dir = detector.train(data_yaml, cfg["detection"])
    weights = pick_best_weights(train_dir)

    artifacts = update_artifacts(
        cfg,
        {
            "detector_train_dir": str(Path(train_dir).resolve()),
            "detector_weights": str(Path(weights).resolve()),
        },
    )
    print(f"[pipeline] Detector weights: {artifacts.get('detector_weights')}", flush=True)
    _stage_end("Train Detector", t0)
    return artifacts


def build_pseudo_segmentation(cfg: dict[str, Any], detector_weights: str | Path | None = None) -> dict[str, Any]:
    t0 = _stage_start("Build Pseudo Segmentation Dataset")
    artifacts = load_artifacts(cfg)
    if "train_txt" not in artifacts or "val_txt" not in artifacts:
        artifacts = prepare_detection_data(cfg)

    class_names = artifacts.get("class_names") or get_class_names(cfg)
    train_txt = artifacts["train_txt"]
    val_txt = artifacts["val_txt"]

    prompt_source = str(cfg["pseudo_segmentation"]["prompt_source"]).strip().lower()

    detector = None
    if prompt_source == "detector":
        if detector_weights is None:
            detector_weights = artifacts.get("detector_weights")
        if detector_weights is None:
            raise ValueError(
                "prompt_source='detector' requires detector weights. "
                "Run train-detector first or pass --detector-weights."
            )
        detector = YOLOv12Detector(detector_weights)

    sam = SAM3MaskGenerator(
        model_path=resolve_path(cfg["sam"]["model_path"]),
        conf_threshold=float(cfg["sam"]["conf_threshold"]),
    )

    seg_info = build_pseudo_segmentation_dataset(
        train_txt=train_txt,
        val_txt=val_txt,
        source_labels_dir=resolve_path(cfg["paths"]["source_labels"]),
        class_names=class_names,
        sam=sam,
        output_dir=resolve_path(cfg["pseudo_segmentation"]["output_dir"]),
        prompt_source=prompt_source,
        detector=detector,
        detector_cfg=cfg["detection"],
        min_mask_area=int(cfg["sam"]["min_mask_area"]),
        copy_images=bool(cfg["pseudo_segmentation"]["copy_images"]),
        save_debug_masks=bool(cfg["pseudo_segmentation"]["save_debug_masks"]),
    )

    artifacts = update_artifacts(cfg, seg_info)
    print(f"[pipeline] Segmentation data YAML: {artifacts.get('segmentation_data_yaml')}", flush=True)
    _stage_end("Build Pseudo Segmentation Dataset", t0)
    return artifacts


def train_segmenter(cfg: dict[str, Any], seg_data_yaml: str | Path | None = None) -> dict[str, Any]:
    t0 = _stage_start("Train Segmenter")
    artifacts = load_artifacts(cfg)
    if seg_data_yaml is None:
        if "segmentation_data_yaml" not in artifacts:
            artifacts = build_pseudo_segmentation(cfg)
        seg_data_yaml = artifacts["segmentation_data_yaml"]

    segmenter = YOLOSegmenter(cfg["segmentation"]["model"])
    train_dir = segmenter.train(seg_data_yaml, cfg["segmentation"])
    weights = pick_best_weights(train_dir)

    artifacts = update_artifacts(
        cfg,
        {
            "segmenter_train_dir": str(Path(train_dir).resolve()),
            "segmenter_weights": str(Path(weights).resolve()),
        },
    )
    print(f"[pipeline] Segmenter weights: {artifacts.get('segmenter_weights')}", flush=True)
    _stage_end("Train Segmenter", t0)
    return artifacts


def _match_segmentation_instance(
    detection_row: dict[str, Any],
    seg_instances: list[dict[str, Any]],
) -> dict[str, Any] | None:
    cls_id = int(detection_row["class_id"])
    det_box = detection_row["bbox_xyxy"]

    candidates = [r for r in seg_instances if int(r["class_id"]) == cls_id]
    if not candidates:
        return None

    best = None
    best_iou = -1.0
    for row in candidates:
        iou = box_iou_xyxy(det_box, row["bbox_xyxy"])
        if iou > best_iou:
            best_iou = iou
            best = row
    if best is None:
        return None
    if best_iou <= 0.0:
        return None
    return best


def run_inference(
    cfg: dict[str, Any],
    image_path: str | Path,
    detector_weights: str | Path | None = None,
    segmenter_weights: str | Path | None = None,
) -> dict[str, Any]:
    t0 = _stage_start("Inference")
    image_path = resolve_path(image_path)
    artifacts = load_artifacts(cfg)

    if detector_weights is None:
        detector_weights = artifacts.get("detector_weights")
    if detector_weights is None:
        raise ValueError("Detector weights are required. Train detector first or pass --detector-weights.")

    if segmenter_weights is None:
        segmenter_weights = artifacts.get("segmenter_weights")
    if segmenter_weights is None:
        raise ValueError("Segmenter weights are required. Train segmenter first or pass --segmenter-weights.")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]

    detector = YOLOv12Detector(detector_weights)
    segmenter = YOLOSegmenter(segmenter_weights)
    detections = detector.predict(image_path, cfg["detection"])
    seg_instances = segmenter.predict(image_path, cfg["segmentation"])

    leaf_area_mode = str(cfg["inference"]["leaf_area_mode"]).strip().lower()
    sam_leaf = None
    if leaf_area_mode == "sam":
        sam_leaf = SAM3MaskGenerator(
            model_path=resolve_path(cfg["sam"]["model_path"]),
            conf_threshold=float(cfg["sam"]["conf_threshold"]),
        )

    rows_for_output: list[dict[str, Any]] = []
    for det in detections:
        bbox = det["bbox_xyxy"]
        det_cls = int(det["class_id"])
        det_conf = float(det["confidence"])

        matched = _match_segmentation_instance(det, seg_instances)
        if matched is not None:
            lesion_mask = matched["mask"].astype(bool)
            roi_mask = binary_mask_from_bbox(bbox, w, h)
            lesion_mask = np.logical_and(lesion_mask, roi_mask)
        else:
            lesion_mask = binary_mask_from_bbox(bbox, w, h)

        leaf_mask = None
        if sam_leaf is not None:
            leaf_mask, _ = sam_leaf.generate_best_mask(image_path, bbox)

        severity_stats = compute_severity(
            lesion_mask=lesion_mask,
            leaf_mask=leaf_mask,
            fallback_leaf_box=bbox,
            min_leaf_area=float(cfg["inference"]["min_leaf_area"]),
        )

        row = {
            "class_id": det_cls,
            "det_conf": det_conf,
            "bbox_xyxy": [float(v) for v in bbox],
            "lesion_area": severity_stats["lesion_area"],
            "leaf_area": severity_stats["leaf_area"],
            "severity_percent": severity_stats["severity_percent"],
            "lesion_mask": lesion_mask,
            "leaf_mask": leaf_mask,
        }
        rows_for_output.append(row)

    class_names = artifacts.get("class_names") or get_class_names(cfg)
    vis = draw_inference_result(image, rows_for_output, class_names)

    output_dir = ensure_dir(resolve_path(cfg["inference"]["output_dir"]))
    json_path = output_dir / f"{image_path.stem}_severity.json"
    vis_path = output_dir / f"{image_path.stem}_severity.jpg"

    serializable_rows: list[dict[str, Any]] = []
    for row in rows_for_output:
        cls_id = int(row["class_id"])
        class_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
        serializable_rows.append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "det_conf": float(row["det_conf"]),
                "bbox_xyxy": [float(v) for v in row["bbox_xyxy"]],
                "lesion_area": float(row["lesion_area"]),
                "leaf_area": float(row["leaf_area"]),
                "severity_percent": float(row["severity_percent"]),
            }
        )

    report = {
        "image_path": str(image_path),
        "detections": serializable_rows,
        "summary": {
            "num_detections": len(serializable_rows),
            "mean_severity_percent": float(
                np.mean([r["severity_percent"] for r in serializable_rows])
                if serializable_rows
                else 0.0
            ),
            "max_severity_percent": float(
                np.max([r["severity_percent"] for r in serializable_rows])
                if serializable_rows
                else 0.0
            ),
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    cv2.imwrite(str(vis_path), vis)

    _stage_end("Inference", t0)
    return {
        "report_json": str(json_path.resolve()),
        "visualization_image": str(vis_path.resolve()),
    }


def run_all(cfg: dict[str, Any]) -> dict[str, Any]:
    t0 = _stage_start("Full Pipeline")
    prepare_detection_data(cfg)
    train_detector(cfg)
    build_pseudo_segmentation(cfg)
    train_segmenter(cfg)
    artifacts = load_artifacts(cfg)
    _stage_end("Full Pipeline", t0)
    return artifacts
