from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from ..models.detector import YOLOv12Detector
from ..models.sam3_wrapper import SAM3MaskGenerator
from ..utils.fs import copy_or_link_file, ensure_dir, read_lines, write_yaml
from ..utils.geometry import (
    clip_box_xyxy,
    mask_to_polygon,
    polygon_from_bbox,
    yolo_xywh_to_xyxy,
)


def _load_gt_boxes(label_path: Path, image_w: int, image_h: int) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x_c, y_c, w, h = map(float, parts[1:5])
            box = yolo_xywh_to_xyxy(x_c, y_c, w, h, image_w, image_h)
            box = clip_box_xyxy(box, image_w, image_h)
            rows.append({"class_id": class_id, "bbox_xyxy": box, "confidence": 1.0})
    return rows


def build_pseudo_segmentation_dataset(
    train_txt: str | Path,
    val_txt: str | Path,
    source_labels_dir: str | Path,
    class_names: list[str],
    sam: SAM3MaskGenerator,
    output_dir: str | Path,
    prompt_source: str = "gt",
    detector: YOLOv12Detector | None = None,
    detector_cfg: dict[str, Any] | None = None,
    min_mask_area: int = 64,
    copy_images: bool = False,
    save_debug_masks: bool = False,
) -> dict[str, str]:
    output_root = ensure_dir(output_dir)
    source_labels = Path(source_labels_dir)
    prompt_source = prompt_source.strip().lower()

    if prompt_source not in {"gt", "detector"}:
        raise ValueError("prompt_source must be either 'gt' or 'detector'.")
    if prompt_source == "detector" and detector is None:
        raise ValueError("Detector must be provided when prompt_source='detector'.")

    split_to_list = {
        "train": [Path(p) for p in read_lines(train_txt)],
        "val": [Path(p) for p in read_lines(val_txt)],
    }

    debug_root = output_root / "debug_masks"
    if save_debug_masks:
        ensure_dir(debug_root)

    for split_name, image_list in split_to_list.items():
        images_out = ensure_dir(output_root / "images" / split_name)
        labels_out = ensure_dir(output_root / "labels" / split_name)
        split_debug_dir = ensure_dir(debug_root / split_name) if save_debug_masks else None

        for image_path in tqdm(image_list, desc=f"Pseudo labels [{split_name}]"):
            if not image_path.exists():
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue
            h, w = image.shape[:2]

            prompts: list[dict[str, Any]]
            if prompt_source == "gt":
                label_path = source_labels / f"{image_path.stem}.txt"
                prompts = _load_gt_boxes(label_path, w, h)
            else:
                assert detector is not None
                assert detector_cfg is not None
                prompts = detector.predict(image_path, detector_cfg)

            image_dst = images_out / image_path.name
            copy_or_link_file(image_path, image_dst, force_copy=copy_images)
            label_dst = labels_out / f"{image_path.stem}.txt"

            label_lines: list[str] = []
            debug_canvas = np.zeros((h, w), dtype=np.uint8) if split_debug_dir is not None else None

            for row in prompts:
                class_id = int(row["class_id"])
                box = clip_box_xyxy(row["bbox_xyxy"], w, h)

                mask, score = sam.generate_best_mask(image_path, box)
                polygon: list[float] | None = None

                if mask is not None:
                    area = int(np.count_nonzero(mask))
                    if area >= int(min_mask_area):
                        polygon = mask_to_polygon(mask)
                        if debug_canvas is not None:
                            debug_canvas[mask] = np.maximum(debug_canvas[mask], 255)

                if polygon is None:
                    polygon = polygon_from_bbox(box, w, h)

                if polygon is None or len(polygon) < 6:
                    continue

                numbers = " ".join(f"{v:.6f}" for v in polygon)
                label_lines.append(f"{class_id} {numbers}")

            with label_dst.open("w", encoding="utf-8") as f:
                for line in label_lines:
                    f.write(f"{line}\n")

            if split_debug_dir is not None and debug_canvas is not None:
                debug_path = split_debug_dir / f"{image_path.stem}.png"
                cv2.imwrite(str(debug_path), debug_canvas)

    seg_data_yaml = output_root / "segmentation_data.yaml"
    write_yaml(
        seg_data_yaml,
        {
            "path": str(output_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names),
            "names": class_names,
        },
    )

    return {
        "segmentation_data_yaml": str(seg_data_yaml.resolve()),
        "segmentation_dataset_root": str(output_root.resolve()),
    }

