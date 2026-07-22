from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from plant_disease_pipeline.utils.visualization import class_color


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class LabeledImage:
    image_path: Path
    label_path: Path
    boxes: list[dict[str, Any]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _load_class_names(dataset_yaml: Path) -> list[str]:
    data = _load_yaml(dataset_yaml)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda value: int(value))]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def _clip_xyxy(box: list[float], image_w: int, image_h: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, image_w - 1))
    y1 = float(np.clip(y1, 0, image_h - 1))
    x2 = float(np.clip(x2, 0, image_w - 1))
    y2 = float(np.clip(y2, 0, image_h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _xywhn_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_w: int,
    image_h: int,
) -> list[float]:
    return [
        (x_center - width / 2.0) * image_w,
        (y_center - height / 2.0) * image_h,
        (x_center + width / 2.0) * image_w,
        (y_center + height / 2.0) * image_h,
    ]


def _xyxy_to_xywhn(box: list[float], image_w: int, image_h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _clip_xyxy(box, image_w, image_h)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0
    return (
        float(np.clip(x_center / image_w, 0.0, 1.0)),
        float(np.clip(y_center / image_h, 0.0, 1.0)),
        float(np.clip(width / image_w, 0.0, 1.0)),
        float(np.clip(height / image_h, 0.0, 1.0)),
    )


def _read_boxes(label_path: Path, image_w: int, image_h: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not label_path.exists():
        return rows

    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            print(f"Skipping invalid label line {label_path}:{line_no}")
            continue

        if len(values) == 4:
            box = _xywhn_to_xyxy(values[0], values[1], values[2], values[3], image_w, image_h)
        elif len(values) >= 6 and len(values) % 2 == 0:
            points = np.array(values, dtype=np.float32).reshape(-1, 2)
            x1 = float(points[:, 0].min() * image_w)
            y1 = float(points[:, 1].min() * image_h)
            x2 = float(points[:, 0].max() * image_w)
            y2 = float(points[:, 1].max() * image_h)
            box = [x1, y1, x2, y2]
        else:
            box = _xywhn_to_xyxy(values[0], values[1], values[2], values[3], image_w, image_h)

        x1, y1, x2, y2 = _clip_xyxy(box, image_w, image_h)
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            rows.append({"class_id": class_id, "bbox_xyxy": [x1, y1, x2, y2]})
    return rows


def _write_yolo_labels(label_path: Path, boxes: list[dict[str, Any]], image_w: int, image_h: int) -> None:
    lines: list[str] = []
    for row in boxes:
        x_center, y_center, width, height = _xyxy_to_xywhn(row["bbox_xyxy"], image_w, image_h)
        if width <= 0 or height <= 0:
            continue
        lines.append(
            f"{int(row['class_id'])} "
            f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _draw_boxes(image_bgr: np.ndarray, boxes: list[dict[str, Any]], class_names: list[str]) -> np.ndarray:
    canvas = image_bgr.copy()
    image_h, image_w = canvas.shape[:2]
    thickness = max(2, round(min(image_w, image_h) / 250))
    font_scale = max(0.45, min(image_w, image_h) / 900)

    for row in boxes:
        class_id = int(row["class_id"])
        color = class_color(class_id)
        x1, y1, x2, y2 = [int(round(v)) for v in _clip_xyxy(row["bbox_xyxy"], image_w, image_h)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        label = f"{class_id}: {class_name}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        label_x1 = max(0, min(x1, image_w - label_w - 2))
        label_y2 = y1 - 4 if y1 - label_h - baseline - 6 >= 0 else y1 + label_h + baseline + 6
        label_y2 = max(label_h + baseline + 4, min(label_y2, image_h - 1))
        label_y1 = max(0, label_y2 - label_h - baseline - 4)
        cv2.rectangle(canvas, (label_x1, label_y1), (label_x1 + label_w + 2, label_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (label_x1 + 1, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    return canvas


def _random_perspective_matrix(
    image_w: int,
    image_h: int,
    rng: np.random.Generator,
    degrees: float,
    shear: float,
    perspective: float,
) -> np.ndarray:
    center = np.eye(3, dtype=np.float32)
    center[0, 2] = -image_w / 2.0
    center[1, 2] = -image_h / 2.0

    persp = np.eye(3, dtype=np.float32)
    persp[2, 0] = rng.uniform(-perspective, perspective)
    persp[2, 1] = rng.uniform(-perspective, perspective)

    angle = rng.uniform(-degrees, degrees)
    rot_2d = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    rot = np.eye(3, dtype=np.float32)
    rot[:2, :] = rot_2d

    shear_matrix = np.eye(3, dtype=np.float32)
    shear_matrix[0, 1] = np.tan(rng.uniform(-shear, shear) * np.pi / 180.0)
    shear_matrix[1, 0] = np.tan(rng.uniform(-shear, shear) * np.pi / 180.0)

    translate = np.eye(3, dtype=np.float32)
    translate[0, 2] = image_w / 2.0
    translate[1, 2] = image_h / 2.0

    return translate @ shear_matrix @ rot @ persp @ center


def _transform_boxes(
    boxes: list[dict[str, Any]],
    matrix: np.ndarray,
    image_w: int,
    image_h: int,
) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []

    for row in boxes:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        corners = np.array(
            [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
            dtype=np.float32,
        )
        warped = cv2.perspectiveTransform(corners, matrix)[0]
        nx1 = float(warped[:, 0].min())
        ny1 = float(warped[:, 1].min())
        nx2 = float(warped[:, 0].max())
        ny2 = float(warped[:, 1].max())
        clipped = _clip_xyxy([nx1, ny1, nx2, ny2], image_w, image_h)
        bw = clipped[2] - clipped[0]
        bh = clipped[3] - clipped[1]
        if bw >= 2 and bh >= 2:
            transformed.append({"class_id": row["class_id"], "bbox_xyxy": clipped})

    return transformed


def _apply_spatial_augment(
    image_bgr: np.ndarray,
    boxes: list[dict[str, Any]],
    rng: np.random.Generator,
    aug_cfg: dict[str, float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    image_h, image_w = image_bgr.shape[:2]
    matrix = _random_perspective_matrix(
        image_w=image_w,
        image_h=image_h,
        rng=rng,
        degrees=float(aug_cfg.get("degrees", 0.0)),
        shear=float(aug_cfg.get("shear", 0.0)),
        perspective=float(aug_cfg.get("perspective", 0.0)),
    )
    augmented = cv2.warpPerspective(
        image_bgr,
        matrix,
        dsize=(image_w, image_h),
        flags=cv2.INTER_LINEAR,
        borderValue=(114, 114, 114),
    )
    augmented_boxes = _transform_boxes(boxes, matrix, image_w, image_h)

    if rng.random() < float(aug_cfg.get("flipud", 0.0)):
        augmented = cv2.flip(augmented, 0)
        flipped_boxes: list[dict[str, Any]] = []
        for row in augmented_boxes:
            x1, y1, x2, y2 = row["bbox_xyxy"]
            flipped_boxes.append(
                {
                    "class_id": row["class_id"],
                    "bbox_xyxy": _clip_xyxy([x1, image_h - y2, x2, image_h - y1], image_w, image_h),
                }
            )
        augmented_boxes = flipped_boxes

    return augmented, augmented_boxes


def _resize_image_and_boxes(
    image_bgr: np.ndarray,
    boxes: list[dict[str, Any]],
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    image_h, image_w = image_bgr.shape[:2]
    if image_w == target_w and image_h == target_h:
        return image_bgr, boxes
    resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    scale_x = target_w / max(1, image_w)
    scale_y = target_h / max(1, image_h)
    resized_boxes = []
    for row in boxes:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        resized_boxes.append(
            {
                "class_id": row["class_id"],
                "bbox_xyxy": [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
            }
        )
    return resized, resized_boxes


def _box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _clip_to_patch(box: list[float], patch: tuple[int, int, int, int], image_w: int, image_h: int) -> list[float]:
    px1, py1, px2, py2 = patch
    x1, y1, x2, y2 = box
    return _clip_xyxy([max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2)], image_w, image_h)


def _load_augmented_pair(
    item: LabeledImage,
    rng: np.random.Generator,
    aug_cfg: dict[str, float],
) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    image_bgr = cv2.imread(str(item.image_path))
    if image_bgr is None:
        return None
    return _apply_spatial_augment(image_bgr, item.boxes, rng, aug_cfg)


def _apply_mixup(
    image_bgr: np.ndarray,
    boxes: list[dict[str, Any]],
    donor_items: list[LabeledImage],
    rng: np.random.Generator,
    aug_cfg: dict[str, float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not donor_items or rng.random() >= float(aug_cfg.get("mixup", 0.0)):
        return image_bgr, boxes

    donor = donor_items[int(rng.integers(0, len(donor_items)))]
    donor_pair = _load_augmented_pair(donor, rng, aug_cfg)
    if donor_pair is None:
        return image_bgr, boxes

    donor_image, donor_boxes = donor_pair
    image_h, image_w = image_bgr.shape[:2]
    donor_image, donor_boxes = _resize_image_and_boxes(donor_image, donor_boxes, image_w, image_h)
    ratio = float(rng.beta(32.0, 32.0))
    mixed = cv2.addWeighted(image_bgr, ratio, donor_image, 1.0 - ratio, 0)
    return mixed, boxes + donor_boxes


def _apply_cutmix(
    image_bgr: np.ndarray,
    boxes: list[dict[str, Any]],
    donor_items: list[LabeledImage],
    rng: np.random.Generator,
    aug_cfg: dict[str, float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not donor_items or rng.random() >= float(aug_cfg.get("cutmix", 0.0)):
        return image_bgr, boxes

    donor = donor_items[int(rng.integers(0, len(donor_items)))]
    donor_pair = _load_augmented_pair(donor, rng, aug_cfg)
    if donor_pair is None:
        return image_bgr, boxes

    donor_image, donor_boxes = donor_pair
    image_h, image_w = image_bgr.shape[:2]
    donor_image, donor_boxes = _resize_image_and_boxes(donor_image, donor_boxes, image_w, image_h)

    patch_w = int(rng.uniform(0.25, 0.55) * image_w)
    patch_h = int(rng.uniform(0.25, 0.55) * image_h)
    px1 = int(rng.integers(0, max(1, image_w - patch_w)))
    py1 = int(rng.integers(0, max(1, image_h - patch_h)))
    px2 = min(image_w, px1 + patch_w)
    py2 = min(image_h, py1 + patch_h)
    patch = (px1, py1, px2, py2)

    image_bgr[py1:py2, px1:px2] = donor_image[py1:py2, px1:px2]

    kept_boxes = []
    for row in boxes:
        cx, cy = _box_center(row["bbox_xyxy"])
        if not (px1 <= cx <= px2 and py1 <= cy <= py2):
            kept_boxes.append(row)

    patch_boxes = []
    for row in donor_boxes:
        cx, cy = _box_center(row["bbox_xyxy"])
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            clipped = _clip_to_patch(row["bbox_xyxy"], patch, image_w, image_h)
            if clipped[2] - clipped[0] >= 2 and clipped[3] - clipped[1] >= 2:
                patch_boxes.append({"class_id": row["class_id"], "bbox_xyxy": clipped})

    return image_bgr, kept_boxes + patch_boxes


def _apply_copy_paste(
    image_bgr: np.ndarray,
    boxes: list[dict[str, Any]],
    donor_items: list[LabeledImage],
    rng: np.random.Generator,
    aug_cfg: dict[str, float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not donor_items or rng.random() >= float(aug_cfg.get("copy_paste", 0.0)):
        return image_bgr, boxes

    donor = donor_items[int(rng.integers(0, len(donor_items)))]
    donor_pair = _load_augmented_pair(donor, rng, aug_cfg)
    if donor_pair is None:
        return image_bgr, boxes

    donor_image, donor_boxes = donor_pair
    image_h, image_w = image_bgr.shape[:2]
    donor_image, donor_boxes = _resize_image_and_boxes(donor_image, donor_boxes, image_w, image_h)
    if not donor_boxes:
        return image_bgr, boxes

    new_boxes = list(boxes)
    paste_count = int(rng.integers(1, min(3, len(donor_boxes)) + 1))
    chosen = rng.choice(len(donor_boxes), size=paste_count, replace=False)

    for index in np.atleast_1d(chosen):
        row = donor_boxes[int(index)]
        x1, y1, x2, y2 = [int(round(v)) for v in row["bbox_xyxy"]]
        crop_w = max(2, x2 - x1)
        crop_h = max(2, y2 - y1)
        if x1 < 0 or y1 < 0 or x2 > image_w or y2 > image_h:
            continue
        crop = donor_image[y1:y2, x1:x2]
        if crop.size == 0 or crop_w >= image_w or crop_h >= image_h:
            continue

        dst_x1 = int(rng.integers(0, max(1, image_w - crop_w)))
        dst_y1 = int(rng.integers(0, max(1, image_h - crop_h)))
        dst_x2 = dst_x1 + crop_w
        dst_y2 = dst_y1 + crop_h
        image_bgr[dst_y1:dst_y2, dst_x1:dst_x2] = crop
        new_boxes.append({"class_id": row["class_id"], "bbox_xyxy": [dst_x1, dst_y1, dst_x2, dst_y2]})

    return image_bgr, new_boxes


def _collect_labeled_images(
    images_dir: Path,
    labels_dir: Path,
    limit: int,
    rng: random.Random,
) -> list[LabeledImage]:
    items: list[LabeledImage] = []
    image_paths = [path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS]
    rng.shuffle(image_paths)

    for image_path in image_paths:
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_h, image_w = image.shape[:2]
        boxes = _read_boxes(label_path, image_w, image_h)
        if boxes:
            items.append(LabeledImage(image_path=image_path, label_path=label_path, boxes=boxes))
        if len(items) >= limit:
            break
    return items


def _save_original_boxed_preview(
    item: LabeledImage,
    output_path: Path,
    class_names: list[str],
) -> bool:
    image_bgr = cv2.imread(str(item.image_path))
    if image_bgr is None:
        return False
    cv2.imwrite(str(output_path), _draw_boxes(image_bgr, item.boxes, class_names))
    return True


def generate_augmented_bbox_previews(
    config_path: Path,
    output_root: Path,
    source_images: Path | None,
    source_labels: Path | None,
    dataset_yaml: Path | None,
    base_count: int,
    augment_count: int,
    seed: int,
) -> dict[str, Any]:
    repo_root = _repo_root()
    config = _load_yaml(config_path)
    paths = config.get("paths", {})
    detection = config.get("detection", {})

    images_dir = source_images or _resolve_path(paths.get("source_images", "dataset/train/images"), repo_root)
    labels_dir = source_labels or _resolve_path(paths.get("source_labels", "dataset/train/labels"), repo_root)
    dataset_yaml_path = dataset_yaml or _resolve_path(paths.get("dataset_yaml", "dataset/data.yaml"), repo_root)

    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_boxed = output_root / "boxed_augmented"
    output_original_boxed = output_root / "boxed_original"
    for directory in (output_images, output_labels, output_boxed, output_original_boxed):
        directory.mkdir(parents=True, exist_ok=True)

    class_names = _load_class_names(dataset_yaml_path)
    py_rng = random.Random(seed)
    items = _collect_labeled_images(
        images_dir=images_dir,
        labels_dir=labels_dir,
        limit=max(base_count, augment_count),
        rng=py_rng,
    )
    if not items:
        raise RuntimeError(f"No labeled images found in {images_dir} with labels in {labels_dir}")

    selected = items[: min(base_count, len(items))]
    if not selected:
        raise RuntimeError("No source images selected for augmentation")

    aug_cfg = {
        "degrees": float(detection.get("degrees", 0.0)),
        "shear": float(detection.get("shear", 0.0)),
        "perspective": float(detection.get("perspective", 0.0)),
        "flipud": float(detection.get("flipud", 0.0)),
        "mixup": float(detection.get("mixup", 0.0)),
        "cutmix": float(detection.get("cutmix", 0.0)),
        "copy_paste": float(detection.get("copy_paste", 0.0)),
    }
    np_rng = np.random.default_rng(seed)

    generated = 0
    attempts = 0
    max_attempts = max(augment_count * 5, augment_count + 20)
    manifest_rows = ["augmented_image,augmented_boxed_preview,original_image,original_boxed_preview,label_file"]
    progress = tqdm(total=augment_count, desc="Augmented boxed images")

    while generated < augment_count and attempts < max_attempts:
        attempts += 1
        item = selected[generated % len(selected)]
        pair = _load_augmented_pair(item, np_rng, aug_cfg)
        if pair is None:
            continue
        augmented, boxes = pair
        augmented, boxes = _apply_mixup(augmented, boxes, selected, np_rng, aug_cfg)
        augmented, boxes = _apply_cutmix(augmented, boxes, selected, np_rng, aug_cfg)
        augmented, boxes = _apply_copy_paste(augmented, boxes, selected, np_rng, aug_cfg)
        if not boxes:
            continue

        image_h, image_w = augmented.shape[:2]
        stem = f"{item.image_path.stem}_aug_{generated + 1:04d}"
        image_out = output_images / f"{stem}.jpg"
        label_out = output_labels / f"{stem}.txt"
        boxed_out = output_boxed / f"{stem}_boxed.jpg"
        original_boxed_out = output_original_boxed / f"{stem}_original_boxed.jpg"

        cv2.imwrite(str(image_out), augmented)
        _write_yolo_labels(label_out, boxes, image_w, image_h)
        cv2.imwrite(str(boxed_out), _draw_boxes(augmented, boxes, class_names))
        _save_original_boxed_preview(item, original_boxed_out, class_names)
        manifest_rows.append(
            ",".join(
                [
                    image_out.as_posix(),
                    boxed_out.as_posix(),
                    item.image_path.as_posix(),
                    original_boxed_out.as_posix(),
                    label_out.as_posix(),
                ]
            )
        )

        generated += 1
        progress.update(1)

    progress.close()

    selected_path = output_root / "selected_source_images.txt"
    selected_path.write_text(
        "\n".join(str(item.image_path.relative_to(repo_root)) for item in selected) + "\n",
        encoding="utf-8",
    )
    (output_root / "preview_pairs.csv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    summary = {
        "source_images": str(images_dir),
        "source_labels": str(labels_dir),
        "dataset_yaml": str(dataset_yaml_path),
        "selected_source_count": len(selected),
        "requested_augmented_count": augment_count,
        "generated_augmented_count": generated,
        "augmentation_config": aug_cfg,
        "output_images": str(output_images),
        "output_labels": str(output_labels),
        "output_boxed_augmented": str(output_boxed),
        "output_boxed_original": str(output_original_boxed),
        "preview_pairs": str(output_root / "preview_pairs.csv"),
    }
    summary_path = output_root / "summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate augmented YOLO images, transformed labels, and boxed preview images "
            "for the augmented outputs."
        )
    )
    parser.add_argument("--config", default="configs/pipeline.yaml", type=Path)
    parser.add_argument("--output-root", default="artifacts/augmented_bbox_previews", type=Path)
    parser.add_argument("--source-images", type=Path, default=None)
    parser.add_argument("--source-labels", type=Path, default=None)
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--base-count", type=int, default=200)
    parser.add_argument("--augment-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    config_path = _resolve_path(args.config, repo_root)
    output_root = _resolve_path(args.output_root, repo_root)
    source_images = _resolve_path(args.source_images, repo_root) if args.source_images else None
    source_labels = _resolve_path(args.source_labels, repo_root) if args.source_labels else None
    dataset_yaml = _resolve_path(args.dataset_yaml, repo_root) if args.dataset_yaml else None

    summary = generate_augmented_bbox_previews(
        config_path=config_path,
        output_root=output_root,
        source_images=source_images,
        source_labels=source_labels,
        dataset_yaml=dataset_yaml,
        base_count=max(1, args.base_count),
        augment_count=max(1, args.augment_count),
        seed=args.seed,
    )
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
