from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from ..utils.fs import list_images, read_yaml, write_lines, write_yaml


def extract_class_names(dataset_yaml_path: str | Path) -> list[str]:
    data = read_yaml(dataset_yaml_path)
    names = data.get("names", [])
    if isinstance(names, dict):
        # Roboflow sometimes stores class map as {0: name0, 1: name1, ...}
        return [str(names[k]) for k in sorted(names.keys(), key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(x) for x in names]
    raise ValueError("Invalid 'names' format in dataset YAML.")


def _class_count_to_named(counts: dict[int, int], class_names: list[str]) -> dict[str, int]:
    return {
        class_names[i] if 0 <= i < len(class_names) else f"class_{i}": int(v)
        for i, v in sorted(counts.items(), key=lambda kv: kv[0])
    }


def _validate_and_clean_label_file(
    label_path: Path,
    num_classes: int,
    auto_fix_labels: bool,
) -> tuple[list[str], Counter, int]:
    cleaned_lines: list[str] = []
    class_counts: Counter = Counter()
    invalid_line_count = 0

    if not label_path.exists():
        return cleaned_lines, class_counts, invalid_line_count

    lines = label_path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            invalid_line_count += 1
            continue

        try:
            class_id = int(float(parts[0]))
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except Exception:
            invalid_line_count += 1
            continue

        if class_id < 0 or class_id >= num_classes:
            invalid_line_count += 1
            continue

        if auto_fix_labels:
            x_c = min(1.0, max(0.0, x_c))
            y_c = min(1.0, max(0.0, y_c))
            w = min(1.0, max(0.0, w))
            h = min(1.0, max(0.0, h))

        if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            invalid_line_count += 1
            continue

        class_counts[class_id] += 1
        cleaned_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    if auto_fix_labels:
        new_text = "\n".join(cleaned_lines) + ("\n" if cleaned_lines else "")
        old_text = "\n".join([ln.strip() for ln in lines if ln.strip()]) + ("\n" if lines else "")
        if new_text != old_text:
            label_path.write_text(new_text, encoding="utf-8")

    return cleaned_lines, class_counts, invalid_line_count


def _sum_class_counts(images: list[Path], image_class_counts: dict[Path, Counter]) -> Counter:
    total: Counter = Counter()
    for image in images:
        total.update(image_class_counts.get(image, Counter()))
    return total


def _ensure_train_class_coverage(
    train_images: list[Path],
    val_images: list[Path],
    image_class_counts: dict[Path, Counter],
    num_classes: int,
) -> tuple[list[Path], list[Path], list[int]]:
    train_set = set(train_images)
    val_list = val_images[:]
    moved_for_classes: list[int] = []

    def current_present_classes(images: list[Path]) -> set[int]:
        present: set[int] = set()
        for img in images:
            present.update(image_class_counts.get(img, Counter()).keys())
        return present

    present_in_train = current_present_classes(train_images)
    missing = [c for c in range(num_classes) if c not in present_in_train]
    if not missing:
        return train_images, val_images, moved_for_classes

    for class_id in missing:
        candidate = None
        for img in val_list:
            if class_id in image_class_counts.get(img, Counter()):
                candidate = img
                break
        if candidate is None:
            continue
        val_list.remove(candidate)
        train_images.append(candidate)
        train_set.add(candidate)
        moved_for_classes.append(class_id)

    return train_images, val_list, moved_for_classes


def _oversample_train_images(
    train_images: list[Path],
    image_class_counts: dict[Path, Counter],
    target_ratio: float,
    max_repeat: int,
    seed: int,
) -> tuple[list[Path], dict[Path, int], Counter, int]:
    if not train_images:
        return [], {}, Counter(), 0

    base_counts = _sum_class_counts(train_images, image_class_counts)
    if not base_counts:
        repeat_map = {img: 1 for img in train_images}
        return train_images[:], repeat_map, Counter(), 0

    max_count = max(base_counts.values())
    target = max(1, int(max_count * target_ratio))

    class_boost: dict[int, int] = {}
    for class_id, count in base_counts.items():
        if count >= target:
            class_boost[class_id] = 1
        else:
            boost = math.ceil(target / max(1, count))
            class_boost[class_id] = min(max_repeat, max(1, boost))

    repeat_map: dict[Path, int] = {}
    for image in train_images:
        cls_counts = image_class_counts.get(image, Counter())
        image_classes = list(cls_counts.keys())
        if not image_classes:
            repeat_map[image] = 1
            continue
        repeat_map[image] = max(class_boost.get(cid, 1) for cid in image_classes)

    balanced: list[Path] = []
    for image in train_images:
        balanced.extend([image] * repeat_map[image])

    rng = random.Random(seed)
    rng.shuffle(balanced)

    balanced_counts = _sum_class_counts(balanced, image_class_counts)
    extra_samples = max(0, len(balanced) - len(train_images))
    return balanced, repeat_map, balanced_counts, extra_samples


def build_detection_split(
    source_images_dir: str | Path,
    source_labels_dir: str | Path,
    class_names: list[str],
    val_ratio: float,
    seed: int,
    output_dir: str | Path,
    validate_images: bool = True,
    auto_fix_labels: bool = True,
    oversample_minority: bool = True,
    imbalance_target_ratio: float = 0.6,
    imbalance_max_repeat: int = 4,
) -> dict[str, Any]:
    source_images = Path(source_images_dir)
    source_labels = Path(source_labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(source_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {source_images}")

    num_classes = len(class_names)
    valid_pairs: list[Path] = []
    missing_labels: list[Path] = []
    unreadable_images: list[Path] = []
    invalid_label_files: list[Path] = []

    image_class_counts: dict[Path, Counter] = {}
    total_invalid_lines = 0

    for image_path in image_paths:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path)
            continue

        if validate_images:
            img = cv2.imread(str(image_path))
            if img is None:
                unreadable_images.append(image_path)
                continue

        _, class_counts, invalid_lines = _validate_and_clean_label_file(
            label_path=label_path,
            num_classes=num_classes,
            auto_fix_labels=auto_fix_labels,
        )
        total_invalid_lines += int(invalid_lines)
        if invalid_lines > 0:
            invalid_label_files.append(label_path)

        image_class_counts[image_path] = class_counts
        valid_pairs.append(image_path)

    if not valid_pairs:
        raise FileNotFoundError(
            f"No image-label pairs found in {source_images} and {source_labels}."
        )

    rng = random.Random(seed)
    rng.shuffle(valid_pairs)

    val_count = max(1, int(len(valid_pairs) * val_ratio))
    if val_count >= len(valid_pairs):
        val_count = max(1, len(valid_pairs) - 1)

    val_images = valid_pairs[:val_count]
    train_images = valid_pairs[val_count:]
    train_images, val_images, moved_class_ids = _ensure_train_class_coverage(
        train_images=train_images,
        val_images=val_images,
        image_class_counts=image_class_counts,
        num_classes=num_classes,
    )

    train_counts_before = _sum_class_counts(train_images, image_class_counts)
    val_counts = _sum_class_counts(val_images, image_class_counts)

    balanced_train_images = train_images
    repeat_map: dict[Path, int] = {img: 1 for img in train_images}
    train_counts_after = train_counts_before
    extra_samples = 0

    if oversample_minority:
        (
            balanced_train_images,
            repeat_map,
            train_counts_after,
            extra_samples,
        ) = _oversample_train_images(
            train_images=train_images,
            image_class_counts=image_class_counts,
            target_ratio=float(imbalance_target_ratio),
            max_repeat=int(imbalance_max_repeat),
            seed=seed,
        )

    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"
    train_raw_txt = output_dir / "train_raw.txt"
    detection_data_yaml = output_dir / "detection_data.yaml"
    split_report_yaml = output_dir / "split_report.yaml"
    class_distribution_yaml = output_dir / "class_distribution.yaml"

    write_lines(train_raw_txt, [str(p.resolve()) for p in train_images])
    write_lines(train_txt, [str(p.resolve()) for p in balanced_train_images])
    write_lines(val_txt, [str(p.resolve()) for p in val_images])

    write_yaml(
        detection_data_yaml,
        {
            "train": str(train_txt.resolve()),
            "val": str(val_txt.resolve()),
            "nc": len(class_names),
            "names": class_names,
        },
    )

    class_distribution = {
        "train_before_oversample": _class_count_to_named(dict(train_counts_before), class_names),
        "train_after_oversample": _class_count_to_named(dict(train_counts_after), class_names),
        "val": _class_count_to_named(dict(val_counts), class_names),
    }
    write_yaml(class_distribution_yaml, class_distribution)

    repeated_examples = sorted(
        [
            (str(img.name), rep)
            for img, rep in repeat_map.items()
            if rep > 1
        ],
        key=lambda x: x[1],
        reverse=True,
    )[:30]

    write_yaml(
        split_report_yaml,
        {
            "total_images": len(image_paths),
            "paired_images": len(valid_pairs),
            "train_images_unique": len(train_images),
            "train_images_written": len(balanced_train_images),
            "val_images": len(val_images),
            "missing_label_count": len(missing_labels),
            "missing_label_examples": [str(p.name) for p in missing_labels[:20]],
            "unreadable_image_count": len(unreadable_images),
            "unreadable_image_examples": [str(p.name) for p in unreadable_images[:20]],
            "invalid_label_file_count": len(set(invalid_label_files)),
            "invalid_label_file_examples": [str(p.name) for p in invalid_label_files[:20]],
            "invalid_label_line_count": int(total_invalid_lines),
            "auto_fix_labels": bool(auto_fix_labels),
            "oversample_minority": bool(oversample_minority),
            "imbalance_target_ratio": float(imbalance_target_ratio),
            "imbalance_max_repeat": int(imbalance_max_repeat),
            "oversample_extra_samples": int(extra_samples),
            "repeated_train_images_examples": repeated_examples,
            "classes_moved_from_val_to_train": [class_names[cid] for cid in moved_class_ids],
        },
    )

    return {
        "train_txt": str(train_txt.resolve()),
        "train_raw_txt": str(train_raw_txt.resolve()),
        "val_txt": str(val_txt.resolve()),
        "detection_data_yaml": str(detection_data_yaml.resolve()),
        "split_report_yaml": str(split_report_yaml.resolve()),
        "class_distribution_yaml": str(class_distribution_yaml.resolve()),
    }
