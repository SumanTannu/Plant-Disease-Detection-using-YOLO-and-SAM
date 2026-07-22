from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def class_color(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id + 1337)
    bgr = rng.integers(40, 255, size=3, endpoint=True).tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_inference_result(
    image_bgr: np.ndarray,
    rows: list[dict[str, Any]],
    class_names: list[str],
) -> np.ndarray:
    canvas = image_bgr.copy()
    overlay = image_bgr.copy()

    for row in rows:
        cls_id = int(row["class_id"])
        color = class_color(cls_id)
        x1, y1, x2, y2 = [int(round(v)) for v in row["bbox_xyxy"]]

        lesion_mask = row.get("lesion_mask")
        leaf_mask = row.get("leaf_mask")

        if isinstance(leaf_mask, np.ndarray):
            overlay[leaf_mask.astype(bool)] = (80, 180, 80)
        if isinstance(lesion_mask, np.ndarray):
            overlay[lesion_mask.astype(bool)] = color

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        class_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
        label = (
            f"{class_name} "
            f"det={row['det_conf']:.2f} "
            f"sev={row['severity_percent']:.1f}%"
        )
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    alpha = 0.35
    composed = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    return composed

