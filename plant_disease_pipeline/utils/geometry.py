from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def yolo_xywh_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_w: int,
    image_h: int,
) -> list[float]:
    x1 = (x_center - width / 2.0) * image_w
    y1 = (y_center - height / 2.0) * image_h
    x2 = (x_center + width / 2.0) * image_w
    y2 = (y_center + height / 2.0) * image_h
    return [x1, y1, x2, y2]


def clip_box_xyxy(box: Iterable[float], image_w: int, image_h: int) -> list[float]:
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


def bbox_area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = box
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    return w * h


def box_iou_xyxy(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    union = bbox_area(box_a) + bbox_area(box_b) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def polygon_from_bbox(box: Iterable[float], image_w: int, image_h: int) -> list[float]:
    x1, y1, x2, y2 = box
    coords = [
        x1 / image_w,
        y1 / image_h,
        x2 / image_w,
        y1 / image_h,
        x2 / image_w,
        y2 / image_h,
        x1 / image_w,
        y2 / image_h,
    ]
    return [float(np.clip(v, 0.0, 1.0)) for v in coords]


def mask_to_polygon(mask: np.ndarray, eps_ratio: float = 0.002) -> list[float] | None:
    mask_u8 = (mask.astype(np.uint8) * 255).copy()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 5.0:
        return None

    epsilon = eps_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if approx is None or len(approx) < 3:
        return None

    h, w = mask.shape[:2]
    points = approx.reshape(-1, 2).astype(np.float32)
    points[:, 0] = points[:, 0] / max(1.0, float(w))
    points[:, 1] = points[:, 1] / max(1.0, float(h))
    points = np.clip(points, 0.0, 1.0)
    flat = points.reshape(-1).tolist()

    if len(flat) < 6:
        return None
    return [float(v) for v in flat]


def binary_mask_from_bbox(box: Iterable[float], image_w: int, image_h: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(0, min(x2, image_w))
    y2 = max(0, min(y2, image_h))
    mask = np.zeros((image_h, image_w), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask

