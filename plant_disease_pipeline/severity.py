from __future__ import annotations

from typing import Iterable

import numpy as np

from .utils.geometry import bbox_area


def compute_severity(
    lesion_mask: np.ndarray,
    leaf_mask: np.ndarray | None = None,
    fallback_leaf_box: Iterable[float] | None = None,
    min_leaf_area: float = 1.0,
) -> dict[str, float]:
    lesion_area = float(np.count_nonzero(lesion_mask))

    if leaf_mask is not None and np.count_nonzero(leaf_mask) > 0:
        leaf_area = float(np.count_nonzero(leaf_mask))
    elif fallback_leaf_box is not None:
        leaf_area = float(bbox_area(fallback_leaf_box))
    else:
        leaf_area = max(float(min_leaf_area), lesion_area)

    leaf_area = max(leaf_area, float(min_leaf_area))
    severity = (lesion_area / leaf_area) * 100.0
    severity = float(np.clip(severity, 0.0, 100.0))

    return {
        "lesion_area": lesion_area,
        "leaf_area": leaf_area,
        "severity_percent": severity,
    }

