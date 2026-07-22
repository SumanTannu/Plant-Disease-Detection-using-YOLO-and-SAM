from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from ultralytics import SAM
except Exception as exc:  # pragma: no cover - runtime dependency
    SAM = None
    _SAM_IMPORT_ERROR = exc
else:
    _SAM_IMPORT_ERROR = None


class SAM3MaskGenerator:
    def __init__(self, model_path: str | Path, conf_threshold: float = 0.25):
        if SAM is None:
            raise ImportError(
                "ultralytics is required for SAM operations. Install requirements.txt first."
            ) from _SAM_IMPORT_ERROR

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"SAM weight file not found: {model_path}. "
                "Download sam3.pt manually from the official SAM 3 model page and place it here."
            )

        self.model_path = str(model_path)
        self.conf_threshold = float(conf_threshold)
        self.model = SAM(self.model_path)

    def _extract_scores(self, result: Any, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0,), dtype=np.float32)

        if getattr(result, "boxes", None) is not None:
            conf = getattr(result.boxes, "conf", None)
            if conf is not None:
                scores = conf.cpu().numpy().astype(np.float32)
                if len(scores) == count:
                    return scores

        masks_obj = getattr(result, "masks", None)
        if masks_obj is not None:
            conf = getattr(masks_obj, "conf", None)
            if conf is not None:
                scores = np.asarray(conf, dtype=np.float32)
                if len(scores) == count:
                    return scores

        return np.ones((count,), dtype=np.float32)

    def generate_best_mask(
        self,
        image_path: str | Path,
        bbox_xyxy: list[float],
    ) -> tuple[np.ndarray | None, float]:
        results = self.model.predict(
            source=str(image_path),
            bboxes=[list(map(float, bbox_xyxy))],
            conf=self.conf_threshold,
            verbose=False,
        )
        if not results:
            return None, 0.0

        result = results[0]
        if result.masks is None or result.masks.data is None:
            return None, 0.0

        masks = result.masks.data.cpu().numpy()
        if len(masks) == 0:
            return None, 0.0

        scores = self._extract_scores(result, len(masks))
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx] > 0.5
        best_score = float(scores[best_idx])
        return best_mask, best_score

