from __future__ import annotations

from pathlib import Path
from typing import Any

from .training_logger import attach_epoch_progress_callback
from .train_safety import train_with_auto_fallback

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - runtime dependency
    YOLO = None
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None


class YOLOv12Detector:
    def __init__(self, weights: str | Path):
        if YOLO is None:
            raise ImportError(
                "ultralytics is required for YOLO operations. Install requirements.txt first."
            ) from _YOLO_IMPORT_ERROR
        self.weights = str(weights)
        self.model = YOLO(self.weights)

    def train(self, data_yaml: str | Path, cfg: dict[str, Any]) -> Path:
        if bool(cfg.get("print_epoch_metrics", True)):
            attach_epoch_progress_callback(self.model, task_name="detector")

        train_result = train_with_auto_fallback(
            model=self.model,
            data_yaml=str(data_yaml),
            cfg=cfg,
            task_name="detector",
        )
        save_dir = getattr(train_result, "save_dir", None)
        if save_dir is None and getattr(self.model, "trainer", None) is not None:
            save_dir = getattr(self.model.trainer, "save_dir", None)
        if save_dir is None:
            save_dir = Path(cfg["project"]) / cfg["name"]
        return Path(save_dir)

    def predict(self, image_path: str | Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=str(image_path),
            conf=float(cfg["conf_threshold"]),
            iou=float(cfg["iou_threshold"]),
            max_det=int(cfg["max_det"]),
            device=str(cfg["device"]),
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()

        detections: list[dict[str, Any]] = []
        for box, class_id, score in zip(xyxy, cls, conf):
            detections.append(
                {
                    "class_id": int(class_id),
                    "confidence": float(score),
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                }
            )
        return detections

    def predict_batch(
        self,
        image_paths: list[str | Path],
        cfg: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        results = self.model.predict(
            source=[str(p) for p in image_paths],
            conf=float(cfg["conf_threshold"]),
            iou=float(cfg["iou_threshold"]),
            max_det=int(cfg["max_det"]),
            device=str(cfg["device"]),
            verbose=False,
        )

        out: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            path = str(result.path)
            if result.boxes is None or len(result.boxes) == 0:
                out[path] = []
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()
            rows: list[dict[str, Any]] = []
            for box, class_id, score in zip(xyxy, cls, conf):
                rows.append(
                    {
                        "class_id": int(class_id),
                        "confidence": float(score),
                        "bbox_xyxy": [float(v) for v in box.tolist()],
                    }
                )
            out[path] = rows
        return out


def pick_best_weights(train_dir: str | Path) -> Path:
    train_dir = Path(train_dir)
    best = train_dir / "weights" / "best.pt"
    last = train_dir / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"No weights found in: {train_dir}")
