from __future__ import annotations

from typing import Any


def _first_metric(metrics: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in metrics:
            try:
                return float(metrics[key])
            except Exception:
                return None
    return None


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.4f}"


def attach_epoch_progress_callback(model: Any, task_name: str) -> None:
    if getattr(model, "_plant_epoch_callback_attached", False):
        return

    def on_fit_epoch_end(trainer: Any) -> None:
        rank = getattr(trainer, "rank", -1)
        if rank not in (-1, 0):
            return

        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", 0))
        metrics = getattr(trainer, "metrics", {}) or {}

        train_losses: dict[str, float] = {}
        try:
            train_losses = trainer.label_loss_items(trainer.tloss, prefix="train") or {}
        except Exception:
            train_losses = {}

        box_loss = _first_metric(train_losses, ["train/box_loss"])
        cls_loss = _first_metric(train_losses, ["train/cls_loss"])
        dfl_loss = _first_metric(train_losses, ["train/dfl_loss"])

        precision = _first_metric(metrics, ["metrics/precision(B)", "metrics/precision(M)"])
        recall = _first_metric(metrics, ["metrics/recall(B)", "metrics/recall(M)"])
        map50 = _first_metric(metrics, ["metrics/mAP50(B)", "metrics/mAP50(M)"])
        map5095 = _first_metric(metrics, ["metrics/mAP50-95(B)", "metrics/mAP50-95(M)"])

        val_box = _first_metric(metrics, ["val/box_loss"])
        val_cls = _first_metric(metrics, ["val/cls_loss"])
        val_dfl = _first_metric(metrics, ["val/dfl_loss"])

        print(
            f"[{task_name}] epoch {epoch}/{total_epochs} "
            f"train(box={_fmt(box_loss)}, cls={_fmt(cls_loss)}, dfl={_fmt(dfl_loss)}) "
            f"val(P={_fmt(precision)}, R={_fmt(recall)}, mAP50={_fmt(map50)}, mAP50-95={_fmt(map5095)}, "
            f"box={_fmt(val_box)}, cls={_fmt(val_cls)}, dfl={_fmt(val_dfl)})",
            flush=True,
        )

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    setattr(model, "_plant_epoch_callback_attached", True)

