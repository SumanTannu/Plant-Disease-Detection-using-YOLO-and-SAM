from __future__ import annotations

import gc
import csv
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None


def is_cuda_memory_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    keywords = (
        "out of memory",
        "cuda error",
        "cublas_status_execution_failed",
        "cudnn",
        "memory allocation failure",
        "cuda out of memory",
    )
    return any(k in msg for k in keywords)


def clear_cuda_cache() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


def _find_auto_resume_checkpoint(project: str, run_name: str) -> Path | None:
    direct = Path(project) / run_name / "weights" / "last.pt"
    candidates: list[Path] = []
    if direct.exists():
        candidates.append(direct)

    runs_root = Path("runs")
    if runs_root.exists():
        candidates.extend(runs_root.rglob(f"{run_name}/weights/last.pt"))

    if not candidates:
        return None

    # Prefer most recently updated checkpoint.
    candidates = sorted(
        {c.resolve() for c in candidates if c.exists()},
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _checkpoint_finished_for_target_epochs(checkpoint: Path, target_epochs: int) -> bool:
    # Prefer results.csv because some exported checkpoints may not carry full optimizer/epoch metadata.
    try:
        run_dir = checkpoint.parent.parent  # .../<run_name>/weights/last.pt -> .../<run_name>
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            with results_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                last_epoch = None
                for row in reader:
                    if "epoch" not in row:
                        continue
                    try:
                        last_epoch = int(float(row["epoch"]))
                    except Exception:
                        continue
                if last_epoch is not None:
                    # results.csv in this project is 1-based (epoch starts from 1).
                    return int(last_epoch) >= int(target_epochs)
    except Exception:
        pass

    if torch is None or not checkpoint.exists():
        return False
    try:
        ckpt = torch.load(str(checkpoint), map_location="cpu")
    except Exception:
        return False

    if not isinstance(ckpt, dict):
        return False

    epoch = ckpt.get("epoch", None)
    if epoch is None:
        return False
    try:
        completed_epochs = int(epoch) + 1
    except Exception:
        return False
    return completed_epochs >= int(target_epochs)


def _build_attempts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "imgsz": int(cfg["imgsz"]),
        "batch": int(cfg["batch"]),
        "workers": int(cfg["workers"]),
        "amp": bool(cfg.get("amp", True)),
        "device": str(cfg["device"]),
    }

    attempts = [base]
    if not bool(cfg.get("auto_safe_mode", True)):
        return attempts

    # Progressive memory-safe fallback schedule.
    fallback_shapes = [512, 448, 384, 320]
    for target_imgsz in fallback_shapes:
        attempts.append(
            {
                "imgsz": min(base["imgsz"], target_imgsz),
                "batch": max(1, min(base["batch"] // 2, 8)),
                "workers": min(base["workers"], 4),
                "amp": False,
                "device": base["device"],
            }
        )
        base["batch"] = max(1, base["batch"] // 2)

    if bool(cfg.get("cpu_fallback", False)):
        attempts.append(
            {
                "imgsz": min(int(cfg["imgsz"]), 320),
                "batch": 1,
                "workers": 0,
                "amp": False,
                "device": "cpu",
            }
        )

    # Deduplicate identical attempts in order.
    unique: list[dict[str, Any]] = []
    for a in attempts:
        if a not in unique:
            unique.append(a)
    return unique


def train_with_auto_fallback(model: Any, data_yaml: str, cfg: dict[str, Any], task_name: str) -> Any:
    resume_arg: str | bool = False
    resume_source = "none"
    manual_resume = str(cfg.get("resume_checkpoint", "")).strip()
    if manual_resume:
        resume_path = Path(manual_resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"{task_name}: resume checkpoint not found: {resume_path}")
        if _checkpoint_finished_for_target_epochs(resume_path, int(cfg["epochs"])):
            print(
                f"[{task_name}] Resume checkpoint already reached target epochs ({cfg['epochs']}). "
                "Starting a new training run instead of resuming.",
                flush=True,
            )
        else:
            resume_arg = str(resume_path.resolve())
            resume_source = "manual"
            print(f"[{task_name}] Resuming from checkpoint: {resume_arg}")
    elif bool(cfg.get("auto_resume", True)):
        auto_last = _find_auto_resume_checkpoint(str(cfg["project"]), str(cfg["name"]))
        if auto_last is not None and auto_last.exists():
            if _checkpoint_finished_for_target_epochs(auto_last, int(cfg["epochs"])):
                print(
                    f"[{task_name}] Auto-resume checkpoint already reached target epochs ({cfg['epochs']}). "
                    "Skipping resume and starting fresh.",
                    flush=True,
                )
            else:
                resume_arg = str(auto_last.resolve())
                resume_source = "auto"
                print(f"[{task_name}] Auto-resume enabled. Found checkpoint: {resume_arg}")

    attempts = _build_attempts(cfg)
    last_error: RuntimeError | None = None
    aug_keys = [
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "erasing",
        "auto_augment",
        "close_mosaic",
    ]
    aug_kwargs = {k: cfg[k] for k in aug_keys if k in cfg}

    for idx, attempt in enumerate(attempts, start=1):
        print(
            f"[{task_name}] Training attempt {idx}/{len(attempts)} "
            f"(device={attempt['device']}, imgsz={attempt['imgsz']}, "
            f"batch={attempt['batch']}, workers={attempt['workers']}, amp={attempt['amp']})"
        )
        clear_cuda_cache()

        try:
            return model.train(
                data=str(data_yaml),
                epochs=int(cfg["epochs"]),
                imgsz=int(attempt["imgsz"]),
                batch=int(attempt["batch"]),
                device=str(attempt["device"]),
                workers=int(attempt["workers"]),
                amp=bool(attempt["amp"]),
                resume=resume_arg,
                project=str(cfg["project"]),
                name=str(cfg["name"]),
                exist_ok=True,
                **aug_kwargs,
            )
        except AssertionError as exc:
            msg = str(exc).lower()
            if resume_arg and "nothing to resume" in msg:
                if resume_source == "manual":
                    raise
                print(
                    f"[{task_name}] Resume checkpoint is not resumable for current epochs. "
                    "Retrying as a fresh run.",
                    flush=True,
                )
                resume_arg = False
                resume_source = "none"
                continue
            raise
        except RuntimeError as exc:
            if not is_cuda_memory_error(exc):
                raise
            last_error = exc
            print(f"[{task_name}] Attempt {idx} failed due to CUDA memory issue. Retrying with safer settings...")

    if last_error is not None:
        raise RuntimeError(
            f"{task_name} training failed after {len(attempts)} memory-safe attempts. "
            "Set a smaller batch/imgsz or enable cpu_fallback in config."
        ) from last_error
    raise RuntimeError(f"{task_name} training failed unexpectedly.")
