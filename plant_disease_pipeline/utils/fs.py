from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(path: str | Path, lines: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def copy_or_link_file(src: str | Path, dst: str | Path, force_copy: bool = False) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        return

    if force_copy:
        shutil.copy2(src_path, dst_path)
        return

    try:
        dst_path.hardlink_to(src_path)
    except Exception:
        shutil.copy2(src_path, dst_path)

