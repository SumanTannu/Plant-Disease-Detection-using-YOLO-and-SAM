from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import (
    build_pseudo_segmentation,
    load_artifacts,
    prepare_detection_data,
    run_all,
    run_inference,
    train_detector,
    train_segmenter,
)


def build_parser() -> argparse.ArgumentParser:
    def add_config_argument(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            type=str,
            default="configs/pipeline.yaml",
            help="Path to pipeline config YAML.",
        )

    parser = argparse.ArgumentParser(
        description="Plant disease workflow: YOLOv12 detection + SAM3 pseudo labels + segmentation + severity.",
    )
    add_config_argument(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_prepare = subparsers.add_parser("prepare", help="Create train/val split and detection data YAML.")
    add_config_argument(parser_prepare)

    parser_train_det = subparsers.add_parser("train-detector", help="Train YOLOv12 detector.")
    add_config_argument(parser_train_det)
    parser_train_det.add_argument(
        "--data-yaml",
        type=str,
        default=None,
        help="Optional detection data YAML. If omitted, artifacts/detection split YAML is used.",
    )

    parser_pseudo = subparsers.add_parser(
        "build-pseudo-seg",
        help="Generate pseudo segmentation labels with SAM3.",
    )
    add_config_argument(parser_pseudo)
    parser_pseudo.add_argument(
        "--detector-weights",
        type=str,
        default=None,
        help="Detector weights path (required only if prompt_source=detector).",
    )

    parser_train_seg = subparsers.add_parser("train-segmenter", help="Train segmentation model.")
    add_config_argument(parser_train_seg)
    parser_train_seg.add_argument(
        "--seg-data-yaml",
        type=str,
        default=None,
        help="Segmentation data YAML path. Defaults to artifacts output.",
    )

    parser_infer = subparsers.add_parser("infer", help="Inference + severity estimation for one image.")
    add_config_argument(parser_infer)
    parser_infer.add_argument("--image", type=str, required=True, help="Input image path.")
    parser_infer.add_argument("--detector-weights", type=str, default=None, help="Detector weights path.")
    parser_infer.add_argument("--segmenter-weights", type=str, default=None, help="Segmenter weights path.")

    parser_run_all = subparsers.add_parser("run-all", help="Run complete end-to-end training pipeline.")
    add_config_argument(parser_run_all)
    parser_show = subparsers.add_parser("show-artifacts", help="Print current artifacts manifest.")
    add_config_argument(parser_show)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "prepare":
        artifacts = prepare_detection_data(cfg)
        print("Detection split created.")
        print(artifacts.get("detection_data_yaml"))
        return

    if args.command == "train-detector":
        artifacts = train_detector(cfg, data_yaml=args.data_yaml)
        print("Detector training complete.")
        print(artifacts.get("detector_weights"))
        return

    if args.command == "build-pseudo-seg":
        artifacts = build_pseudo_segmentation(cfg, detector_weights=args.detector_weights)
        print("Pseudo segmentation dataset created.")
        print(artifacts.get("segmentation_data_yaml"))
        return

    if args.command == "train-segmenter":
        artifacts = train_segmenter(cfg, seg_data_yaml=args.seg_data_yaml)
        print("Segmentation training complete.")
        print(artifacts.get("segmenter_weights"))
        return

    if args.command == "infer":
        out = run_inference(
            cfg,
            image_path=args.image,
            detector_weights=args.detector_weights,
            segmenter_weights=args.segmenter_weights,
        )
        print("Inference complete.")
        print(out["report_json"])
        print(out["visualization_image"])
        return

    if args.command == "run-all":
        artifacts = run_all(cfg)
        print("Full pipeline completed.")
        print(artifacts.get("detector_weights"))
        print(artifacts.get("segmenter_weights"))
        return

    if args.command == "show-artifacts":
        artifacts = load_artifacts(cfg)
        print(artifacts)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
