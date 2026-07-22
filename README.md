# Plant Disease Pipeline (YOLOv12 + SAM3)

This project implements the full workflow you shared:

1. `YOLOv12` detection training on bounding boxes.
2. ROI selection from detection (or GT box prompts).
3. `SAM3` pseudo mask generation and best-mask selection.
4. Pseudo segmentation dataset construction.
5. Segmentation model training (`YOLO-seg`).
6. Inference + disease severity estimation.

## Project Structure

```text
configs/
  pipeline.yaml

plant_disease_pipeline/
  cli.py
  config.py
  pipeline.py
  severity.py
  data/
    detection_split.py
    pseudo_segmentation_builder.py
  models/
    detector.py
    sam3_wrapper.py
    segmenter.py
  utils/
    fs.py
    geometry.py
    visualization.py

dataset/
  train/images
  train/labels
```

## Install

```bash
pip install -r requirements.txt
```

Note:
- `sam3.pt` must be downloaded manually and placed at repo root (or set `sam.model_path` in `configs/pipeline.yaml`).
- Keep `ultralytics` updated (`>=8.3.237`) for SAM3 support.

## Configuration

Edit [`configs/pipeline.yaml`](configs/pipeline.yaml) for:
- dataset paths
- YOLOv12 model/epochs/device
- training safety (`auto_safe_mode`, `amp`, `cpu_fallback`)
- terminal epoch logs (`print_epoch_metrics`)
- resume training (`auto_resume`, `resume_checkpoint`)
- augmentation controls (`degrees`, `shear`, `perspective`, `flipud`, `mixup`, `cutmix`, `copy_paste`)
- preprocessing checks (`split.validate_images`, `split.auto_fix_labels`)
- imbalance handling (`split.oversample_minority`, `split.imbalance_target_ratio`, `split.imbalance_max_repeat`)
- SAM3 path and thresholds
- pseudo-label prompt source (`gt` or `detector`)
- segmentation training params
- inference options (`leaf_area_mode: bbox|sam`)

### CUDA Memory Safety

Detection and segmentation training have automatic fallback retries enabled:
- lower batch size
- lower image size
- fewer workers
- disable AMP if needed

If GPU still fails, you can enable:

```yaml
detection:
  cpu_fallback: true
segmentation:
  cpu_fallback: true
```

### Resume From Last Checkpoint

If training stopped early and you want to continue (for example from epoch 25 to 50), keep the same
`project` and `name`, set:

```yaml
detection:
  auto_resume: true
  epochs: 50
```

Then run training again. It will resume from `weights/last.pt` automatically when available.

## Commands

### 1) Prepare train/val split for detection

```bash
python -m plant_disease_pipeline.cli prepare --config configs/pipeline.yaml
```

This step now also writes:
- `artifacts/detection_split/split_report.yaml` (data quality + balancing report)
- `artifacts/detection_split/class_distribution.yaml` (before/after imbalance handling)
- `artifacts/detection_split/train_raw.txt` (unbalanced train list)
- `artifacts/detection_split/train.txt` (balanced train list used for training)

### 2) Train YOLOv12 detector

```bash
python -m plant_disease_pipeline.cli train-detector --config configs/pipeline.yaml
```

### 3) Build pseudo segmentation dataset with SAM3

```bash
python -m plant_disease_pipeline.cli build-pseudo-seg --config configs/pipeline.yaml
```

If you use:
```yaml
pseudo_segmentation:
  prompt_source: detector
```
train the detector first (or provide `--detector-weights`).

### 4) Train segmentation model

```bash
python -m plant_disease_pipeline.cli train-segmenter --config configs/pipeline.yaml
```

### 5) Inference + severity

```bash
python -m plant_disease_pipeline.cli infer --config configs/pipeline.yaml --image path/to/image.jpg
```

Outputs are written to `artifacts/inference/`:
- `*_severity.json`
- `*_severity.jpg`

### Run entire workflow

```bash
python -m plant_disease_pipeline.cli run-all --config configs/pipeline.yaml
```

## Severity Formula

For each detected lesion:

```text
Lesion Area  = count(predicted lesion mask pixels)
Leaf Area    = count(leaf mask pixels) OR estimated ROI box area
Severity (%) = (Lesion Area / Leaf Area) * 100
```
