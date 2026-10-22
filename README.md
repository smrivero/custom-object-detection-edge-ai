# Custom YOLO26: From Dataset to Edge Deployment

Training and deploying a custom object detection model on NVIDIA Jetson Orin Nano Super.

**A complete applied ML workflow: dataset auditing, annotation correction, YOLO26 fine-tuning, CUDA training and live camera inference.**

The model targets two custom categories, mate and thermos. The implementation uses 99 curated images and documents annotation corrections, dataset splitting, training configuration and real-camera evaluation.

Observed false positives and missed detections are documented to assess generalization beyond the training data.

## Overview

This project documents the process of preparing a labeled dataset, fine-tuning a YOLO26 Nano detector and evaluating predictions on a live camera feed.

The trained model targets two custom classes: mate and thermos.

## Machine Learning Pipeline

1. Review source images and annotations.
2. Correct incorrect bounding boxes.
3. Split images into training, validation and test sets.
4. Train YOLO26 Nano using CUDA.
5. Evaluate detection behavior.
6. Execute inference on NVIDIA Jetson.
7. Compare successful detections, false positives and missed objects.

## Dataset

The source dataset was published by Carinas Workspace on Roboflow Universe.

Source:
https://universe.roboflow.com/carinas-workspace-j6jgl/tp-deteccion-mate-y-termo/dataset/1

Declared license: CC BY 4.0.

| Split | Images |
|---|---:|
| Training | 69 |
| Validation | 14 |
| Test | 16 |
| Total | 99 |

Classes:

- 0: Mate
- 1: Thermos

See `DATASET_NOTES.md` for the dataset audit, annotation corrections and split methodology.

Dataset images are not included in the repository.

## Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLO26 Nano |
| Maximum epochs | 60 |
| Batch size | 2 |
| Image size | 640 |
| Early stopping patience | 15 |
| Mixed precision | Enabled |
| GPU | NVIDIA Jetson CUDA |

## Repository Components

| File | Purpose |
|---|---|
| `train.py` | Custom model training |
| `check_dataset.py` | Dataset integrity checks |
| `live_custom_detection.py` | Live inference with trained custom model |
| `live_coco_detection.py` | COCO pretrained model comparison |
| `DATASET_NOTES.md` | Dataset preparation and quality assessment |

## Engineering Highlights

- Custom dataset preparation and annotation quality control.
- Training and validation split management.
- YOLO26 fine-tuning using CUDA-enabled PyTorch.
- Real-time camera inference on NVIDIA Jetson.
- Custom class detection and COCO model comparison.
- Analysis of false positives and generalization limitations.

## Requirements

- NVIDIA Jetson Orin Nano Super 8 GB
- CUDA-enabled PyTorch
- Compatible Ultralytics YOLO26
- NumPy
- OpenCV
- Intel RealSense D435 and Python SDK for live camera inference

## Training

Prepare the dataset at:

    dataset/

Validate the dataset:

    python3 check_dataset.py

Train:

    python3 train.py

The training script checks CUDA availability and backward computation before starting.

Trained checkpoints are written to a local `runs/` directory.

## Live Inference

Run the customized detector using a checkpoint:

    python3 live_custom_detection.py --model /path/to/best.pt

Run a COCO pretrained model:

    python3 live_coco_detection.py --model /path/to/yolo26n.pt

The application exposes a browser-based camera preview on the configured network port.

## Evaluation Findings

The trained model detected the thermos in live camera testing.

Mate detection was less reliable, and false positives occurred, including misclassifying a person as mate.

These observations indicate limitations in generalization beyond the training dataset.

## Evaluation Metrics

Recommended measurements:

- Per-class precision and recall
- mAP50 and mAP50-95
- Confusion matrix
- Validation on independent camera recordings
- GPU inference latency
- End-to-end FPS

No unverified accuracy or production-readiness claims are made.
