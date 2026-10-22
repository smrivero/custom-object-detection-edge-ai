# Dataset Preparation and Quality Assessment

## Overview

Custom YOLO26 object detection for two classes:

- Mate: traditional South American drinking vessel
- Thermos: insulated beverage container

Target hardware: NVIDIA Jetson Orin Nano Super.

## Dataset Source

Publisher: Carinas Workspace, Roboflow Universe.

Dataset: TP: Deteccion Mate y Termo

https://universe.roboflow.com/carinas-workspace-j6jgl/tp-deteccion-mate-y-termo/dataset/1

Declared license: Creative Commons Attribution 4.0.

## Dataset Composition

| Split | Images | Mate annotations | Thermos annotations |
|---|---:|---:|---:|
| Training | 69 | 41 | 35 |
| Validation | 14 | 6 | 8 |
| Test | 16 | 9 | 7 |
| Total | 99 | 56 | 50 |

## Annotation Quality

The original dataset contained 100 images.

Quality improvements:

1. Corrected an incorrectly classified thermos.
2. Removed three incorrect annotations on an electric kettle.
3. Excluded an image containing an unannotated thermos.

The electric kettle image was retained as a negative example.

No retained image pixels were modified.

## Data Splitting

Related image sequences were assigned to the same subset
to reduce data leakage.

The HEIC room sequence was assigned to testing.
The WhatsApp 21:21 sequence was assigned to validation.

Some overlap in objects and locations may remain.

## Training Configuration

| Parameter | Value |
|---|---|
| Architecture | YOLO26 Nano |
| Maximum epochs | 60 |
| Batch size | 2 |
| Workers | 0 |
| Image caching | Disabled |
| Mixed precision | Enabled |
| Early stopping patience | 15 |
| Accelerator | CUDA |

## Execution

Validate the dataset with check_dataset.py.

Train the model with train.py.

A compatible CUDA-enabled PyTorch and Ultralytics
environment is required.

## Limitations

The dataset contains only 99 images.

Real-camera testing showed false positives and inconsistent
detection of some objects.

Offline metrics do not establish production performance.

Further improvements require additional annotated images,
negative examples and independent camera evaluations.

## Attribution

Original dataset published by Carinas Workspace.

https://universe.roboflow.com/carinas-workspace-j6jgl/tp-deteccion-mate-y-termo/dataset/1

License declared by publisher:
https://creativecommons.org/licenses/by/4.0/
