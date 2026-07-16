# YOLOv8 Object Detection - Old Building Objects

Custom object detection model trained on YOLOv8 to identify objects in old building environments.

## Project Overview

This project trains a YOLOv8 model to detect 4 classes of objects commonly found in old building environments:
- Door
- Dustbin
- Outdoor Plant
- Window

## Technologies


- Python 3.12
- PyTorch 2.9.0
- Ultralytics YOLOv8
- CUDA 12.6
- Google Colab

## Dataset

- **Source**: Roboflow - LGU Old Building Objects
- **Classes**: 4
- **Format**: YOLOv8
- **Images**: 760+ annotated images
- **Train/Val/Test Split**: 70/20/10

## Model Architecture

- **Base Model**: YOLOv8n (nano) - lightweight and fast
- **Input Size**: 640x640
- **Epochs**: 50
- **Device**: NVIDIA Tesla T4 (Google Colab)

## Training Results

| Metric | Value |
|--------|-------|
| Precision (Door) | 0.927 |
| Recall (Door) | 0.75 |
| mAP50 (Door) | 0.748 |
| Precision (Dustbin) | 0.893 |
| Recall (Dustbin) | 0.909 |
| mAP50 (Dustbin) | 0.906 |
| Precision (Outdoor Plant) | 0.893 |
| Recall (Outdoor Plant) | 0.909 |
| mAP50 (Outdoor Plant) | 0.906 |

## Inference

The model successfully processes video input in real-time:
- **Input**: original.mp4 (1722 frames)
- **Output**: Annotated video with bounding boxes
- **Speed**: ~10ms per frame (100+ FPS on T4)
- **Confidence Threshold**: 0.25

## Files

| File | Description |
|------|-------------|
| `src/train.py` | Training script |
| `src/predict.py` | Inference script |
| `models/best.pt` | Trained model weights |
| `data/data.yaml` | Dataset configuration |
| `videos/demo_output.mp4` | Sample detection output |

## Author
Mahmoona Khan

## Installation

```bash
pip install -r requirements.txt
