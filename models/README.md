# Model Directory

This directory contains trained models for parcel damage classification.

## Available Models

- `best1.pt`: Best PyTorch model, suitable for PyTorch inference
- `best1.onnx`: ONNX version of the best model, suitable for cross-platform deployment

## Model Architecture

The models are based on YOLOv11n-cls architecture with the following modifications:
- Classification head adapted for binary classification (damaged/undamaged)

## Training Setting
These models were trained with the following parameters:
- Epochs: 100
- Image size: 640×640
- Batch size: 16
- Optimizer: AdamW
- Learning rate: 0.001