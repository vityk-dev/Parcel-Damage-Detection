# Parcel Damage Classification
## Overview

This project provides a complete pipeline for training, evaluating, and deploying a parcel damage classification model. It uses YOLOv11n classification models to determine whether a parcel is damaged or undamaged.

## Installation

1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

### Using pre-trained models
Run inference on an image using PyTorch model
python scripts/inference.py --model models/best1.pt --source test/testing_on_real

Run real-time inference using a webcam
python scripts/inference.py --model models/best1.pt --source 0 --view-img

### Training model 
python scripts/train.py

## Exporting the model
Export to Onnx
python scripts/export_model.py --model models/best1.pt --format onnx

Export to other formats
python scripts/export_model.py --model models/best1.pt --format coreml  
python scripts/export_model.py --model models/best1.pt --format tensorrt

## Visualizing dataset
python scripts/load_dataset.py