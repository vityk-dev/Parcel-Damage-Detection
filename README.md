# Automatic Parcel Damage Recognition Module for an Inspection Robot

## Abstract

This repository presents an automatic parcel damage recognition module designed for integration with autonomous inspection robots in warehouse environments. Leveraging state-of-the-art YOLO-based deep learning models, the system detects and classifies various types of parcel damages from images, enabling efficient quality control and operational safety. The module is optimized for real-time inference and includes training, evaluation, and deployment pipelines.

## Repository Structure

```
parcel-damage-detection/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── runtime.txt
├── IEEE Article/
│   └── YOLO_FEDCIS_2025___iDS.pdf
├── models/
│   ├── best0.pt
│   ├── best1.pt
│   ├── best0.onnx
│   ├── best1.onnx
│   └── best1.mlpackage/
├── results/
│   ├── yolo_training_process_static.png
│   └── yolo_training_process_interactive.html
├── runs/
├── src/
│   ├── __init__.py
│   ├── dashboard.py
│   ├── inference.py
│   ├── train.py
│   ├── dataset.py
│   ├── export.py
│   ├── convert.py
│   ├── evalu1.py
│   └── visualize_training.py
├── test/
└── yolo-project/
```

## Live Demo

Explore the interactive dashboard for parcel damage detection and model comparison:

👉 [https://parcel-damage-detection.onrender.com/](https://parcel-damage-detection.onrender.com/)

## Installation

This project uses `pyproject.toml` for dependency management. To install the package and dependencies locally, run:

```bash
pip install -e .
```

## Usage

### Running the Dashboard

Launch the interactive dashboard to upload parcel images, perform damage detection, and visualize metrics:

```bash
python src/dashboard.py
```

### Running Inference

Run inference on new images using the trained models:

```bash
python src/inference.py
```

### Training the Model

Train the YOLO-based models on the parcel damage dataset:

```bash
python src/train.py
```

## Docker Usage

### Build Docker Image

Build the Docker image for the parcel damage detection module:

```bash
docker build -t parcel-damage-detection .
```

### Run Docker Container

Run the container with port 8050 exposed to access the dashboard:

```bash
docker run --rm -p 8050:8050 parcel-damage-detection
```

## Docker Compose

### Build and Start Services

Use Docker Compose to build and start the application services:

```bash
docker compose up --build
```

### Run Training Inside Container

Execute training within the running container environment:

```bash
docker compose run app python src/train.py
```

## Dataset

The dataset consists of annotated images of parcels with various damage types commonly encountered in logistics and warehouse handling. Annotations are provided in YOLO format to facilitate training and evaluation of detection models.

## Architecture

The detection module is built upon the YOLO (You Only Look Once) architecture, enabling efficient real-time object detection. Multiple trained models (`best0.pt`, `best1.pt`) are provided alongside their ONNX and Core ML package exports for deployment flexibility.

## Results

The repository includes training logs, performance metrics, and visualization tools to analyze model accuracy and confusion matrices. Sample results are available in the `results/` directory, showcasing the training process and evaluation outcomes.

## References

- YOLO: Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR, 2016.
- Dataset and annotations curated for parcel damage types specific to warehouse logistics.

---

*This repository accompanies our IEEE conference paper demonstrating practical computer vision implementation for autonomous warehouse inspection systems.*
