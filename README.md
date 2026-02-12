End-to-end computer vision system for real-time parcel damage classification, designed for an autonomous warehouse inspection robot and digital twin environment.

⸻

🚀 Live Demo

Interactive dashboard:
https://huggingface.co/spaces/wiktorgit/dashboard

⸻

📋 Overview

The project covers:

	•	Dataset engineering
	•	Model training and evaluation
	•	Experiment tracking with MLflow
	•	Inference optimization (ONNX, CoreML)
	•	Edge deployment (NVIDIA Jetson Nano)
	•	Docker-based production setup

The YOLOv11n-cls model achieves:

	•	98.50% accuracy
	•	97.04% precision
	•	99.74% recall

Inference speed reaches 251+ FPS after optimization.

⸻

🎯 Key Highlights

	•	End-to-end ML pipeline (data → training → deployment)
	•	MLflow-based experiment tracking and reproducibility
	•	ONNX and CoreML inference optimization
	•	Deployment-ready Docker setup with Gunicorn
	•	Designed for Kubernetes-based scaling
	•	Validated on edge device (Jetson Nano)

⸻

📊 Model Performance

Final Results (best1)

Metric	Score
Accuracy	98.50%
Precision	97.04%
Recall	99.74%
F1-Score	98.37%

Real-world Lighting Test

Metric	Score
Accuracy	94.44%
Precision	91.14%
Recall	100%
F1-Score	95.36%


⸻

🏗️ System Architecture

Inspection Robot

	•	Omnidirectional drive (mecanum wheels)
	•	NVIDIA Jetson Nano
	•	RGB camera (servo-mounted)
	•	LiDAR
	•	Arduino for motor control

⸻

Computer Vision Pipeline

	•	YOLOv11n-cls (PyTorch)
	•	640×640 resolution
	•	Domain-specific augmentation (Albumentations)
	•	Iterative training workflow
	•	Clear separation between training and inference

⸻

Optimization & Deployment

Training and deployment pipeline:

	1.	Train model in PyTorch
	2.	Track experiments with MLflow
	3.	Export to ONNX
	4.	Optimize inference via ONNX Runtime
	5.	Convert to CoreML (Apple Neural Engine)
	6.	Deploy on Jetson Nano

Backend:

	•	FastAPI-compatible inference server
	•	Gunicorn production server
	•	Dockerized environment
	•	Structured for Kubernetes deployment

⸻

🔁 ML Lifecycle

This project follows a simple but production-oriented ML lifecycle:

	•	MLflow for experiment tracking
	•	Reproducible training configuration
	•	Clear separation between training, evaluation, and inference
	•	Containerized deployment
	•	Prepared for CI/CD and scalable orchestration

⸻

📈 Dataset

	•	6,800+ images
	•	Balanced damaged / undamaged samples
	•	10× augmentation per image
	•	Combination of Kaggle and custom warehouse data

Augmentation includes:

	•	Small spatial transforms
	•	Lighting simulation
	•	Shadow effects
	•	Minor translations and flips

The goal was realistic operational variability.

⸻

🛠️ Tech Stack

ML
	•	PyTorch
	•	YOLOv11
	•	MLflow
	•	ONNX Runtime
	•	CoreML

Data
	•	Albumentations
	•	FiftyOne
	•	NumPy
	•	Pandas

Backend & Deployment
	•	FastAPI
	•	Dash (Plotly)
	•	Gunicorn
	•	Docker
	•	Kubernetes-ready structure

Hardware
	•	NVIDIA Jetson Nano
	•	Apple M1

⸻

📦 Run Locally

git clone https://github.com/vityk-dev/Parcel-Damage-Detection.git
cd Parcel-Damage-Detection
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python src/dashboard.py

Open:
http://localhost:8050

⸻

🐳 Run with Docker

docker build -t parcel-damage-detection .
docker run -p 8050:8050 parcel-damage-detection

Or:

docker compose up --build


⸻

🔮 Future Improvements

	•	Damage localization
	•	Multi-class damage detection
	•	Automated retraining pipeline
	•	Drift monitoring
	•	Full Kubernetes deployment with monitoring stack

⸻

📄 Paper

IEEE Conference Paper:
“Automatic Parcel Damage Recognition Module for an Inspection Robot”

⸻

📞 Contact

Wiktor Goszczynski
wiktorg@student.agh.edu.pl
AGH University of Krakow