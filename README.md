# Automatic Parcel Damage Recognition Module for an Inspection Robot

[![IEEE Conference](https://img.shields.io/badge/IEEE-Conference%20Paper-blue.svg)](https://github.com/vityk-dev/Parcel-Damage-Detection/blob/main/IEEE%20Article/YOLO_FEDCIS_2025___iDS.pdf)  
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)  
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Classification-green.svg)](https://github.com/ultralytics/ultralytics)

**Author:** Wiktor Goszczynski  
**Author of dataset:** Szymon Wałęga  
**Institution:** AGH University of Krakow, Department of Automatic Control and Robotics

---

## 🚀 Live Demo (Dashboard)

Try the interactive web dashboard for real-time parcel damage classification here:  
[https://parcel-damage-detection.onrender.com/](https://parcel-damage-detection.onrender.com/)

---

## 📋 Abstract

This repository presents our work on expanding machine learning hardware and algorithm solutions for a damage detection inspection robot as part of creating a digital twin warehouse system. We developed a comprehensive dataset of over 6,800 images and applied tailored data augmentation to capture operational environment variability. Our YOLOv11n-cls-based model achieves **98.50% accuracy**, **97.04% precision**, and **99.74% recall** on validation data, with inference speeds exceeding **251 FPS** on Apple M1 hardware via CoreML optimization.

---

## 🎯 Key Achievements

- **High Performance**: 98.50% accuracy with 99.74% recall (critical for minimizing missed damage)  
- **Real-time Processing**: 251+ FPS inference speed on optimized hardware  
- **Comprehensive Dataset**: 6,800+ images with domain-specific augmentation  
- **Production Ready**: CoreML optimization for deployment  
- **Interactive Dashboard**: Real-time model comparison and testing interface  

---

## 📊 Model Performance

### Final Results (best1 model)

| Metric     | Score   | Improvement over best0 |
|------------|---------|-----------------------|
| **Accuracy**  | 98.50%  | +1.50pp               |
| **Precision** | 97.04%  | +2.24pp               |
| **Recall**    | 99.74%  | +0.93pp               |
| **F1-Score**  | 98.37%  | +1.61pp               |

### Real-world Conditions Testing

Tested under challenging lighting conditions (darker/lighter environments):

- **Accuracy**: 94.44%  
- **Precision**: 91.14%  
- **Recall**: 100%  
- **F1-Score**: 95.36%  

---

## 🏗️ System Architecture

### 1. Custom Inspection Robot

- **Omnidirectional drive** with mecanum wheels  
- **NVIDIA Jetson Nano** for ML processing  
- **RGB Camera** with servo mount for 360° coverage  
- **LiDAR** for spatial awareness  
- **Arduino** for motor control and sensor integration  

### 2. Computer Vision Pipeline

- **YOLOv11n-cls** classification model  
- **640×640 pixel** input resolution  
- **Domain-specific augmentation** using Albumentations  
- **Iterative improvement** process (best0 → best1)  

### 3. Optimization Stack

- **PyTorch** → **ONNX** → **CoreML** conversion pipeline  
- **Apple Neural Engine** optimization  
- **30 FPS** (PyTorch) → **251+ FPS** (CoreML)  

---

## 📈 Dataset Details

- **Total Images**: 6,800+  
- **Training Set**: 4,972 images (2,372 damaged + 2,600 undamaged)  
- **Validation Set**: 1,903 images (903 damaged + 1,000 undamaged)  
- **Augmentation**: 10x per original image with realistic transformations  
- **Sources**: Kaggle dataset + custom real-world captures  

### Augmentation Strategy

- Conservative spatial transforms (±10° rotation, 0.95-1.05 scale)  
- Warehouse lighting simulation (±10% brightness/contrast)  
- Realistic shadow effects (20% probability)  
- Horizontal flips and minor translations  

---

## 🔬 Research Contribution

This work contributes to logistics automation by:

- Demonstrating practical deep learning implementation for quality control  
- Achieving production-ready performance with real-time constraints  
- Providing comprehensive evaluation methodology  
- Enabling integration into existing warehouse workflows  

**IEEE Conference Paper**: *"Automatic Parcel Damage Recognition Module for an Inspection Robot"*

---

## 📁 Repository Structure

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
└── test/
```

---

## 🛠️ Technical Stack

- **Deep Learning**: YOLOv11 (Ultralytics), PyTorch  
- **Optimization**: ONNX, CoreML, Apple Neural Engine  
- **Data Processing**: Albumentations, FiftyOne, NumPy, Pandas  
- **Visualization**: Dash, Plotly, Bootstrap  
- **Hardware**: NVIDIA Jetson Nano, Apple M1 (inference testing)  

---

## 📋 Installation & Usage

### Run the Dashboard (Live Demo)

If you just want to test the system, use the hosted dashboard:

- **Dashboard:** https://parcel-damage-detection.onrender.com/

---

### Run Locally (Python)

This project is configured via **`pyproject.toml`**.

1. Clone the repository:

    ```bash
    git clone https://github.com/vityk-dev/Parcel-Damage-Detection.git
    cd Parcel-Damage-Detection
    ```

2. Create and activate a virtual environment (recommended):

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

3. Install the project using the `pyproject.toml` configuration:

    ```bash
    pip install -e .
    ```

4. Run the dashboard locally:

    ```bash
    python src/dashboard.py
    ```

5. Open your browser:

- `http://127.0.0.1:8050` (or `http://localhost:8050`)

> Note: If your dashboard entry file is located elsewhere in `src/`, run it accordingly (e.g., `python -m src.app`).



6. Run inference on new images using the trained models:

    ```bash
    python src/inference.py
    ```
---

### Run with Docker

#### Build & Run (Docker)

The Docker container runs the production web server via Gunicorn using `src.inference:server`.

1. Build the image:

```bash
docker build -t parcel-damage-detection .
```

2. Run the container:

```bash
docker run --rm -p 8050:8050 parcel-damage-detection
```

3. Open:

- `http://localhost:8050`

#### Run with Docker Compose

If you prefer Compose (recommended for consistency):

```bash
docker compose up --build
```

Then open:

- `http://localhost:8050`

Stop with **Ctrl+C**, then clean up:

```bash
docker compose down
```

---

### Testing Tips (Multiple Images per Parcel)

- The dashboard supports uploading **multiple images for a single parcel** to improve robustness.
- When testing, drag & drop (or select) multiple images at once.
- The system aggregates predictions across images and returns a combined decision.

---

## 🔮 Future Work

- **Damage Localization**: Extend to highlight specific damaged areas  
- **Multi-class Classification**: Distinguish damage types (tears, water, crushing)  
- **Edge Deployment**: Further optimization for conveyor system integration  
- **Multi-view Integration**: Combine multiple camera angles  
- **Continuous Learning**: Operational feedback integration  

---

## 📞 Contact

- **Wiktor Goszczynski**: wiktorg@student.agh.edu.pl  
- **Institution**: AGH University of Krakow  
- **Department**: Automatic Control and Robotics  

---

## 🏆 Acknowledgments

This work was conducted within the Industrial Data Science (IDS) student research group at AGH University of Krakow as part of developing a digital twin warehouse system.

---

*This repository accompanies our IEEE conference paper demonstrating practical computer vision implementation for autonomous warehouse inspection systems.*
