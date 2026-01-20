# Automatic Parcel Damage Recognition Module for an Inspection Robot

[![IEEE Conference](https://img.shields.io/badge/IEEE-Conference%20Paper-blue.svg)](https://github.com/vityk-dev/Parcel-Damage-Detection/blob/main/IEEE%20Article/YOLO_FEDCIS_2025___iDS.pdf)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Classification-green.svg)](https://github.com/ultralytics/ultralytics)

**Author:** Wiktor Goszczynski
**Institution:** AGH University of Krakow, Department of Automatic Control and Robotics

## 📋 Abstract

This repository presents our work on expanding machine learning hardware and algorithm solutions for a damage detection inspection robot as part of creating a digital twin warehouse system. We developed a comprehensive dataset of over 6,800 images and applied tailored data augmentation to capture operational environment variability. Our YOLOv11n-cls-based model achieves **98.50% accuracy**, **97.04% precision**, and **99.74% recall** on validation data, with inference speeds exceeding **251 FPS** on Apple M1 hardware via CoreML optimization.

## 🎯 Key Achievements

- **High Performance**: 98.50% accuracy with 99.74% recall (critical for minimizing missed damage)
- **Real-time Processing**: 251+ FPS inference speed on optimized hardware
- **Comprehensive Dataset**: 6,800+ images with domain-specific augmentation
- **Production Ready**: CoreML optimization for deployment
- **Interactive Dashboard**: Real-time model comparison and testing interface

## 📊 Model Performance

### Final Results (best1 model)
| Metric | Score | Improvement over best0 |
|--------|-------|----------------------|
| **Accuracy** | 98.50% | +1.50pp |
| **Precision** | 97.04% | +2.24pp |
| **Recall** | 99.74% | +0.93pp |
| **F1-Score** | 98.37% | +1.61pp |

### Real-world Conditions Testing
Tested under challenging lighting conditions (darker/lighter environments):
- **Accuracy**: 94.44%
- **Precision**: 91.14% 
- **Recall**: 100%
- **F1-Score**: 95.36%

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

## 🚀 Model Development Process

1. **Initial Training** (best0): Baseline model with original dataset
2. **Performance Analysis**: Identified weakness in opened box classification
3. **Dataset Enhancement**: Added specific opened box samples labeled as damaged
4. **Retraining** (best1): Improved model with enhanced dataset
5. **Optimization**: CoreML conversion for production deployment

## 📱 Interactive Dashboard

Web-based interface built with Dash and Plotly featuring:
- **Model Comparison**: Side-by-side performance analysis
- **Real-time Testing**: Drag-and-drop image classification
- **Comprehensive Metrics**: Confusion matrices, ROC curves, PR curves
- **Performance Monitoring**: Cached results and batch evaluation

## 🔬 Research Contribution

This work contributes to logistics automation by:
- Demonstrating practical deep learning implementation for quality control
- Achieving production-ready performance with real-time constraints
- Providing comprehensive evaluation methodology
- Enabling integration into existing warehouse workflows

**IEEE Conference Paper**: *"Automatic Parcel Damage Recognition Module for an Inspection Robot"*

## 📁 Repository Structure

```
parcel-damage-detection/
├── README.md
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── .gitignore
├── IEEE Article/
│   └── YOLO_FEDCIS_2025___iDS.pdf/ 
├── results/
│   ├── figures/
│   │   ├── model_after_finetuning/
│   │   ├── model_before_finetuning/
│   │   └── models_comparison/
│   └── metrics/
│       ├── best0_results.json
│       └── best1_results.json
└── dashboard/
    └── screenshots/
```

## 🛠️ Technical Stack

- **Deep Learning**: YOLOv11 (Ultralytics), PyTorch
- **Optimization**: ONNX, CoreML, Apple Neural Engine
- **Data Processing**: Albumentations, FiftyOne, NumPy, Pandas
- **Visualization**: Dash, Plotly, Bootstrap
- **Hardware**: NVIDIA Jetson Nano, Apple M1 (inference testing)

## 📋 Installation

```bash
pip install ultralytics torch torchvision
pip install dash plotly dash-bootstrap-components
pip install albumentations opencv-python pillow
pip install numpy pandas scikit-learn
```

For CoreML optimization:
```bash
pip install coremltools onnx
```

## 🔮 Future Work

- **Damage Localization**: Extend to highlight specific damaged areas
- **Multi-class Classification**: Distinguish damage types (tears, water, crushing)
- **Edge Deployment**: Further optimization for conveyor system integration
- **Multi-view Integration**: Combine multiple camera angles
- **Continuous Learning**: Operational feedback integration

## 📞 Contact

- **Wiktor Goszczynski**: wiktorg@student.agh.edu.pl
- **Institution**: AGH University of Krakow
- **Department**: Automatic Control and Robotics

## 🏆 Acknowledgments

This work was conducted within the Industrial Data Science (IDS) student research group at AGH University of Krakow as part of developing a digital twin warehouse system.

---

*This repository accompanies our IEEE conference paper demonstrating practical computer vision implementation for autonomous warehouse inspection systems.*
