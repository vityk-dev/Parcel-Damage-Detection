# src/__init__.py
"""Top-level package exports.

IMPORTANT:
- Do not perform heavy work (e.g., training) at import time.
- We therefore expose *lazy* wrappers that import underlying modules only when called.
"""

from __future__ import annotations

from typing import Any


def train_pipeline(*args: Any, **kwargs: Any):
    from .train import model
    return model(*args, **kwargs)


def run_dataset_manager(*args: Any, **kwargs: Any):
    from .dataset import run_dataset_manager as _fn
    return _fn(*args, **kwargs)


def run_export_pipeline(*args: Any, **kwargs: Any):
    from .export import export_model_pipeline as _fn
    return _fn(*args, **kwargs)


def run_inference_pipeline(*args: Any, **kwargs: Any):
    from .inference import main as _fn
    return _fn(*args, **kwargs)


def run_dashboard(*args: Any, **kwargs: Any):
    from .dashboard import run_dashboard as _fn
    return _fn(*args, **kwargs)


def run_yolo_training_visualization(*args: Any, **kwargs: Any):
    from .visualize_training import run_yolo_training_visualization as _fn
    return _fn(*args, **kwargs)


__version__ = "1.0.0"
__author__ = "Wiktor Goszczynski, Szymon Wałęga"

__all__ = [
    "train_pipeline", 
    "run_dataset_manager", 
    "run_export_pipeline", 
    "run_inference_pipeline",
    "run_dashboard",  # Added this
    "run_yolo_training_visualization",
]
