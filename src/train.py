# src/train.py
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt") 

results = model.train(data="data/dataset", epochs=100, imgsz=640)