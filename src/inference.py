# WSGI entrypoint for hosting the Dash app (e.g., `gunicorn src.inference:server`)
# This is safe: it only creates `server` if the dashboard can be imported.
try:
    from .dashboard import app as dash_app
    server = dash_app.server
except Exception as e:
    server = None
    print(f"Warning: could not initialize Dash server: {e}")

# src/inference.py
import os
import time
import cv2
import numpy as np
try:
    import coremltools as ct
except ModuleNotFoundError:
    ct = None
from PIL import Image
from datetime import datetime
import sys

class CoreMLClassifier:
    def __init__(self, model_path, confidence_threshold=0.5, camera_index=0):
        """Initialize the classifier using CoreML for Mac."""
        
        if ct is None:
            raise RuntimeError(
                "coremltools is not available on this platform. "
                "Run this script on macOS or use the ONNX dashboard/inference path."
            )
        
        self.model_path = os.path.abspath(model_path)
        print(f"Loading CoreML model from {self.model_path}...")
        
        try:
            self.model = ct.models.MLModel(self.model_path)
            print("Model loaded successfully!")
            
            spec = self.model.get_spec()
            self.input_name = spec.description.input[0].name
            self.output_names = [output.name for output in spec.description.output]
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        self.class_names = ['undamaged', 'damaged']
        self.conf_threshold = confidence_threshold
        self.camera_index = camera_index
        self.camera = None
        self.init_camera()

    def init_camera(self):
        """Initialize the camera with macOS-specific handling."""
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            self.camera = None
        
        methods = [
            lambda: cv2.VideoCapture(self.camera_index),
            lambda: cv2.VideoCapture(self.camera_index + cv2.CAP_AVFOUNDATION),
            lambda: cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
        ]
        
        for i, method in enumerate(methods):
            try:
                self.camera = method()
                if self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        print(f"Camera initialized successfully with method {i+1}")
                        return
                self.camera.release()
            except:
                continue
        
        print("WARNING: Using dummy camera for testing...")
        self.camera = DummyCamera()

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (640, 640))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def run_inference(self, frame):
        try:
            pil_image = self.preprocess_frame(frame)
            input_dict = {self.input_name: pil_image}
            output = self.model.predict(input_dict)
            
            result = {'class': 'unknown', 'confidence': 0.0, 'is_damaged': False}
            
            if 'classLabel' in output:
                label = output['classLabel'].lower()
                result['class'] = label
                result['is_damaged'] = (label == 'damaged')
            
            if 'classLabel_probs' in output:
                probs = output['classLabel_probs']
                result['undamaged_prob'] = float(probs.get('undamaged', probs.get(0, 0)))
                result['damaged_prob'] = float(probs.get('damaged', probs.get(1, 0)))
                result['confidence'] = probs.get(output['classLabel'], 0)
                
                if result['confidence'] < self.conf_threshold:
                    result['class'] = 'uncertain'
            
            return result
        except Exception as e:
            return {'class': 'error', 'confidence': 0.0}

    def visualize_result(self, frame, result):
        output = frame.copy()
        color = (0, 0, 255) if result.get('is_damaged') else (0, 255, 0)
        
        cv2.putText(output, f"{result['class'].upper()}: {result.get('confidence', 0):.2f}", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        return output

    def run(self):
        """Main loop for real-time inference."""
        print("Real-time inference started. Press 'q' to quit.")
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret: break
                
                result = self.run_inference(frame)
                display_frame = self.visualize_result(frame, result)
                
                cv2.imshow("Parcel Damage Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

class DummyCamera:
    def __init__(self):
        self.width, self.height = 640, 480
    def isOpened(self): return True
    def read(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(frame, "DUMMY CAMERA", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return True, frame
    def release(self): pass

def main(model_path="models/best1.mlpackage", threshold=0.5, camera_idx=0):
    """Entry point for the inference pipeline."""
    classifier = CoreMLClassifier(
        model_path=model_path,
        confidence_threshold=threshold,
        camera_index=camera_idx
    )
    classifier.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", default="models/best1.mlpackage", help="Path to CoreML model")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    
    main(model_path=args.model, threshold=args.threshold, camera_idx=args.camera)