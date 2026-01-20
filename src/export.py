# src/export.py
import os
from ultralytics import YOLO

def export_model_pipeline(model_path, format_type, device=None):
    """
    Export a YOLOv11 model to a specified optimized format.
    
    Args:
        model_path (str): Path to the source .pt model weights.
        format_type (str): Target format (e.g., 'onnx', 'tensorrt', 'coreml').
        device (str, optional): Device to use for export processes.
    """
    
    abs_model_path = os.path.abspath(model_path)
    
    if not os.path.exists(abs_model_path):
        print(f"ERROR: Model file not found at {abs_model_path}")
        return

    print(f"Loading model for export: {abs_model_path}")
    model = YOLO(abs_model_path)
    
    
    is_classification = hasattr(model, 'task') and model.task == 'classify'
    print(f"Detected Task: {'Classification' if is_classification else 'Detection/Segmentation'}")
    
    try:
        if format_type == 'onnx':
            model.export(format='onnx', simplify=True)
            print(f"✓ Exported to ONNX: {abs_model_path.replace('.pt', '.onnx')}")
        
        elif format_type == 'tensorrt':
            
            model.export(format='engine', device=device or 0)
            print(f"✓ Exported to TensorRT: {abs_model_path.replace('.pt', '.engine')}")
        
        elif format_type == 'coreml':
            if is_classification:
                model.export(format='coreml')
            else:
                
                model.export(format='coreml', nms=True)
            print(f"✓ Exported to CoreML: {abs_model_path.replace('.pt', '.mlpackage')}")
        
        elif format_type == 'torchscript':
            model.export(format='torchscript', optimize=True)
            print(f"✓ Exported to TorchScript: {abs_model_path.replace('.pt', '.torchscript')}")
        
        else:
            print(f"Unsupported format: {format_type}")
            print("Supported formats: 'onnx', 'tensorrt', 'coreml', 'torchscript'")
            
    except Exception as e:
        print(f"❌ An error occurred during export: {e}")

def main():
    """CLI entry point for the export script."""
    import argparse
    parser = argparse.ArgumentParser(description="Export YOLOv11 model to optimized production formats")
    parser.add_argument("--model", default="models/best1.pt", help="Path to the .pt model")
    parser.add_argument("--format", default="onnx", help="onnx, tensorrt, coreml, torchscript")
    parser.add_argument("--device", default=None, help="Device for export (optional, e.g., 'cpu' or '0')")
    args = parser.parse_args()
    
    export_model_pipeline(args.model, args.format, args.device)

if __name__ == "__main__":
    main()