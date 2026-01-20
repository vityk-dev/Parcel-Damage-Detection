# src/dataset.py
import os
import sys
import argparse
import fiftyone as fo
from fiftyone import ViewField as F
from pathlib import Path

def parse_args():
    """Parse command line arguments for dataset management."""
    parser = argparse.ArgumentParser(description="Load parcel damage dataset with FiftyOne")
    
    parser.add_argument('--dataset_dir', type=str, default='data/dataset', 
                       help='path to dataset directory')
    parser.add_argument('--force_reload', action='store_true', 
                       help='Force deletion and reload of dataset')
    parser.add_argument('--test_only', action='store_true', 
                       help='Only load first 10 images per class for testing')
    parser.add_argument('--port', type=int, default=5151, 
                       help='Port for FiftyOne App')
    return parser.parse_args()

def run_dataset_manager(dataset_dir='data/dataset', force_reload=False, test_only=False, port=5151):
    """
    Core logic to load and visualize the parcel damage dataset using FiftyOne.
    
    Args:
        dataset_dir (str): Path to the dataset directory.
        force_reload (bool): Whether to delete and recreate the dataset.
        test_only (bool): If True, only loads a small subset of images.
        port (int): The port to launch the FiftyOne app on.
    """
    
    abs_dataset_dir = os.path.abspath(dataset_dir)
    print(f"Using dataset directory: {abs_dataset_dir}")
    
    if not os.path.exists(abs_dataset_dir):
        print(f"ERROR: Dataset directory does not exist: {abs_dataset_dir}")
        return

    dataset_name = "parcel_damage_dataset"
    
    
    if force_reload and dataset_name in fo.list_datasets():
        print(f"Deleting existing dataset {dataset_name}...")
        fo.delete_dataset(dataset_name)
    
    
    if dataset_name not in fo.list_datasets():
        print(f"Creating new dataset {dataset_name}...")
        dataset = fo.Dataset(dataset_name)
        dataset.persistent = True
        
        stats = {}
        total_added = 0
        failed_images = []
        
        for split_name in ["train", "val", "test"]:
            split_dir = os.path.join(abs_dataset_dir, split_name)
            if not os.path.exists(split_dir): continue
            
            print(f"\n📁 Processing {split_name} split...")
            stats[split_name] = {"damaged": 0, "undamaged": 0}
            
            for class_name in ["damaged", "undamaged"]:
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.exists(class_dir): continue
                
                image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
                all_images = [f for f in os.listdir(class_dir) if f.endswith(image_extensions)]
                
                if test_only:
                    all_images = all_images[:10]
                
                for img_name in all_images:
                    img_path = os.path.abspath(os.path.join(class_dir, img_name))
                    
                    try:
                        if os.path.getsize(img_path) == 0: continue
                        
                        sample = fo.Sample(filepath=img_path)
                        sample.tags = [split_name, class_name]
                        sample["ground_truth"] = fo.Classification(label=class_name)
                        sample["split"] = split_name
                        
                        dataset.add_sample(sample)
                        stats[split_name][class_name] += 1
                        total_added += 1
                        
                    except Exception as e:
                        failed_images.append(img_path)
        
        print(f"✓ Total images added: {total_added}")
        dataset.compute_metadata()
        dataset.save()
        
    else:
        print(f"Loading existing dataset {dataset_name}...")
        dataset = fo.load_dataset(dataset_name)

    
    if len(dataset) > 0:
        print(f"\n🚀 Launching FiftyOne App on http://localhost:{port}...")
        session = fo.launch_app(dataset, port=port)
        try:
            session.wait()
        except KeyboardInterrupt:
            print("\n👋 Shutting down FiftyOne app...")
    else:
        print("\n❌ Dataset is empty!")

def main():
    """CLI entry point."""
    args = parse_args()
    run_dataset_manager(
        dataset_dir=args.dataset_dir,
        force_reload=args.force_reload,
        test_only=args.test_only,
        port=args.port
    )

if __name__ == "__main__":
    main()