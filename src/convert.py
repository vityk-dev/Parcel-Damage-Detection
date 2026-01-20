# # To convert a single image
# python convert_images.py path/to/your/image.png

# # To convert a directory of images
# python others/convert.py test/testing_on_real/Sigm

# # To specify a custom output directory
# python others/convert.py test/testing_on_real/Sigm/und_sigm test/testing_on_real/Sigm/undamaged
import os
import sys
from PIL import Image

def convert_to_640x640_jpg(input_path, output_dir=None):
    """
    Convert any image to 640x640 pixels in JPG format.
    
    Parameters:
    input_path (str): Path to input image or directory of images
    output_dir (str, optional): Directory to save converted images.
                               If None, creates a 'converted' folder in the same location.
    """
    # Check if input is a file or directory
    if os.path.isfile(input_path):
        file_paths = [input_path]
        # Default output directory is a 'converted' folder in the same directory as the input file
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_path), 'converted')
    elif os.path.isdir(input_path):
        # Get all image files in the directory
        file_paths = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    file_paths.append(os.path.join(root, file))
        
        # Default output directory is a 'converted' folder inside the input directory
        if output_dir is None:
            output_dir = os.path.join(input_path, 'converted')
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Process each file
    for file_path in file_paths:
        try:
            # Open the image
            img = Image.open(file_path)
            
            # Convert to RGB mode if it's not already (required for JPEG)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 640x640
            img_resized = img.resize((640, 640), Image.Resampling.LANCZOS)
            
            # Create output file path
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.jpg")
            
            # Save as JPEG
            img_resized.save(output_path, 'JPEG', quality=95)
            print(f"Converted: {file_path} -> {output_path}")
            
        except Exception as e:
            print(f"Error converting {file_path}: {e}")

def main():
    """Main function to process command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python convert_images.py <input_path> [output_directory]")
        print("  <input_path>: Path to an image file or a directory containing images")
        print("  [output_directory]: Optional. Directory to save converted images")
        return
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_to_640x640_jpg(input_path, output_dir)
    
if __name__ == "__main__":
    main()
