#!/usr/bin/env python3
"""Convert all .jpg files to .png format using PIL."""

from pathlib import Path
from PIL import Image

def convert_jpg_to_png():
    """Convert all .jpg files in current directory to .png format."""
    current_dir = Path(".")
    jpg_files = list(current_dir.glob("*.jpg"))
    
    if not jpg_files:
        print("No .jpg files found in current directory")
        return
    
    for jpg_file in jpg_files:
        png_file = jpg_file.with_suffix(".png")
        
        try:
            with Image.open(jpg_file) as img:
                img.save(png_file, "PNG")
            print(f"Converted: {jpg_file} -> {png_file}")
        except Exception as e:
            print(f"Error converting {jpg_file}: {e}")

if __name__ == "__main__":
    convert_jpg_to_png()