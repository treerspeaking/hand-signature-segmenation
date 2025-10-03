from typing import List
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image


class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def rle_to_mask(rle: List[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image


def process_annotation_file(annotation_path: str, output_dir: str):
    """
    Process a single annotation JSON file and save the mask
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    mask = None
    
    # Navigate through the new structure: annotations -> result
    for annotation in data.get('annotations', []):
        for result in annotation.get('result', []):
            # Check if this result contains RLE data
            if (result.get('type') == 'brushlabels' or result.get('type') == 'rectanglelabels') and 'rle' in result.get('value', {}):
                rle = result['value']['rle']
                
                # Skip empty RLE data
                if not rle or rle == [0]:
                    continue
                
                height = result['original_height']
                width = result['original_width']
                
                # Convert RLE to mask
                try:
                    current_mask = rle_to_mask(rle, height, width)
                    if mask is None:
                        mask = current_mask
                    else:
                        mask = np.maximum(mask, current_mask)
                except Exception as e:
                    print(f"  Warning: Could not convert RLE to mask: {e}")
                    continue
    
    if mask is not None:
        # Get the image filename from the data
        image_path = data['data']['image']
        
        # Extract the full path components
        # Example: "/data/local-files/?d=src/python/hand_seg/inference/renamed_data_annotate/VBHC/1019.jpg"
        # We want to extract "VBHC" (parent folder) and "1019.jpg" (filename)
        
        # Split by '/' and get the last two parts (parent_folder/filename)
        path_parts = image_path.split('/')
        parent_folder = path_parts[-2]  # e.g., "VBHC"
        filename = path_parts[-1]  # e.g., "1019.jpg"
        
        # Change extension to .png
        mask_filename = os.path.splitext(filename)[0] + '.png'
        
        # Create parent folder in output directory
        output_subfolder = os.path.join(output_dir, parent_folder)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Save mask in the parent folder
        output_path = os.path.join(output_subfolder, mask_filename)
        Image.fromarray(mask).save(output_path)
        print(f"  Saved mask: {output_path} (shape: {mask.shape})")
    else:
        print(f"  No valid mask data found")


def main():
    # Define paths
    annotation_dir = Path("renamed_new_annotation")
    output_dir = Path("renamed_data_masked")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Process all annotation files
    for annotation_file in annotation_dir.iterdir():
        if annotation_file.is_file() and annotation_file.suffix == '.json':
            print(f"Processing: {annotation_file.name}")
            try:
                process_annotation_file(str(annotation_file), str(output_dir))
            except Exception as e:
                print(f"  Error processing {annotation_file.name}: {e}")


if __name__ == "__main__":
    main()