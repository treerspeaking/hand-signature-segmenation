import os
import shutil
from pathlib import Path

def move_files_to_test_dataset():
    """
    Move images from renamed_data/BHXH to test/images
    Move masks from renamed_data_masked/BHXH to test/masks_hand_signature
    """
    # Define source and destination paths
    source_images = Path("inference/renamed_data/BHXH")
    source_masks = Path("inference/renamed_data_masked/BHXH")
    
    dest_images = Path("dataset/unlabeled-dataset/test/images")
    dest_masks = Path("dataset/unlabeled-dataset/test/masks_hand_signature")
    
    # Create destination directories if they don't exist
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_masks.mkdir(parents=True, exist_ok=True)
    
    # Get list of mask files (these are the labeled ones)
    mask_files = sorted(source_masks.glob("*.png"))
    
    moved_count = 0
    skipped_count = 0
    
    for mask_file in mask_files:
        filename = mask_file.name
        
        # Check if corresponding image exists
        image_file = source_images / filename
        
        if image_file.exists():
            # Move image to test/images
            dest_image_path = dest_images / filename
            shutil.copy2(image_file, dest_image_path)
            
            # Move mask to test/masks_hand_signature
            dest_mask_path = dest_masks / filename
            shutil.copy2(mask_file, dest_mask_path)
            
            moved_count += 1
            print(f"Moved: {filename}")
        else:
            skipped_count += 1
            print(f"Skipped: {filename} (no corresponding image found)")
    
    print(f"\n✓ Successfully moved {moved_count} image-mask pairs")
    print(f"✗ Skipped {skipped_count} files")
    print(f"\nImages location: {dest_images}")
    print(f"Masks location: {dest_masks}")

if __name__ == "__main__":
    move_files_to_test_dataset()