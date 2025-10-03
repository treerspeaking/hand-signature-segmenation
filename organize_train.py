import os
import shutil
from pathlib import Path

def organize_train_dataset():
    """
    Organize DEGREE and VBHC datasets:
    - Move labeled images (with masks) to train/images and train/masks_hand_signature
    - Move unlabeled images (without masks) to train/unlabeled_images
    - Add dataset prefix to avoid filename collisions (VBHC_*.png, DEGREE_*.png)
    """
    
    # Define dataset configurations
    datasets = [
        {
            'name': 'VBHC',
            'prefix': 'VBHC',
            'source_images': Path("inference/renamed_data/VBHC"),
            'source_masks': Path("inference/renamed_data_masked/VBHC"),
            'image_ext': '.jpg',
            'mask_ext': '.png'
        },
        {
            'name': 'DEGREE',
            'prefix': 'DEGREE',
            'source_images': Path("inference/renamed_data/DEGREE"),
            'source_masks': Path("inference/renamed_data_masked/DEGREE"),
            'image_ext': '.png',
            'mask_ext': '.png'
        }
    ]
    
    # Define destination paths
    dest_labeled_images = Path("dataset/unlabeled-dataset/train/images")
    dest_masks = Path("dataset/unlabeled-dataset/train/masks_hand_signature")
    dest_unlabeled_images = Path("dataset/unlabeled-dataset/train/unlabeled_images")
    
    # Create destination directories
    dest_labeled_images.mkdir(parents=True, exist_ok=True)
    dest_masks.mkdir(parents=True, exist_ok=True)
    dest_unlabeled_images.mkdir(parents=True, exist_ok=True)
    
    total_labeled = 0
    total_unlabeled = 0
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {dataset['name']} dataset")
        print(f"{'='*50}")
        
        source_images = dataset['source_images']
        source_masks = dataset['source_masks']
        image_ext = dataset['image_ext']
        mask_ext = dataset['mask_ext']
        prefix = dataset['prefix']
        
        # Get all image files
        image_files = sorted(source_images.glob(f"*{image_ext}"))
        
        # Get all mask filenames (without extension)
        mask_files = set()
        for mask_file in source_masks.glob(f"*{mask_ext}"):
            mask_files.add(mask_file.stem)  # filename without extension
        
        labeled_count = 0
        unlabeled_count = 0
        
        for image_file in image_files:
            filename_stem = image_file.stem  # filename without extension
            
            # Add prefix to avoid collisions
            dest_image_name = f"{prefix}_{filename_stem}.png"
            dest_mask_name = f"{prefix}_{filename_stem}.png"
            
            # Check if this image has a corresponding mask
            if filename_stem in mask_files:
                # This is a LABELED image
                # Copy image to train/images (convert to PNG if needed)
                dest_image_path = dest_labeled_images / dest_image_name
                if image_ext == '.jpg':
                    # Convert JPG to PNG
                    from PIL import Image
                    img = Image.open(image_file)
                    img.save(dest_image_path)
                else:
                    shutil.copy2(image_file, dest_image_path)
                
                # Copy mask to train/masks_hand_signature
                source_mask_path = source_masks / f"{filename_stem}{mask_ext}"
                dest_mask_path = dest_masks / dest_mask_name
                shutil.copy2(source_mask_path, dest_mask_path)
                
                labeled_count += 1
                print(f"✓ Labeled: {filename_stem}{image_ext} -> {dest_image_name}")
                
            else:
                # This is an UNLABELED image
                dest_unlabeled_path = dest_unlabeled_images / dest_image_name
                
                if image_ext == '.jpg':
                    # Convert JPG to PNG
                    from PIL import Image
                    img = Image.open(image_file)
                    img.save(dest_unlabeled_path)
                else:
                    shutil.copy2(image_file, dest_unlabeled_path)
                
                unlabeled_count += 1
                if unlabeled_count <= 5:  # Show first 5 examples
                    print(f"○ Unlabeled: {filename_stem}{image_ext} -> {dest_image_name}")
        
        if unlabeled_count > 5:
            print(f"○ ... and {unlabeled_count - 5} more unlabeled images")
        
        print(f"\n{dataset['name']} Summary:")
        print(f"  Labeled: {labeled_count}")
        print(f"  Unlabeled: {unlabeled_count}")
        
        total_labeled += labeled_count
        total_unlabeled += unlabeled_count
    
    print(f"\n{'='*50}")
    print(f"FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"✓ Total labeled pairs moved: {total_labeled}")
    print(f"○ Total unlabeled images moved: {total_unlabeled}")
    print(f"\nDestination folders:")
    print(f"  Labeled images: {dest_labeled_images}")
    print(f"  Masks: {dest_masks}")
    print(f"  Unlabeled images: {dest_unlabeled_images}")
    print(f"\nFilename format:")
    print(f"  VBHC files: VBHC_*.png")
    print(f"  DEGREE files: DEGREE_*.png")

if __name__ == "__main__":
    organize_train_dataset()