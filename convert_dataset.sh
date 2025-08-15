#!/bin/bash
# filepath: /home/treerspeaking/src/python/hand_seg/convert_dataset.sh

# Script to convert old dataset format to new dataset format
# Old format: train/2500.jpg, train/2500_mask.jpg, train/2500_gt.jpg
# New format: train/images/2500.jpg, train/masks_*/2500.jpg, train/gt/2500.jpg

# Remove set -e to prevent script from exiting on first error
# set -e  # Exit on any error

# Function to convert a directory
convert_directory() {
    local source_dir="$1"
    local target_dir="$2"
    
    echo "Converting $source_dir to $target_dir..."
    
    # Create target directory structure
    mkdir -p "$target_dir/images"
    mkdir -p "$target_dir/masks_hand_signature" 
    mkdir -p "$target_dir/gt"
    
    # Debug: Check if source directory exists and has files
    if [[ ! -d "$source_dir" ]]; then
        echo "Error: Source directory $source_dir does not exist"
        return 1
    fi
    
    # Count base images directly using find
    local total_images=$(find "$source_dir" -maxdepth 1 -name "*.jpg" ! -name "*_mask.jpg" ! -name "*_gt.jpg" | wc -l)
    local count=0
    
    echo "Found $total_images base images to process..."
    
    # Check if we found any images
    if [[ $total_images -eq 0 ]]; then
        echo "No base images found in $source_dir"
        return 0
    fi
    
    # Process each base image file using while loop with find
    find "$source_dir" -maxdepth 1 -name "*.jpg" ! -name "*_mask.jpg" ! -name "*_gt.jpg" | while read -r image_file; do
        # Extract base filename without extension
        local basename=$(basename "$image_file" .jpg)
        local mask_file="$source_dir/${basename}_mask.jpg"
        local gt_file="$source_dir/${basename}_gt.jpg"
        
        echo "Processing: $basename"
        
        # Copy base image to images folder
        if [[ -f "$image_file" ]]; then
            cp "$image_file" "$target_dir/images/" || echo "  ✗ Failed to copy $basename.jpg"
            echo "  ✓ Copied: $basename.jpg -> images/"
        else
            echo "  ✗ Error: Base image not found: $image_file"
            continue
        fi
        
        # Copy mask file to masks folder (if exists)
        if [[ -f "$mask_file" ]]; then
            cp "$mask_file" "$target_dir/masks_hand_signature/${basename}.jpg" || echo "  ✗ Failed to copy mask for $basename"
            echo "  ✓ Copied: ${basename}_mask.jpg -> masks_hand_signature/${basename}.jpg"
        else
            echo "  ⚠ Warning: Mask file not found for $basename"
        fi
        
        # Copy gt file to gt folder (if exists)
        if [[ -f "$gt_file" ]]; then
            cp "$gt_file" "$target_dir/gt/${basename}.jpg" || echo "  ✗ Failed to copy GT for $basename"
            echo "  ✓ Copied: ${basename}_gt.jpg -> gt/${basename}.jpg"
        else
            echo "  ⚠ Warning: GT file not found for $basename"
        fi
        
        ((count++))
        if (( count % 50 == 0 )); then
            echo "  Progress: $count/$total_images"
        fi
    done
    
    echo "Conversion of $source_dir completed!"
    echo "Images: $(find "$target_dir/images" -name "*.jpg" 2>/dev/null | wc -l)"
    echo "Masks: $(find "$target_dir/masks_hand_signature" -name "*.jpg" 2>/dev/null | wc -l)"
    echo "GT: $(find "$target_dir/gt" -name "*.jpg" 2>/dev/null | wc -l)"
    echo ""
}

# Main script
OLD_DATASET_PATH="/home/treerspeaking/src/python/hand_seg/dataset/old_dataset"
NEW_DATASET_PATH="/home/treerspeaking/src/python/hand_seg/dataset/converted_dataset"

echo "Starting dataset conversion..."
echo "Source: $OLD_DATASET_PATH"
echo "Target: $NEW_DATASET_PATH"
echo ""

# Check if old dataset exists
if [[ ! -d "$OLD_DATASET_PATH" ]]; then
    echo "Error: Old dataset directory not found at $OLD_DATASET_PATH"
    exit 1
fi

# Create new dataset root directory
mkdir -p "$NEW_DATASET_PATH"

# Convert train directory
if [[ -d "$OLD_DATASET_PATH/train" ]]; then
    echo "Processing train directory..."
    convert_directory "$OLD_DATASET_PATH/train" "$NEW_DATASET_PATH/train"
else
    echo "Warning: Train directory not found in old dataset"
fi

# Convert test directory  
if [[ -d "$OLD_DATASET_PATH/test" ]]; then
    echo "Processing test directory..."
    convert_directory "$OLD_DATASET_PATH/test" "$NEW_DATASET_PATH/test"
else
    echo "Warning: Test directory not found in old dataset"
fi

echo "Dataset conversion completed!"
echo "New dataset structure created at: $NEW_DATASET_PATH"
echo ""
echo "Final summary:"
if [[ -d "$NEW_DATASET_PATH/train" ]]; then
    echo "Train - Images: $(find "$NEW_DATASET_PATH/train/images" -name "*.jpg" 2>/dev/null | wc -l)"
    echo "Train - Masks: $(find "$NEW_DATASET_PATH/train/masks_hand_signature" -name "*.jpg" 2>/dev/null | wc -l)"
    echo "Train - GT: $(find "$NEW_DATASET_PATH/train/gt" -name "*.jpg" 2>/dev/null | wc -l)"
fi
if [[ -d "$NEW_DATASET_PATH/test" ]]; then
    echo "Test - Images: $(find "$NEW_DATASET_PATH/test/images" -name "*.jpg" 2>/dev/null | wc -l)"
    echo "Test - Masks: $(find "$NEW_DATASET_PATH/test/masks_hand_signature" -name "*.jpg" 2>/dev/null | wc -l)"
    echo "Test - GT: $(find "$NEW_DATASET_PATH/test/gt" -name "*.jpg" 2>/dev/null | wc -l)"
fi

echo ""
echo "Directory structure:"
tree "$NEW_DATASET_PATH" -L 3 2>/dev/null || find "$NEW_DATASET_PATH" -type d | head -10