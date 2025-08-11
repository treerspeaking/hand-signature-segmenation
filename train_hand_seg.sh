#!/bin/bash

# Hand Segmentation Training Script
# This script sets up and runs the training for DeepLabV3+ MobileNetV2

echo "========================================="
echo "Hand Segmentation Training Setup"
echo "========================================="

# Check if Python environment is set up
echo "Checking Python environment..."
python --version

# Install required packages
echo "Installing required packages..."
pip install torch torchvision pytorch-lightning tensorboard Pillow numpy opencv-python matplotlib scikit-learn tqdm

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p logs

# Check if dataset exists
if [ ! -d "dataset" ]; then
    echo "ERROR: Dataset directory not found!"
    echo "Please ensure your dataset is organized as:"
    echo "dataset/"
    echo "  train/"
    echo "    *.jpg (original images)"
    echo "    *_mask.jpg (segmentation masks)"
    echo "  test/"
    echo "    *.jpg (original images)"
    echo "    *_mask.jpg (segmentation masks)"
    exit 1
fi

# Check if pretrained weights exist
if [ ! -f "pretrained_weight/best_deeplabv3plus_mobilenet_cityscapes_os16.pth" ]; then
    echo "WARNING: Pretrained weights not found at pretrained_weight/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
    echo "Training will start from scratch (ImageNet pretrained backbone only)"
fi

# Run training with default parameters
echo "Starting training..."
echo "========================================="

python train.py \
    --data_root ./dataset \
    --pretrained_path ./pretrained_weight/best_deeplabv3plus_mobilenet_cityscapes_os16.pth \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 0.01 \
    --weight_decay 1e-4 \
    --input_size 512 \
    --num_workers 4 \
    --output_dir ./checkpoints \
    --log_dir ./logs

echo "========================================="
echo "Training completed!"
echo "Check ./checkpoints for saved models"
echo "Check ./logs for TensorBoard logs"
echo "To view training progress, run:"
echo "tensorboard --logdir=./logs"
echo "========================================="
