#!/bin/bash
set -e

# 1. Define variables
MODEL_URL="https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors?download=true"
MODEL_NAME="v1-5-pruned-emaonly-fp16.safetensors"
DEST_DIR="/home/treerspeaking/comfy/ComfyUI/models/checkpoints/"

# 2. Create the destination folder if needed
# mkdir -p "${DEST_DIR}"

# 3. Download the model (rename properly)
echo "Downloading model..."
wget -c "${MODEL_URL}" -O "${DEST_DIR}/${MODEL_NAME}"

echo "Model downloaded to ${DEST_DIR}/${MODEL_NAME}"
