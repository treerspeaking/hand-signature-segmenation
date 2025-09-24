import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from networks.model import deeplabv3plus_mobilenet
import segmentation_models_pytorch as smp

# from train.train import DeepLabLightningModule, HandSegDataModule
from train.train import SegmentationLightningModule
from train.train_finetune import FineTuneLightningModule

import argparse
from pathlib import Path

def load_model(checkpoint_path, num_classes=2, output_stride=8):
    # """Load trained model from checkpoint."""
    # model = deeplabv3plus_mobilenet(
    #     num_classes=num_classes,
    #     output_stride=output_stride,
    #     pretrained_backbone=False
    # )
    
    # # Load checkpoint
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # # Handle Lightning checkpoint format
    # if 'state_dict' in checkpoint:
    #     state_dict = {}
    #     for key, value in checkpoint['state_dict'].items():
    #         # Remove 'model.' prefix from Lightning checkpoints
    #         new_key = key.replace('model.', '') if key.startswith('model.') else key
    #         state_dict[new_key] = value
    # else:
    #     state_dict = checkpoint
    
    # model.load_state_dict(state_dict, strict=False)
    # model.eval()
    # model = DeepLabLightningModule(num_classes=num_classes, output_stride=output_stride).load_from_checkpoint("checkpoint_path")
    # model = DeepLabLightningModule.load_from_checkpoint(checkpoint_path)
    model = smp.create_model(
            arch="unetplusplus",
            encoder_name="mobilenet_v2",
            in_channels=3,
            classes=1
        )
    model = FineTuneLightningModule.load_from_checkpoint(checkpoint_path, model=model)
    
    # select_class = 1
    
    # # will optimize latter for multiclass if needed
    # final_seg_weight = model.model.segmentation_head[0].weight
    # final_seg_bias = model.model.segmentation_head[0].bias
    
    # new_conv = torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
    
    # with torch.no_grad():
    #     new_conv.weight.copy_(final_seg_weight[select_class])
    #     new_conv.bias.copy_(final_seg_bias[select_class])
    
    # # Replace the segmentation head
    # model.model.segmentation_head[0] = new_conv
    
    model.eval()
    return model

def visualize_results(image, mask, save_path=None):
    """Visualize original image and predicted mask."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(len(mask)):
    # Original image
        axes[i ,0].imshow(image)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Predicted mask
        axes[i, 1].imshow(mask[i], cmap='gray')
        axes[i, 1].set_title('Predicted Mask')
        axes[i, 1].axis('off')
        
        # Overlay
        overlay = np.array(image).copy().astype(np.float32)
        
        # Create a colored mask (green for hand regions)
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask[i], 1] = 255  # Green channel
        
        # Apply opacity blending
        alpha = 0.3  # Opacity level (0.0 = transparent, 1.0 = opaque)
        overlay = overlay * (1 - alpha) + colored_mask * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        # overlay[mask[i]] = [0, 128, 128]
        # overlay = overlay.astype(np.uint8)
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=250)
        print(f"Results saved to: {save_path}")
    
    # plt.show()


def preprocess_image(image_path, input_size=512):
    """Preprocess image for inference."""
    image = Image.open(image_path).convert('RGB')
    
    # Store original size
    original_size = image.size
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor, image, original_size

def postprocess_output(output, original_size, input_size=512):
    """Postprocess model output to get final mask."""
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(output)
    
    # Get predicted mask (threshold at 0.5)
    pred_mask = probs >= 0.5
    
    # Convert to numpy and remove batch dimension
    mask_np = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)
    
    # Handle multi-channel case
    if mask_np.ndim == 3:  # [channels, height, width]
        masks_resized = []
        for c in range(mask_np.shape[0]):
            # Process each channel separately
            channel_mask = mask_np[c]
            # Resize maintaining sharp edges
            resized = cv2.resize(channel_mask, original_size, interpolation=cv2.INTER_NEAREST)
            masks_resized.append(resized)
        mask_resized = np.stack(masks_resized)
    else:  # [height, width]
        # Single channel case
        mask_resized = cv2.resize(mask_np, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Convert back to boolean mask
    return mask_resized.astype(bool)

def main():
    parser = argparse.ArgumentParser(description='Hand Segmentation Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--folder', type=str, required=True,
                       help='Path to inference folder')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save output visualization')
    
    args = parser.parse_args()
    
    device = "cuda"
    model = load_model(args.checkpoint)
    model = model.to(device)
    
    folder = Path(args.folder)
    
    out_folder = Path(args.output)
    
    out_folder.mkdir(parents=True, exist_ok=True)
    
    
    
    for dir in [p for p in folder.iterdir() if p.is_dir()]:
        
        if "mask" in str(dir):
            continue
        out_dir = out_folder / dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for i, f in enumerate([f for f in dir.iterdir() if f.is_file()]):
            input_tensor, image, original_size = preprocess_image(f)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            final_mask = postprocess_output(output, original_size, 512)
            
            # out_path = out_dir / f"{i} - visualize.png"
            out_path = out_dir / f"{f.name} - visualize.png"
            
            visualize_results(image, final_mask, save_path=out_path)
            
            image.save(out_dir / f"{f.name}_og.png")
            
             # Fix: Convert PIL image to numpy array for proper indexing
            image_array = np.array(image)
            
            # Handle multi-channel mask - use the first channel or combine them
            if final_mask.ndim == 3:
                # If multiple channels, use any channel that has True values
                # combined_mask = np.any(final_mask, axis=0)
                combined_mask =  final_mask[0]
            else:
                combined_mask = final_mask
            
            # Apply the mask to set background to white
            image_array[~combined_mask] = [255, 255, 255]
            
            # Convert back to PIL Image and save
            processed_image = Image.fromarray(image_array)
            processed_image.save(out_dir / f"{f.name}.png")
        
if __name__ == '__main__':
    main()