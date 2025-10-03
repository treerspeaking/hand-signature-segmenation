# Imports
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt

from BiRefNet.models.birefnet import BiRefNet
from ultralytics.models import YOLO

import argparse
from pathlib import Path

# Add BiRefNet directory to Python path

# import sys
# Add BiRefNet directory to Python path
# birefnet_path = Path(__file__).parent / "BiRefNet"
# sys.path.insert(0, str(birefnet_path))
# from models.birefnet import BiRefNet

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
        # Resize image to match mask dimensions
        image_resized = image.resize((mask[i].shape[1], mask[i].shape[0]), Image.Resampling.LANCZOS)
        overlay = np.array(image_resized).copy().astype(np.float32)
        
        # Create a colored mask (green for hand regions)
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask[i], 1] = 255  # Green channel
        
        # Apply opacity blending
        alpha = 0.3  # Opacity level (0.0 = transparent, 1.0 = opaque)
        overlay = overlay * (1 - alpha) + colored_mask * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=250)
        print(f"Results saved to: {save_path}")

def pred_and_show(image_path, box, model, transform_image, out_file):
    # box: left, top, right, bottom
    image = Image.open(image_path)
    w, h = image.size[:2]
    
    # Handle default box values
    if box is None or len(box) != 4:
        box = [0, 0, w, h]
    else:
        for idx_coord_value, coord_value in enumerate(box):
            if coord_value == -1:
                box[idx_coord_value] = [0, 0, w, h][idx_coord_value]

    # Create image with bounding box visualization
    image_with_box = image.copy()
    draw = ImageDraw.Draw(image_with_box)
    # Draw bounding box with red rectangle
    draw.rectangle(box, outline='red', width=3)

    # Extract cropped region
    image_crop = image.crop(box)

    input_images = transform_image(image_crop).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    canvas = torch.zeros_like(pred)
    box_to_canvas = [int(round(coord_value * (canvas.shape[-1] / w, canvas.shape[-2] / h)[idx_coord_value % 2])) for idx_coord_value, coord_value in enumerate(box)]
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(0).unsqueeze(0),
        size=(box_to_canvas[3] - box_to_canvas[1], box_to_canvas[2] - box_to_canvas[0]),
        mode='bilinear',
        align_corners=True
    ).squeeze()
    canvas[box_to_canvas[1]:box_to_canvas[3], box_to_canvas[0]:box_to_canvas[2]] = pred

    # Show Results
    pred_pil = transforms.ToPILImage()(canvas)
    
    # Create masked image
    image_masked = image.resize((1024, 1024))
    image_masked.putalpha(pred_pil)

    # Convert to numpy for visualization
    mask_array = np.array(pred_pil) > 128  # Binary threshold
    
    visualize_results(image_with_box, [mask_array], out_file)

    return pred_pil

def main():
    parser = argparse.ArgumentParser(description='Hand Segmentation Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--folder', type=str, required=True,
                       help='Path to inference folder')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save output visualization')
    
    args = parser.parse_args()
    # Load models once outside the loops
    model_yolo = YOLO(args.checkpoint)
    
    device = 'cuda'
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    model_biref = BiRefNet.from_pretrained('zhengpeng7/birefnet')
    model_biref.to(device)
    model_biref.eval()
    print('BiRefNet is ready to use.')
    
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    folder = Path(args.folder)
    out_folder = Path(args.output)

    for dir in [p for p in folder.iterdir() if p.is_dir()]:
        if "mask" in str(dir):
            continue
        out_dir = out_folder / dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for i, f in enumerate([f for f in dir.iterdir() if f.is_file()]):
            image = Image.open(f).convert('RGB')
            
            # Get YOLO predictions
            results = model_yolo(image)
            
            # Extract bounding boxes if hands are detected
            if len(results[0].boxes) > 0:
                # Get the bounding box with highest confidence
                confidences = results[0].boxes.conf
                highest_conf_idx = torch.argmax(confidences)
                box_tensor = results[0].boxes.xyxy[highest_conf_idx]  # Get highest confidence bounding box
                box = box_tensor.cpu().numpy().astype(int).tolist()
            else:
                # Use whole image if no hands detected
                box = None
            
            # Run segmentation
            final_mask = pred_and_show(f, box, model_biref, transform_image, out_dir / f"{i}_visualize.png")

if __name__ == "__main__":
    main()