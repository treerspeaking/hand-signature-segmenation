import cv2
import numpy as np
import os
import random
from PIL import Image, ImageFilter
import glob
from pathlib import Path
import argparse

class SyntheticImageGenerator:
    def __init__(self, mask_dir, crop_dir, output_dir="synthetic_output"):
        self.mask_dir = Path(mask_dir)
        self.crop_dir = Path(crop_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "masks_hand_signature").mkdir(exist_ok=True)
        (self.output_dir / "masks_seal").mkdir(exist_ok=True)
        (self.output_dir / "combined_masks").mkdir(exist_ok=True)
        
        # Load all available images and masks
        self.hand_signature_data = self._load_class_data("class_0_hand_signature")
        self.seal_data = self._load_class_data("class_0_seal")
        
        print(f"Loaded {len(self.hand_signature_data)} hand signature samples")
        print(f"Loaded {len(self.seal_data)} seal samples")
    
    def _load_class_data(self, class_name):
        """Load image and mask pairs for a specific class"""
        data = []
        mask_path = self.mask_dir / class_name
        crop_path = self.crop_dir / class_name
        
        if not mask_path.exists() or not crop_path.exists():
            print(f"Warning: Path not found for {class_name}")
            return data
        
        # Find all mask files
        mask_files = list(mask_path.glob("*_mask.png"))
        
        for mask_file in mask_files:
            # Find corresponding crop image
            base_name = mask_file.name.replace("_mask.png", "")
            
            # Look for corresponding crop image
            crop_candidates = [
                crop_path / f"{base_name}.png",
                crop_path / f"{base_name}.jpg",
                crop_path / f"{base_name}.jpeg"
            ]
            
            crop_file = None
            for candidate in crop_candidates:
                if candidate.exists():
                    crop_file = candidate
                    break
            
            if crop_file:
                data.append({
                    'mask': mask_file,
                    'crop': crop_file,
                    'base_name': base_name
                })
        
        return data
    
    def _load_and_preprocess_image(self, image_path, target_size=None):
        """Load and preprocess an image"""
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return img
    
    def _load_and_preprocess_mask(self, mask_path, target_size=None):
        """Load and preprocess a mask"""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        
        if target_size:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Ensure binary mask
        mask = (mask > 128).astype(np.uint8) * 255
        return mask
    
    def _rescale_to_canvas_constraints(self, image, mask, canvas_size, min_ratio=0.25, max_ratio=1.0):
        """Rescale image and mask to fit canvas size constraints (25% to 100% of canvas)"""
        h, w = image.shape[:2]
        canvas_h, canvas_w = canvas_size
        
        # Calculate current ratios
        height_ratio = h / canvas_h
        width_ratio = w / canvas_w
        current_max_ratio = max(height_ratio, width_ratio)
        
        # Determine target scale
        if current_max_ratio > max_ratio:
            # Too big - scale down to max_ratio
            scale = max_ratio / current_max_ratio
        elif current_max_ratio < min_ratio:
            # Too small - scale up to min_ratio
            scale = min_ratio / current_max_ratio
        else:
            # Already in acceptable range - apply random scaling within constraints
            max_allowed_scale = max_ratio / current_max_ratio
            min_allowed_scale = min_ratio / current_max_ratio
            scale = random.uniform(min_allowed_scale, max_allowed_scale)
        
        # Apply scaling
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Ensure minimum size (at least 10 pixels)
        new_h = max(new_h, 10)
        new_w = max(new_w, 10)
        
        # Ensure it doesn't exceed canvas size
        new_h = min(new_h, canvas_h)
        new_w = min(new_w, canvas_w)
        
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        return scaled_image, scaled_mask

    def _apply_random_transform(self, image, mask, canvas_size):
        """Apply random transformations to image and mask"""
        # First apply canvas size constraints (25% to 100% of canvas)
        image, mask = self._rescale_to_canvas_constraints(image, mask, canvas_size)
        
        # Then apply random rotation
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
        return image, mask
    
    def _blend_images_with_masks(self, background, foreground, fg_mask, position, opacity=0.8):
        """Blend foreground image onto background using mask with opacity"""
        bg_h, bg_w = background.shape[:2]
        fg_h, fg_w = foreground.shape[:2]
        
        x, y = position
        
        # Calculate valid region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + fg_w)
        y2 = min(bg_h, y + fg_h)
        
        # Calculate corresponding foreground region
        fx1 = x1 - x
        fy1 = y1 - y
        fx2 = fx1 + (x2 - x1)
        fy2 = fy1 + (y2 - y1)
        
        if x2 <= x1 or y2 <= y1:
            return background, np.zeros_like(background[:,:,0] if len(background.shape) == 3 else background)
        
        # Extract regions
        bg_region = background[y1:y2, x1:x2]
        fg_region = foreground[fy1:fy2, fx1:fx2]
        mask_region = fg_mask[fy1:fy2, fx1:fx2]
        
        # Normalize mask to [0, 1]
        mask_normalized = mask_region.astype(np.float32) / 255.0
        mask_normalized = mask_normalized * opacity
        
        # Blend
        if len(background.shape) == 3:
            mask_3d = np.stack([mask_normalized] * 3, axis=2)
            blended_region = fg_region * mask_3d + bg_region * (1 - mask_3d)
        else:
            blended_region = fg_region * mask_normalized + bg_region * (1 - mask_normalized)
        
        # Create result
        result = background.copy()
        result[y1:y2, x1:x2] = blended_region.astype(background.dtype)
        
        # Create output mask for this object
        output_mask = np.zeros_like(background[:,:,0] if len(background.shape) == 3 else background)
        output_mask[y1:y2, x1:x2] = mask_region
        
        return result, output_mask
    
    def _create_background(self, size=(512, 512)):
        """Create a background image"""
        # Create a white background with some subtle texture
        background = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        
        # Add some subtle noise for realism
        noise = np.random.normal(0, 5, background.shape).astype(np.int16)
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return background
    
    def generate_synthetic_image(self, num_hand_signatures=None, num_seals=None, 
                                canvas_size=(512, 512), output_name=None):
        """Generate a single synthetic image with overlapping objects"""
        
        # Default number of objects if not specified
        if num_hand_signatures is None:
            num_hand_signatures = random.randint(1, 3)
        if num_seals is None:
            num_seals = random.randint(1, 2)
        
        # Create background
        canvas = self._create_background(canvas_size)
        
        # Initialize combined masks
        hand_signature_mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
        seal_mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
        
        # Add hand signatures
        for i in range(num_hand_signatures):
            if not self.hand_signature_data:
                continue
                
            # Select random hand signature
            data = random.choice(self.hand_signature_data)
            
            # Load image and mask
            img = self._load_and_preprocess_image(data['crop'])
            mask = self._load_and_preprocess_mask(data['mask'])
            
            if img is None or mask is None:
                continue
            
            # Apply random transformations
            img, mask = self._apply_random_transform(img, mask, canvas_size)
            
            # Random position
            max_x = max(0, canvas_size[0] - img.shape[1])
            max_y = max(0, canvas_size[1] - img.shape[0])
            position = (random.randint(0, max_x), random.randint(0, max_y))
            
            # Random opacity for blending
            opacity = random.uniform(0.6, 0.9)
            
            # Blend onto canvas
            canvas, obj_mask = self._blend_images_with_masks(canvas, img, mask, position, opacity)
            
            # Add to hand signature mask
            hand_signature_mask = np.maximum(hand_signature_mask, obj_mask)
        
        # Add seals
        for i in range(num_seals):
            if not self.seal_data:
                continue
                
            # Select random seal
            data = random.choice(self.seal_data)
            
            # Load image and mask
            # Load image and mask
            img = self._load_and_preprocess_image(data['crop'])
            mask = self._load_and_preprocess_mask(data['mask'])
            
            if img is None or mask is None:
                continue
            
            # Apply random transformations
            img, mask = self._apply_random_transform(img, mask, canvas_size)
            
            # Random position
            max_x = max(0, canvas_size[0] - img.shape[1])
            max_y = max(0, canvas_size[1] - img.shape[0])
            position = (random.randint(0, max_x), random.randint(0, max_y))
            
            # Random opacity for blending
            opacity = random.uniform(0.7, 0.95)
            
            # Blend onto canvas
            canvas, obj_mask = self._blend_images_with_masks(canvas, img, mask, position, opacity)
            
            # Add to seal mask
            seal_mask = np.maximum(seal_mask, obj_mask)
        
        # Create combined mask (different values for each class)
        combined_mask = np.zeros_like(hand_signature_mask)
        combined_mask[hand_signature_mask > 0] = 1  # Hand signatures = 1
        combined_mask[seal_mask > 0] = 2  # Seals = 2
        
        # Handle overlapping regions (seal takes priority)
        overlap_mask = (hand_signature_mask > 0) & (seal_mask > 0)
        combined_mask[overlap_mask] = 2
        
        return canvas, hand_signature_mask, seal_mask, combined_mask
    
    def generate_dataset(self, num_images=100, canvas_size=(512, 512)):
        """Generate a dataset of synthetic images"""
        print(f"Generating {num_images} synthetic images...")
        
        for i in range(num_images):
            # Generate synthetic image
            canvas, hand_mask, seal_mask, combined_mask = self.generate_synthetic_image(
                canvas_size=canvas_size
            )
            
            # Save files
            image_name = f"synthetic_{i:04d}"
            
            # Save image
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.output_dir / "images" / f"{image_name}.png"), canvas_bgr)
            
            # Save individual masks
            cv2.imwrite(str(self.output_dir / "masks_hand_signature" / f"{image_name}.png"), hand_mask)
            cv2.imwrite(str(self.output_dir / "masks_seal" / f"{image_name}.png"), seal_mask)
            
            # Save combined mask
            cv2.imwrite(str(self.output_dir / "combined_masks" / f"{image_name}.png"), combined_mask)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_images} images")
        
        print(f"Dataset generation complete! Saved to {self.output_dir}")
        print(f"Generated files:")
        print(f"  - Images: {self.output_dir}/images/")
        print(f"  - Hand signature masks: {self.output_dir}/masks_hand_signature/")
        print(f"  - Seal masks: {self.output_dir}/masks_seal/")
        print(f"  - Combined masks: {self.output_dir}/combined_masks/")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic images with overlapping hand signatures and seals")
    parser.add_argument("--mask_dir", default="/home/treerspeaking/src/python/hand_seg/mask", help="Directory containing mask files")
    parser.add_argument("--crop_dir", default="/home/treerspeaking/src/python/hand_seg/mask", help="Directory containing crop files")
    parser.add_argument("--output_dir", default="/home/treerspeaking/src/python/hand_seg/synthetic_output", help="Output directory")
    parser.add_argument("--num_images", type=int, default=5000, help="Number of images to generate")
    parser.add_argument("--canvas_width", type=int, default=512, help="Canvas width")
    parser.add_argument("--canvas_height", type=int, default=512, help="Canvas height")
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticImageGenerator(
        mask_dir=args.mask_dir,
        crop_dir=args.crop_dir,
        output_dir=args.output_dir
    )
    
    # Generate dataset
    generator.generate_dataset(
        num_images=args.num_images,
        canvas_size=(args.canvas_width, args.canvas_height)
    )

if __name__ == "__main__":
    main()