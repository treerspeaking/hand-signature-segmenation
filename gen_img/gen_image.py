import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from my_default_augraphy import my_default_augraphy
import yaml

import os
import random
import glob
from pathlib import Path
import argparse
import json
from collections import defaultdict



class SyntheticImageGenerator:
    def __init__(self, mask_dir, crop_dir, output_dir="synthetic_output", split_mode=None, split_ratio=0.8, split_file=None, same_color_ratio=0.0):
        self.mask_dir = Path(mask_dir)
        self.crop_dir = Path(crop_dir)
        self.output_dir = Path(output_dir)
        self.split_mode = split_mode  # None, 'train', 'test', or 'create_split'
        self.split_ratio = split_ratio  # Train ratio (0.8 = 80% train, 20% test)
        self.split_file = split_file or "data_split.json"
        self.same_color_ratio = same_color_ratio  # Ratio of images where seal and hand signature have same color
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "images").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "masks_hand_signature").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "masks_seal").mkdir(exist_ok=True, parents=True)
        
        # Load all available images and masks with split handling
        self.hand_signature_data = self._load_class_data_with_split("class_0_hand_signature")
        self.seal_data = self._load_class_data_with_split("class_0_seal")
        
        print(f"Loaded {len(self.hand_signature_data)} hand signature samples")
        print(f"Loaded {len(self.seal_data)} seal samples")
        
        if self.same_color_ratio > 0:
            print(f"Same color feature enabled: {self.same_color_ratio*100:.1f}% of images will have matching colors")
        
        if self.split_mode:
            print(f"Using split mode: {self.split_mode}")
            
        self.augraphy = my_default_augraphy()
    
    def _extract_source_id(self, filename):
        """Extract the source image identifier from filename to group crops from same source"""
        # Pattern: google_0000_png.rf.3e94d56d616e5ff52d02ae57c7d44cff_crop_000_hand_signature.jpg
        # We want: google_0000_png.rf.3e94d56d616e5ff52d02ae57c7d44cff
        parts = filename.split('_crop_')
        if len(parts) >= 2:
            return parts[0]
        return filename  # fallback
    
    def _create_or_load_split(self):
        """Create a new data split or load existing one"""
        split_path = Path(self.split_file)
        
        if split_path.exists() and self.split_mode != 'create_split':
            # Load existing split
            with open(split_path, 'r') as f:
                split_data = json.load(f)
            print(f"Loaded existing split from {split_path}")
            return split_data
        
        # Create new split
        print(f"Creating new data split with ratio {self.split_ratio}")
        
        # Group by source images for hand signatures
        hand_signature_sources = defaultdict(list)
        for data in self._load_class_data("class_0_hand_signature"):
            source_id = self._extract_source_id(data['base_name'])
            hand_signature_sources[source_id].append(data)
        
        # Group by source images for seals
        seal_sources = defaultdict(list)
        for data in self._load_class_data("class_0_seal"):
            source_id = self._extract_source_id(data['base_name'])
            seal_sources[source_id].append(data)
        
        # Split source images (not individual crops)
        hand_source_ids = list(hand_signature_sources.keys())
        seal_source_ids = list(seal_sources.keys())
        
        random.shuffle(hand_source_ids)
        random.shuffle(seal_source_ids)
        
        # Calculate split points
        hand_train_count = int(len(hand_source_ids) * self.split_ratio)
        seal_train_count = int(len(seal_source_ids) * self.split_ratio)
        
        split_data = {
            'hand_signature': {
                'train': hand_source_ids[:hand_train_count],
                'test': hand_source_ids[hand_train_count:]
            },
            'seal': {
                'train': seal_source_ids[:seal_train_count],
                'test': seal_source_ids[seal_train_count:]
            },
            'split_ratio': self.split_ratio,
            'total_hand_sources': len(hand_source_ids),
            'total_seal_sources': len(seal_source_ids)
        }
        
        # Save split data
        with open(split_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"Created data split:")
        print(f"  Hand signatures: {len(split_data['hand_signature']['train'])} train sources, {len(split_data['hand_signature']['test'])} test sources")
        print(f"  Seals: {len(split_data['seal']['train'])} train sources, {len(split_data['seal']['test'])} test sources")
        print(f"  Split saved to: {split_path}")
        
        return split_data
    
    def _load_class_data_with_split(self, class_name):
        """Load image and mask pairs for a specific class with train/test split handling"""
        all_data = self._load_class_data(class_name)
        
        if not self.split_mode:
            return all_data
        
        # Always load or create split when split_mode is specified
        split_data = self._create_or_load_split()
        
        if self.split_mode == 'create_split':
            return all_data  # Return all data when just creating split
        
        # Determine which source IDs to use based on split mode
        class_key = 'hand_signature' if 'hand_signature' in class_name else 'seal'
        if self.split_mode == 'train':
            allowed_sources = set(split_data[class_key]['train'])
        elif self.split_mode == 'test':
            allowed_sources = set(split_data[class_key]['test'])
        else:
            return all_data  # fallback
        
        # Filter data to only include crops from allowed source images
        filtered_data = []
        for data in all_data:
            source_id = self._extract_source_id(data['base_name'])
            if source_id in allowed_sources:
                filtered_data.append(data)
        
        print(f"  Filtered {class_name}: {len(filtered_data)}/{len(all_data)} samples for {self.split_mode} split")
        return filtered_data
    
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
    
    def _apply_color_jitter(
            self, 
            image, 
            brightness_range=(0.8, 1.2), 
            contrast_range=(0.8, 1.2), 
            saturation_range=(0.8, 1.2), 
            hue_shift_range=(-10, 10),
            jitter_strength=230
        ):
        """Apply color jittering to make images more diverse, including pixel noise before hue shift."""
        if len(image.shape) != 3:
            return image
            
        # Convert to PIL Image for easier color manipulation
        pil_image = Image.fromarray(image)
        
        # Random brightness
        brightness_factor = random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
        
        # Random contrast
        contrast_factor = random.uniform(*contrast_range)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
        
        # Random saturation (only for color images)
        if pil_image.mode == 'RGB':
            saturation_factor = random.uniform(*saturation_range)
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(saturation_factor)
        
        # Convert back to numpy array
        jittered_image = np.array(pil_image)
        
        # ---- Add pixel-wise jitter noise BEFORE hue shift ----
        noise = np.random.randint(-jitter_strength, jitter_strength + 1, 3, dtype=np.int16)
        jittered_image = np.clip(jittered_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Ensure black pixels become visible
        # black_mask = np.all(jittered_image == [0, 0, 0], axis=-1)
        # jittered_image[black_mask] = np.random.randint(50, 256, (black_mask.sum(), 3), dtype=np.uint8)
        
        # Optional: Add slight hue shift
        if random.random() < 0.5 and len(jittered_image.shape) == 3:  # 50% chance of hue shift
            hsv_image = cv2.cvtColor(jittered_image, cv2.COLOR_RGB2HSV)
            hue_shift = random.randint(*hue_shift_range)
            hsv_image[:, :, 0] = (hsv_image[:, :, 0].astype(np.int16) + hue_shift) % 180
            jittered_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        
        return jittered_image

    
    def _apply_hand_signature_color_jitter(self, image):
        """Apply color jittering specifically tuned for hand signatures (more conservative)"""
        return self._apply_color_jitter(
            image, 
            brightness_range=(0.85, 1.15),  # More conservative brightness
            contrast_range=(0.9, 1.1),     # Less contrast variation
            saturation_range=(0.9, 1.1),   # Subtle saturation changes
            hue_shift_range=(-5, 5)        # Small hue shifts
        )
    
    def _apply_seal_color_jitter(self, image):
        """Apply color jittering specifically tuned for seals (more variation allowed)"""
        return self._apply_color_jitter(
            image,
            brightness_range=(0.75, 1.25),  # More brightness variation
            contrast_range=(0.8, 1.3),     # Higher contrast variation
            saturation_range=(0.7, 1.3),   # More saturation changes
            hue_shift_range=(-15, 15)      # Larger hue shifts
        )
    
    def _calculate_mean_color(self, image, mask):
        """Calculate the mean color of pixels in the masked region"""
        if len(image.shape) != 3:
            return None
        
        # Create boolean mask
        bool_mask = mask > 128
        
        if not np.any(bool_mask):
            return None
        
        # Calculate mean color for each channel
        mean_color = np.mean(image[bool_mask], axis=0)
        return mean_color.astype(np.float32)
    
    def _shift_color_to_target(self, image, mask, target_color, current_mean_color):
        """Shift all pixels in the masked region to have the target mean color"""
        if len(image.shape) != 3 or current_mean_color is None:
            return image
        
        # Create boolean mask
        bool_mask = mask > 128
        
        if not np.any(bool_mask):
            return image
        
        # Calculate the shift needed
        color_shift = target_color - current_mean_color
        
        # Apply shift to masked pixels
        result_image = image.copy().astype(np.float32)
        result_image[bool_mask] += color_shift
        
        # Clamp to valid range
        result_image = np.clip(result_image, 0, 255)
        
        return result_image.astype(np.uint8)
    
    def _apply_same_color_transformation(self, hand_images, hand_masks, seal_images, seal_masks):
        """Apply same color transformation to make seal and hand signature have the same color"""
        if not hand_images or not seal_images:
            return hand_images, seal_images
        
        # Step 1: Sample a random RGB color
        target_color = np.random.randint(50, 206, 3).astype(np.float32)  # Avoid extreme colors
        
        # Step 2: Calculate mean colors for all hand signatures and seals
        hand_mean_colors = []
        seal_mean_colors = []
        
        for img, mask in zip(hand_images, hand_masks):
            mean_color = self._calculate_mean_color(img, mask)
            hand_mean_colors.append(mean_color)
        
        for img, mask in zip(seal_images, seal_masks):
            mean_color = self._calculate_mean_color(img, mask)
            seal_mean_colors.append(mean_color)
        
        # Step 3 & 4: Shift colors of both hand signatures and seals to the target color
        transformed_hand_images = []
        for img, mask, mean_color in zip(hand_images, hand_masks, hand_mean_colors):
            if mean_color is not None:
                img = self._shift_color_to_target(img, mask, target_color, mean_color)
            transformed_hand_images.append(img)
        
        transformed_seal_images = []
        for img, mask, mean_color in zip(seal_images, seal_masks, seal_mean_colors):
            if mean_color is not None:
                img = self._shift_color_to_target(img, mask, target_color, mean_color)
            transformed_seal_images.append(img)
        
        return transformed_hand_images, transformed_seal_images
    
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

    def _apply_random_transform(self, image, mask, canvas_size, object_type="general"):
        """Apply random transformations to image and mask"""
        # First apply canvas size constraints (25% to 100% of canvas)
        image, mask = self._rescale_to_canvas_constraints(image, mask, canvas_size)
        
        # Apply object-specific color jittering before rotation (to avoid artifacts)
        if random.random() < 0.7:  # 70% chance of color jittering
            if object_type == "hand_signature":
                image = self._apply_hand_signature_color_jitter(image)
            elif object_type == "seal":
                image = self._apply_seal_color_jitter(image)
            else:
                image = self._apply_color_jitter(image)
        
        # Then apply random rotation
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
        return image, mask
    
    def _apply_random_transform_no_color(self, image, mask, canvas_size):
        """Apply random transformations to image and mask without color jittering"""
        # First apply canvas size constraints (25% to 100% of canvas)
        image, mask = self._rescale_to_canvas_constraints(image, mask, canvas_size)
        
        # Apply random rotation (no color jittering)
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
        return image, mask
    
    def _apply_augraphy(self, image):
        return self.augraphy(image)
    
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
        
        # Determine if this image should have same color for seal and hand signature
        apply_same_color = random.random() < self.same_color_ratio
        
        # Create background
        canvas = self._create_background(canvas_size)
        
        # Initialize combined masks
        hand_signature_mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
        seal_mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
        
        # Collect all hand signature images and masks before processing
        hand_images = []
        hand_masks = []
        hand_positions = []
        hand_opacities = []
        
        # Prepare hand signatures
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
            
            # Apply random transformations (but skip color jittering if same_color will be applied)
            if apply_same_color:
                # Apply transformations without color jittering
                img, mask = self._apply_random_transform_no_color(img, mask, canvas_size)
            else:
                # Apply normal transformations with color jittering
                img, mask = self._apply_random_transform(img, mask, canvas_size, object_type="hand_signature")
            
            # Random position
            max_x = max(0, canvas_size[0] - img.shape[1])
            max_y = max(0, canvas_size[1] - img.shape[0])
            position = (random.randint(0, max_x), random.randint(0, max_y))
            
            # Random opacity for blending
            opacity = random.uniform(0.8, 0.99)
            
            hand_images.append(img)
            hand_masks.append(mask)
            hand_positions.append(position)
            hand_opacities.append(opacity)
        
        # Collect all seal images and masks before processing
        seal_images = []
        seal_masks_list = []
        seal_positions = []
        seal_opacities = []
        
        # Prepare seals
        for i in range(num_seals):
            if not self.seal_data:
                continue
                
            # Select random seal
            data = random.choice(self.seal_data)
            
            # Load image and mask
            img = self._load_and_preprocess_image(data['crop'])
            mask = self._load_and_preprocess_mask(data['mask'])
            
            if img is None or mask is None:
                continue
            
            # Apply random transformations (but skip color jittering if same_color will be applied)
            if apply_same_color:
                # Apply transformations without color jittering
                img, mask = self._apply_random_transform_no_color(img, mask, canvas_size)
            else:
                # Apply normal transformations with color jittering
                img, mask = self._apply_random_transform(img, mask, canvas_size, object_type="seal")
            
            # Random position
            max_x = max(0, canvas_size[0] - img.shape[1])
            max_y = max(0, canvas_size[1] - img.shape[0])
            position = (random.randint(0, max_x), random.randint(0, max_y))
            
            # Random opacity for blending
            opacity = random.uniform(0.7, 0.95)
            
            seal_images.append(img)
            seal_masks_list.append(mask)
            seal_positions.append(position)
            seal_opacities.append(opacity)
        
        # Apply same color transformation if enabled
        if apply_same_color and hand_images and seal_images:
            hand_images, seal_images = self._apply_same_color_transformation(
                hand_images, hand_masks, seal_images, seal_masks_list
            )
        
        # Blend hand signatures onto canvas
        for img, mask, position, opacity in zip(hand_images, hand_masks, hand_positions, hand_opacities):
            canvas, obj_mask = self._blend_images_with_masks(canvas, img, mask, position, opacity)
            hand_signature_mask = np.maximum(hand_signature_mask, obj_mask)
        
        # Blend seals onto canvas
        for img, mask, position, opacity in zip(seal_images, seal_masks_list, seal_positions, seal_opacities):
            canvas, obj_mask = self._blend_images_with_masks(canvas, img, mask, position, opacity)
            seal_mask = np.maximum(seal_mask, obj_mask)
        
        # Create combined mask (different values for each class)
        combined_mask = np.zeros_like(hand_signature_mask)
        combined_mask[hand_signature_mask > 0] = 1  # Hand signatures = 1
        combined_mask[seal_mask > 0] = 2  # Seals = 2
        
        # Handle overlapping regions (seal takes priority)
        overlap_mask = (hand_signature_mask > 0) & (seal_mask > 0)
        combined_mask[overlap_mask] = 2
        
        # apply augraphy aug 
        # canvas = self._apply_augraphy(canvas)
        
        return canvas, hand_signature_mask, seal_mask, combined_mask
    
    def generate_dataset(self, num_images=100, canvas_size=(512, 512)):
        """Generate a dataset of synthetic images"""
        print(f"Generating {num_images} synthetic images...")
        
        same_color_count = 0
        
        for i in range(num_images):
            # Track if same color was applied for statistics
            original_same_color_ratio = self.same_color_ratio
            will_apply_same_color = random.random() < self.same_color_ratio
            if will_apply_same_color:
                same_color_count += 1
            
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
            # cv2.imwrite(str(self.output_dir / "combined_masks" / f"{image_name}.png"), combined_mask)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_images} images")
        
        print(f"Dataset generation complete! Saved to {self.output_dir}")
        if self.same_color_ratio > 0:
            print(f"Applied same color to {same_color_count}/{num_images} images ({same_color_count/num_images*100:.1f}%)")
        print("Generated files:")
        print(f"  - Images: {self.output_dir}/images/")
        print(f"  - Hand signature masks: {self.output_dir}/masks_hand_signature/")
        print(f"  - Seal masks: {self.output_dir}/masks_seal/")
        # print(f"  - Combined masks: {self.output_dir}/combined_masks/")
    
    def generate_dataset_yaml(self, train_dir="train", val_dir="val"):
        """Generates a YAML file describing the dataset."""
        
        # The YAML file is typically placed one level above the output directory
        # e.g., if output_dir is 'synthetic_output/train', yaml is in 'synthetic_output'
        yaml_path = self.output_dir.parent / f"{self.output_dir.parent.name}.yaml"
        
        # Define class names. Note: In segmentation, 0 is often background.
        # The script assigns 1 to hand_signature and 2 to seal.
        class_names = {
            0: 'hand_signature',
            1: 'seal'
        }

        # Create the data structure for the YAML file
        dataset_info = {
            'path': str(self.output_dir.parent.resolve()),
            'train': str(Path(train_dir) / 'images'),
            'val': str(Path(val_dir) / 'images'),
            'test': '',  # Optional
            'names': class_names
        }

        # Write the YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_info, f, sort_keys=False, default_flow_style=False)

        print(f"Generated dataset YAML file: {yaml_path}")
        print("--- YAML Content ---")
        with open(yaml_path, 'r') as f:
            print(f.read())
        print("--------------------")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic images with overlapping hand signatures and seals")
    parser.add_argument("--mask_dir", default="mask", help="Directory containing mask files")
    parser.add_argument("--crop_dir", default="crop", help="Directory containing crop files")
    parser.add_argument("--output_dir", default="synthetic_output", help="Output directory")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--canvas_width", type=int, default=512, help="Canvas width")
    parser.add_argument("--canvas_height", type=int, default=512, help="Canvas height")
    
    # Train/Test split arguments
    parser.add_argument("--split_mode", choices=['train', 'test', 'create_split'], 
                       help="Mode for train/test split: 'train' (use only train data), 'test' (use only test data), 'create_split' (create new split file)")
    parser.add_argument("--split_ratio", type=float, default=0.8, 
                       help="Ratio for train split (default: 0.8 = 80%% train, 20%% test)")
    parser.add_argument("--split_file", default="data_split.json", 
                       help="File to save/load train-test split information")
    
    # Same color feature arguments
    parser.add_argument("--same_color_ratio", type=float, default=0.0,
                       help="Ratio of images where seal and hand signature have the same color (default: 0.0 = disabled, 1.0 = all images)")
    
    args = parser.parse_args()
    
    # Create generator with split parameters
    generator = SyntheticImageGenerator(
        mask_dir=args.mask_dir,
        crop_dir=args.crop_dir,
        output_dir=args.output_dir,
        split_mode=args.split_mode,
        split_ratio=args.split_ratio,
        split_file=args.split_file,
        same_color_ratio=args.same_color_ratio
    )
    
    # If creating split only, exit after split creation
    if args.split_mode == 'create_split':
        print("Data split created successfully!")
        return
    
    # Generate dataset
    generator.generate_dataset(
        num_images=args.num_images,
        canvas_size=(args.canvas_width, args.canvas_height)
        
    )
    
    generator.generate_dataset_yaml(train_dir="train", val_dir="val")

if __name__ == "__main__":
    main()