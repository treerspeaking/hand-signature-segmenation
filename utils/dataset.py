from torchvision.tv_tensors import Mask as Mask_tv
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import yaml

import os
import glob
import math

IMAGE_EXTENSION = (".png", ".jpg", ".jpeg")
# class HandSegmentationDataset(Dataset):
#     """Hand Segmentation Dataset for loading images and masks."""
    
#     def __init__(self, yaml_file, split="train",transform=None, target_transform=None):
#         yaml_data = yaml.load(open(yaml_file, "r"), Loader=yaml.Loader)
        
#         self.data_dir = yaml_data["path"]
#         self.transform = transform
#         self.target_transform = target_transform
#         # self.split = yaml_data[split]
#         self.split = split
#         self.names = yaml_data["names"]
        
#         # Find all image files (not _gt.jpg and not _mask.jpg)
#         # image_pattern = os.path.join(data_dir, "*.jpg")
#         # all_files = glob.glob(image_pattern)
#         self.data_dir = os.path.join(self.data_dir, self.split)
#         self.images_folder = os.path.join(self.data_dir, "images")
#         image_pattern = os.path.join(self.images_folder, "*")
#         self.image_files = [f for f in glob.glob(image_pattern) if f.endswith(IMAGE_EXTENSION)]
        
#         self.mask_folders = []
        
#         for key, value in self.names.items():
#             self.mask_folders.append("masks_" + value)
        
#         # mask_folder_pattern = os.path.join(self.data_dir, self.split, "masks*")
#         # self.mask_folders = [os.path.basename(f) for f in glob.glob(mask_folder_pattern) if os.path.isdir(f)]
            
            
        
#         # # Filter to get only base images (not ground truth or mask files)
#         # self.image_files = []
#         # for file in all_files:
#         #     filename = os.path.basename(file)
#         #     if not filename.endswith('_gt.jpg') and not filename.endswith('_mask.jpg'):
#         #         base_name = filename.replace('.jpg', '')
#         #         mask_file = os.path.join(data_dir, f"{base_name}_mask.jpg")
#         #         if os.path.exists(mask_file):
#         #             self.image_files.append(file)
        
#         # print(f"Found {len(self.image_files)} image-mask pairs in {data_dir}")
    
#     def __len__(self):
#         return len(self.image_files)
    
#     def __getitem__(self, idx):
#         # Load image
#         image_path = self.image_files[idx]
#         image = Image.open(image_path).convert('RGB')
#         image_name = os.path.basename(image_path)
#         masks = []
#         for mask_folder in self.mask_folders:
#             mask_path = os.path.join(self.data_dir, mask_folder, image_name)
#             masks.append(Image.open(mask_path).convert('L'))
        
#         masks = np.array(masks)
#         # Apply transforms
#         if self.target_transform:
#             # transform the mask
#             masks = self.target_transform(masks)
            
#         # image, mask = Image_tv(image), Mask_tv(mask)
#         masks = Mask_tv(masks)
        
#         if self.transform:
#             image, masks = self.transform(image, masks)
#         else:
#             # Convert mask to tensor and normalize to 0-1 range
#             masks = torch.from_numpy(np.array(masks)).float() / 255.0
#             # Convert to binary mask (threshold at 0.5)
#             masks = (masks > 0.5).float()
#             masks = masks.unsqueeze(0)  # Add channel dimension
        
#         return image, masks
    
# class HandSegDataModule(pl.LightningDataModule):
#     def __init__(self, yaml_file, train_batch_size, val_batch_size, train_transforms, val_transforms,mask_transforms):
#         super().__init__()
#         # self.datadir = datadir
#         self.yaml_file = yaml_file
#         self.train_batch_size = train_batch_size
#         self.val_batch_size = val_batch_size
#         self.train_transforms = train_transforms
#         self.val_transforms = val_transforms
#         self.mask_transforms = mask_transforms
#         self.train = HandSegmentationDataset(yaml_file, split="train", transform=self.train_transforms, target_transform=self.mask_transforms)
#         self.val = HandSegmentationDataset(yaml_file, split="test", transform=self.val_transforms, target_transform=self.mask_transforms)
#         self.num_iter = math.ceil(len(self.train) / train_batch_size)
        
    
#     def setup(self, stage: str):
#         # self.mnist_test = MNIST(self.data_dir, train=False)
#         # self.mnist_predict = MNIST(self.data_dir, train=False)
#         # mnist_full = MNIST(self.data_dir, train=True)
#         # self.mnist_train, self.mnist_val = random_split(
#         #     mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
#         # )
#         pass
        
        

#     def train_dataloader(self):
#         return DataLoader(self.train, batch_size=self.train_batch_size, num_workers=12, shuffle=True, pin_memory=True)

#     def val_dataloader(self):
#         return DataLoader(self.val, batch_size=self.val_batch_size, num_workers=4)

#     def test_dataloader(self):
#         return DataLoader(self.val, batch_size=self.val_batch_size, num_workers=4)

#     def predict_dataloader(self):
#         return DataLoader(self.val, batch_size=self.val_batch_size, num_workers=4)

class HandSegmentationDataset(Dataset):
    """Hand Segmentation Dataset for loading images and masks."""
    
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Find all image files (not _gt.jpg and not _mask.jpg)
        image_pattern = os.path.join(data_dir, "*.jpg")
        all_files = glob.glob(image_pattern)
        
        # Filter to get only base images (not ground truth or mask files)
        self.image_files = []
        for file in all_files:
            filename = os.path.basename(file)
            if not filename.endswith('_gt.jpg') and not filename.endswith('_mask.jpg'):
                base_name = filename.replace('.jpg', '')
                mask_file = os.path.join(data_dir, f"{base_name}_mask.jpg")
                if os.path.exists(mask_file):
                    self.image_files.append(file)
        
        print(f"Found {len(self.image_files)} image-mask pairs in {data_dir}")
        
        self.mask_folders = ["hand_seg"]
        
#         for key, value in self.names.items():
#             self.mask_folders.append("masks_" + value)
        
#         # mask_folder_pattern = os.path.join(self.data_dir, self.split, "masks*")
#         # self.mask_folders = [os.path.basename(f) for f in glob.glob(mask_folder_pattern) if os.path.isdir(f)]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load corresponding mask
        base_name = os.path.basename(image_path).replace('.jpg', '')
        mask_path = os.path.join(self.data_dir, f"{base_name}_mask.jpg")
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        
        # Apply transforms
        if self.target_transform:
            # transform the mask
            mask = self.target_transform(mask)
            
        # image, mask = Image_tv(image), Mask_tv(mask)
        mask = Mask_tv(mask)
        
        if self.transform:
            # print(f"Image dtype: {image.dtype}, shape: {image.shape}")
            # print(f"Image min/max: {image.min()}, {image.max()}")
            image, mask = self.transform(image, mask)
        else:
            # Convert mask to tensor and normalize to 0-1 range
            mask = torch.from_numpy(np.array(mask)).float() / 255.0
            # Convert to binary mask (threshold at 0.5)
            mask = (mask > 0.5).float()
            mask = mask.unsqueeze(0)  # Add channel dimension
        
        return image, mask
    
class HandSegDataModule(pl.LightningDataModule):
    def __init__(self, datadir, train_batch_size, val_batch_size, train_transforms, val_transforms,mask_transforms):
        super().__init__()
        self.datadir = datadir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.mask_transforms = mask_transforms
        self.train = HandSegmentationDataset(os.path.join(datadir, 'train'), transform=self.train_transforms, target_transform=self.mask_transforms)
        self.val = HandSegmentationDataset(os.path.join(datadir, 'test'), transform=self.val_transforms, target_transform=self.mask_transforms)
        self.num_iter = math.ceil(len(self.train) / train_batch_size)
        

        
    
    def setup(self, stage: str):
        # self.mnist_test = MNIST(self.data_dir, train=False)
        # self.mnist_predict = MNIST(self.data_dir, train=False)
        # mnist_full = MNIST(self.data_dir, train=True)
        # self.mnist_train, self.mnist_val = random_split(
        #     mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        # )
        pass
        
        

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, num_workers=8, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, num_workers=4)