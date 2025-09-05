import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.transforms import v2
from PIL import Image
import numpy as np
from networks.model import deeplabv3plus_mobilenet
from torchvision.tv_tensors import Mask as Mask_tv
import segmentation_models_pytorch as smp
import yaml

import argparse
import math
from pathlib import Path
import shutil
import re
from pathlib import Path
import glob
from typing import List

from utils.ramp import polynomialRamp
from utils.loss import FocalLossV2, FocalLossBCE, CombineLoss

IMAGE_EXTENSION = (".png", ".jpg", ".jpeg")

class HandSegmentationDataset(Dataset):
    """Hand Segmentation Dataset for loading images and masks."""
    
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Find all image files (not _gt.jpg and not _mask.jpg)
        # image_pattern = os.path.join(data_dir, "*.jpg")
        # all_files = glob.glob(image_pattern)
        
        self.images_folder = os.path.join(data_dir, "images")
        image_pattern = os.path.join(self.images_folder, "*")
        self.image_files = [f for f in glob.glob(image_pattern) if f.endswith(IMAGE_EXTENSION)]
        mask_folder_pattern = os.path.join(data_dir, "masks*")
        self.mask_folders = [os.path.basename(f) for f in glob.glob(mask_folder_pattern) if os.path.isdir(f)]
            
            
        
        # # Filter to get only base images (not ground truth or mask files)
        # self.image_files = []
        # for file in all_files:
        #     filename = os.path.basename(file)
        #     if not filename.endswith('_gt.jpg') and not filename.endswith('_mask.jpg'):
        #         base_name = filename.replace('.jpg', '')
        #         mask_file = os.path.join(data_dir, f"{base_name}_mask.jpg")
        #         if os.path.exists(mask_file):
        #             self.image_files.append(file)
        
        # print(f"Found {len(self.image_files)} image-mask pairs in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image_name = os.path.basename(image_path)
        masks = []
        for mask_folder in self.mask_folders:
            mask_path = os.path.join(self.data_dir, mask_folder, image_name)
            masks.append(Image.open(mask_path).convert('L'))
        
        masks = np.array(masks)
        # Apply transforms
        if self.target_transform:
            # transform the mask
            masks = self.target_transform(masks)
            
        # image, mask = Image_tv(image), Mask_tv(mask)
        masks = Mask_tv(masks)
        
        if self.transform:
            image, masks = self.transform(image, masks)
        else:
            # Convert mask to tensor and normalize to 0-1 range
            masks = torch.from_numpy(np.array(masks)).float() / 255.0
            # Convert to binary mask (threshold at 0.5)
            masks = (masks > 0.5).float()
            masks = masks.unsqueeze(0)  # Add channel dimension
        
        return image, masks
    
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
        return DataLoader(self.train, batch_size=self.train_batch_size, num_workers=12, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, num_workers=4)


class DeepLabLightningModule(pl.LightningModule):
    """PyTorch Lightning module for DeepLabV3+ hand segmentation training."""
    
    def __init__(self, 
                 num_classes=2, 
                 learning_rate=0.01, 
                 weight_decay=1e-4,
                pretrained_path=None,
                output_stride=16,

                arch=None,
                 encoder_name=None,
                 classes=2,
                 mask_folders = ['masks_hand_signature', 'masks_seal'],
                 logdir = "logs/hand_segmentation",
                 cfg=None,
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logdir = logdir
        
        # Initialize model
        
        # ahhh
        # preprocessing = A.Compose.from_pretrained(checkpoint)
        # self.model = deeplabv3plus_mobilenet(
        #     num_classes=num_classes, 
        #     output_stride=output_stride, 
        #     pretrained_backbone=True
        # )
        
        # self.model = networks.model.__dict__["deeplabv3plus_mobilenet"](num_classes=19, output_stride=output_stride)
        # self.model.load_state_dict(torch.load("/home/treerspeaking/src/python/hand_seg/pretrained_weight/best_deeplabv3plus_mobilenet_cityscapes_os16.pth", weights_only=False )['model_state'])
    

        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=3,
            classes=classes
        )
         
        # smp.losses.
        self.criterion = CombineLoss([
            [1, FocalLossBCE(alpha=1.0, gamma=2.0, reduction='mean')],
            # [1, smp.losses.(alpha=1.0, gamma=2.0, reduction='mean')],
            [1, smp.losses.DiceLoss(mode=smp.losses.MULTILABEL_MODE)],
            [1, smp.losses.JaccardLoss(mode=smp.losses.MULTILABEL_MODE)],
            ]
            )
    
    def forward(self, x):
        return self.model(x)
    
    def on_train_start(self):
        save_train(self.logger.log_dir)
    
    def _log_each_class_metric(self, stage:str, metrics: List[float], metric_name: str, on_step: bool, on_epoch: bool):
        for i, class_name in enumerate(self.trainer.datamodule.val.mask_folders): 
            self.log(f'{stage}/{metric_name}_{class_name}', metrics[:, i].mean(), on_step, on_epoch)
    
    def _calculate_and_log_metrics(self, stage, pred, target, on_step, on_epoch):
        """Calculate IoU, accuracy, F1, precision, recall, and per-class accuracy metrics."""
        tp, fp, fn, tn = smp.metrics.get_stats(pred, target.to(torch.uint8), mode='multilabel', threshold=0.5)
        
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")
        precision_score = smp.metrics.precision(tp, fp, fn, tn, reduction="none")
        recall_score = smp.metrics.recall(tp, fp, fn, tn, reduction="none")
        
        iou_score_macro = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        f1_score_macro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
        recall_macro = smp.metrics.recall(tp, fp, fn, tn, reduction="macro")
        precision_macro = smp.metrics.precision(tp, fp, fn, tn, reduction="macro")
        
        self._log_each_class_metric(stage, iou_score, "iou", on_step, on_epoch)
        self._log_each_class_metric(stage, f1_score, "f1", on_step, on_epoch)
        self._log_each_class_metric(stage, precision_score, "precision", on_step, on_epoch)
        self._log_each_class_metric(stage, recall_score, "recall", on_step, on_epoch)
        self.log(f'{stage}/accuracy', accuracy, on_step=True, on_epoch=True)
        self.log(f'{stage}/iou', iou_score_macro, on_step=True, on_epoch=True)
        self.log(f'{stage}/f1', f1_score_macro, on_step=True, on_epoch=True)
        self.log(f'{stage}/precision', precision_macro, on_step=True, on_epoch=True)
        self.log(f'{stage}/recall', recall_macro, on_step=True, on_epoch=True)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        
        # Forward pass
        outputs = self(images)
        
        # Resize output to match mask size if needed
        if outputs.shape[2:] != masks.shape[2:]:
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
        
        # Convert masks to proper format for BCE
        targets = masks
        outputs_sigmoid = torch.sigmoid(outputs)
        
        # Calculate loss
        loss = self.criterion(outputs, targets)
        
        # Log metrics
        self._calculate_and_log_metrics("train", outputs_sigmoid, targets, True, True)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        # Forward pass
        outputs = self(images)
        
        # Resize output to match mask size if needed
        if outputs.shape[2:] != masks.shape[2:]:
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
        
        # Convert masks to proper format for BCE
        targets = masks
        outputs_sigmoid = torch.sigmoid(outputs)
        
        # Calculate loss
        loss = self.criterion(outputs, targets)
        
        # Log metrics
        self._calculate_and_log_metrics("val", outputs_sigmoid, targets, False, True)
        
        self.log('val/loss', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        # Use AdamW optimizer for better convergence
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=0.001667, 
            betas=[0.9, 0.999], 
            # weight_decay=0.0005
            weight_decay=0.001
        )
        
        # optimizer = torch.optim.SGD(
        #     self.parameters(), 
        #     lr=self.learning_rate, 
        #     # betas=[0.9, 0.999], 
        #     momentum=0.9,
        #     # weight_decay=0.0005
        #     weight_decay=0.001
        # )
        
        warm_up_iter = self.trainer.datamodule.num_iter * self.trainer.max_epochs / 9
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: polynomialRamp(steps, warm_up_iter, 4.0))
        
        # warm_up_iter = 2000
        # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, 
        #     start_factor=0.001,  # Start at 1% of base LR
        #     total_iters=warm_up_iter    # Warm up for 1000 steps
        # )

        main_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.trainer.datamodule.num_iter * self.trainer.max_epochs , power=4.0)
        
        # Use SequentialLR to combine warm-up and main scheduler
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[warm_up_iter]
        )
        
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            "name": None,
        }
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def forward(self, x):
        return self.model(x)


def create_data_transforms(input_size=512):
    """Create data transforms for training and validation."""
    
    # Training transforms with augmentation
    train_transform = v2.Compose([
        # v2.Resize((input_size, input_size)),
        v2.ToTensor(),
        # v2.RandomResizedCrop((input_size, input_size), scale=(0.4, 1.0)),
        v2.RandomResizedCrop((input_size, input_size), scale=(0.4, 1.0)),
        # v2.Resize((input_size, input_size)),
        # v2.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.2),
        # v2.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.5),
        # v2.RandomPhotometricDistort(),
        # v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(360),
        v2.RandomGrayscale(p=0.1),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = v2.Compose([
        v2.Resize((input_size, input_size)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Target transform for masks
    target_transform = v2.Compose([
        # v2.Resize((input_size, input_size), interpolation=v2.InterpolationMode.NEAREST),
        v2.Lambda(lambda x: torch.from_numpy(np.array(x))),
        v2.Lambda(lambda x: (x > 0.5).float()),
        # v2.Lambda(lambda x: (x > 0.5).to(torch.int32)),
        # v2.Lambda(lambda x: x.unsqueeze(0))
    ])
    
    return train_transform, val_transform, target_transform


def get_next_version_folder(base_path):
    pattern = os.path.join(base_path, "version_*")
    version_folders = glob.glob(pattern)
    
    # Extract version numbers and find the maximum
    max_version = -1
    for folder_path in version_folders:
        folder_name = os.path.basename(folder_path)
        match = re.match(r'version_(\d+)', folder_name)
        if match:
            version = int(match.group(1))
            max_version = max(max_version, version)
    
    # Return the next version folder path
    next_version = max_version + 1
    return os.path.join(base_path, f"version_{next_version}")

def save_train(save_dir):
    save_dir = Path(save_dir)
    
    # newest_folder = get_latest_version_folder(save_dir)
    
    current_script = Path(__file__)
    
    # Create destination path
    # dest_path = Path(newest_folder) / current_script.name
    dest_path = save_dir / current_script.name
    
    try:
        # Copy the current script to the latest version folder
        shutil.copy2(current_script, dest_path)
        print(f"Copied {current_script.name} to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Error copying file: {e}")
        return None
    
    
    
    
    # return latest_folder

# import builtins

# original_print = builtins.print

# def debug_print(*args, **kwargs):
#     message = ' '.join(str(arg) for arg in args)
#     if "Logs will be saved to" in message:
#         # raise ValueError(f"FOUND MESSAGE: {message}")  # VS Code will break here
#         breakpoint()
#         print("hahaha")
#     return original_print(*args, **kwargs)

# builtins.print = debug_print

def main():
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ for Hand Segmentation')
    parser.add_argument('--config', type=str, 
                       help='config file for training')
    
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    # Create directories
    os.makedirs(cfg["log_dir"], exist_ok=True)
    
    # Create transforms
    train_transform, val_transform, target_transform = create_data_transforms(cfg["input_size"])

    
    datamodule = HandSegDataModule(os.path.join(cfg["data_root"]), cfg["batch_size"], cfg["batch_size"], train_transform, val_transform, target_transform)
    
    
    
    
    # Initialize model
    model = DeepLabLightningModule(
        num_classes=2,  # background and hand
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        arch=cfg["arch"],
        encoder_name=cfg["encoder_name"],
        # learning_rate=args.learning_rate,
        # weight_decay=args.weight_decay,
        pretrained_path=cfg.get("pretrained_path", None),
        mask_folders=datamodule.train.mask_folders,
        cfg=cfg
        # output_stride=8

    )
    
    # Setup callbacks
    # ModelCheckpoint for saving best model based on validation IoU
    checkpoint_callback = ModelCheckpoint(
        # dirpath=args.output_dir,
        filename='best_hand_segmentation_{epoch:02d}_{val/iou_epoch:.4f}_{val/f1_epoch:.4f}',
        monitor='val/iou',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # TensorBoard logger
    version = get_next_version_folder("/home/treerspeaking/src/python/hand_seg/logs/hand_segmentation")
    logger = TensorBoardLogger(
        save_dir=cfg["log_dir"],
        name='hand_segmentation',
        version=version + "_" +cfg["name"]
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg["num_epochs"],
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        accelerator='auto',
        devices='auto',
        precision=16,  # Use mixed precision training
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        benchmark=True
    )
    
    
    # Print model summary
    print("\nModel Summary:")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Start training
    print(f"\nStarting training for {cfg['num_epochs']} epochs...")
    # print(f"Training on {len(train_loader)} batches per epoch")
    # print(f"Validation on {len(val_loader)} batches per epoch")
    print(f"Logs will be saved to: {logger.log_dir}")
    
    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model, datamodule=datamodule)
    print("\nTraining completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation IoU: {checkpoint_callback.best_model_score:.4f}")


if __name__ == '__main__':
    main()
