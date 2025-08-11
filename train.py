import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.transforms import v2
from PIL import Image
import numpy as np
from networks.model import deeplabv3plus_mobilenet
import glob
from torch.optim.lr_scheduler import PolynomialLR
from torchvision.tv_tensors import Image as Image_tv, Mask as Mask_tv
import argparse
import math
from utils.ramp import polynomialRamp

class FocalLossV2(torch.nn.Module):
    """
    Alternative stable implementation using cross entropy loss directly.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        # Input validation
        assert not torch.isnan(inputs).any(), "Input contains NaN"
        # assert not torch.isinf(inputs).any(), "Input contains Inf"
        
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                ignore_index=self.ignore_index)
        
        # Compute probabilities using softmax
        with torch.no_grad():
            p = F.softmax(inputs, dim=-1)
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # Handle ignore_index
            if self.ignore_index >= 0:
                mask = targets != self.ignore_index
                p_t = p_t * mask.float() + (1 - mask.float())  # Set ignored to 1 to avoid (1-1)^gamma = 0
        
        # Clamp probabilities to avoid numerical issues
        p_t = torch.clamp(p_t, min=1e-8, max=1.0 - 1e-8)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute focal loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Handle ignore_index for reduction
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            focal_loss = focal_loss * mask.float()
            
            if mask.sum() == 0:
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Apply reduction
        if self.reduction == 'mean':
            if self.ignore_index >= 0:
                return focal_loss.sum() / mask.sum().clamp(min=1)
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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
        return DataLoader(self.train, batch_size=self.train_batch_size, num_workers=14, shuffle=True, pin_memory=True)

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
                 output_stride=8):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize model
        self.model = deeplabv3plus_mobilenet(
            num_classes=num_classes, 
            output_stride=output_stride, 
            pretrained_backbone=True
        )
        
        # Load pretrained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
                
            # Load weights with strict=False to handle size mismatches
            try:
                self.model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded pretrained weights")
            except Exception as e:
                print(f"Warning: Could not load some weights: {e}")
        
        # Loss function - use CrossEntropyLoss for segmentation
        # self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.criterion = FocalLossV2(alpha=1.0, gamma=3.0, reduction='mean',ignore_index=255)
        
        # Metrics for logging
        self.train_iou_sum = 0
        self.train_acc_sum = 0
        self.train_f1_sum = 0
        self.train_precision_sum = 0
        self.train_recall_sum = 0
        self.train_class0_acc_sum = 0
        self.train_class1_acc_sum = 0
        self.train_count = 0
        
        self.val_iou_sum = 0
        self.val_acc_sum = 0
        self.val_f1_sum = 0
        self.val_precision_sum = 0
        self.val_recall_sum = 0
        self.val_class0_acc_sum = 0
        self.val_class1_acc_sum = 0
        self.val_count = 0
    
    def forward(self, x):
        return self.model(x)
    
    def calculate_metrics(self, pred, target):
        """Calculate IoU, accuracy, F1, precision, recall, and per-class accuracy metrics."""
        pred_mask = torch.argmax(pred, dim=1)
        target_mask = target.squeeze(1).long()
        
        # Calculate overall accuracy
        correct = (pred_mask == target_mask).float()
        accuracy = correct.mean()
        
        # Calculate per-class accuracy
        # Class 0 (background) accuracy
        class0_mask = (target_mask == 0)
        if class0_mask.sum() > 0:
            class0_correct = ((pred_mask == 0) & (target_mask == 0)).float().sum()
            class0_accuracy = class0_correct / class0_mask.sum().float()
        else:
            class0_accuracy = torch.tensor(0.0, device=pred.device)
        
        # Class 1 (hand) accuracy
        class1_mask = (target_mask == 1)
        if class1_mask.sum() > 0:
            class1_correct = ((pred_mask == 1) & (target_mask == 1)).float().sum()
            class1_accuracy = class1_correct / class1_mask.sum().float()
        else:
            class1_accuracy = torch.tensor(0.0, device=pred.device)
        
        # Calculate IoU for foreground class (class 1)
        intersection = ((pred_mask == 1) & (target_mask == 1)).float().sum()
        union = ((pred_mask == 1) | (target_mask == 1)).float().sum()
        iou = intersection / (union + 1e-8)
        
        # Calculate precision, recall, and F1 for foreground class (class 1)
        true_positives = intersection  # Already calculated above
        predicted_positives = (pred_mask == 1).float().sum()
        actual_positives = (target_mask == 1).float().sum()
        
        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return accuracy, iou, f1, precision, recall, class0_accuracy, class1_accuracy
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        
        # Forward pass
        outputs = self(images)
        
        # Resize output to match mask size if needed
        if outputs.shape[2:] != masks.shape[2:]:
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
        
        # Convert masks to proper format for CrossEntropyLoss
        targets = masks.squeeze(1).long()
        
        # Calculate loss
        loss = self.criterion(outputs, targets)
        
        # Calculate metrics
        accuracy, iou, f1, precision, recall, class0_acc, class1_acc = self.calculate_metrics(outputs, masks)
        
        # Update running sums
        self.train_iou_sum += iou.item()
        self.train_acc_sum += accuracy.item()
        self.train_f1_sum += f1.item()
        self.train_precision_sum += precision.item()
        self.train_recall_sum += recall.item()
        self.train_class0_acc_sum += class0_acc.item()
        self.train_class1_acc_sum += class1_acc.item()
        self.train_count += 1
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', accuracy, on_step=True, on_epoch=True)
        self.log('train/iou', iou, on_step=True, on_epoch=True)
        self.log('train/f1', f1, on_step=True, on_epoch=True)
        self.log('train/precision', precision, on_step=True, on_epoch=True)
        self.log('train/recall', recall, on_step=True, on_epoch=True)
        self.log('train/class0_accuracy', class0_acc, on_step=True, on_epoch=True)
        self.log('train/class1_accuracy', class1_acc, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        
        # Forward pass
        outputs = self(images)
        
        # Resize output to match mask size if needed
        if outputs.shape[2:] != masks.shape[2:]:
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
        
        # Convert masks to proper format for CrossEntropyLoss
        targets = masks.squeeze(1).long()
        
        # Calculate loss
        loss = self.criterion(outputs, targets)
        
        # Calculate metrics
        accuracy, iou, f1, precision, recall, class0_acc, class1_acc = self.calculate_metrics(outputs, masks)
        
        # Update running sums
        self.val_iou_sum += iou.item()
        self.val_acc_sum += accuracy.item()
        self.val_f1_sum += f1.item()
        self.val_precision_sum += precision.item()
        self.val_recall_sum += recall.item()
        self.val_class0_acc_sum += class0_acc.item()
        self.val_class1_acc_sum += class1_acc.item()
        self.val_count += 1
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', accuracy, on_step=False, on_epoch=True)
        self.log('val/iou', iou, on_step=False, on_epoch=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True)
        self.log('val/precision', precision, on_step=False, on_epoch=True)
        self.log('val/recall', recall, on_step=False, on_epoch=True)
        self.log('val/class0_accuracy', class0_acc, on_step=False, on_epoch=True)
        self.log('val/class1_accuracy', class1_acc, on_step=False, on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        if self.train_count > 0:
            avg_iou = self.train_iou_sum / self.train_count
            avg_acc = self.train_acc_sum / self.train_count
            avg_f1 = self.train_f1_sum / self.train_count
            avg_precision = self.train_precision_sum / self.train_count
            avg_recall = self.train_recall_sum / self.train_count
            avg_class0_acc = self.train_class0_acc_sum / self.train_count
            avg_class1_acc = self.train_class1_acc_sum / self.train_count
            
            self.log('train/epoch_iou', avg_iou)
            self.log('train/epoch_accuracy', avg_acc)
            self.log('train/epoch_f1', avg_f1)
            self.log('train/epoch_precision', avg_precision)
            self.log('train/epoch_recall', avg_recall)
            self.log('train/epoch_class0_accuracy', avg_class0_acc)
            self.log('train/epoch_class1_accuracy', avg_class1_acc)
        
        # Reset counters
        self.train_iou_sum = 0
        self.train_acc_sum = 0
        self.train_f1_sum = 0
        self.train_precision_sum = 0
        self.train_recall_sum = 0
        self.train_class0_acc_sum = 0
        self.train_class1_acc_sum = 0
        self.train_count = 0
    
    def on_validation_epoch_end(self):
        if self.val_count > 0:
            avg_iou = self.val_iou_sum / self.val_count
            avg_acc = self.val_acc_sum / self.val_count
            avg_f1 = self.val_f1_sum / self.val_count
            avg_precision = self.val_precision_sum / self.val_count
            avg_recall = self.val_recall_sum / self.val_count
            avg_class0_acc = self.val_class0_acc_sum / self.val_count
            avg_class1_acc = self.val_class1_acc_sum / self.val_count
            
            self.log('val/epoch_iou', avg_iou)
            self.log('val/epoch_accuracy', avg_acc)
            self.log('val/epoch_f1', avg_f1)
            self.log('val/epoch_precision', avg_precision)
            self.log('val/epoch_recall', avg_recall)
            self.log('val/epoch_class0_accuracy', avg_class0_acc)
            self.log('val/epoch_class1_accuracy', avg_class1_acc)
        
        # Reset counters
        self.val_iou_sum = 0
        self.val_acc_sum = 0
        self.val_f1_sum = 0
        self.val_precision_sum = 0
        self.val_recall_sum = 0
        self.val_class0_acc_sum = 0
        self.val_class1_acc_sum = 0
        self.val_count = 0
    
    def configure_optimizers(self):
        # Use AdamW optimizer for better convergence
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=0.001667, 
            betas=[0.9, 0.999], 
            weight_decay=0.0005
        )
        
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


def create_data_transforms(input_size=512):
    """Create data transforms for training and validation."""
    
    # Training transforms with augmentation
    train_transform = v2.Compose([
        # v2.Resize((input_size, input_size)),
        v2.RandomResizedCrop((input_size, input_size), scale=(0.4, 1.0), ratio=(0.75, 1.33)),
        # v2.Resize((input_size, input_size)),
        v2.ToTensor(),
        v2.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.5),
        # v2.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.5),
        # v2.RandomPhotometricDistort(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(90),
        v2.RandomGrayscale(p=0.4),
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
        v2.Lambda(lambda x: torch.from_numpy(np.array(x)).float() / 255.0),
        v2.Lambda(lambda x: (x > 0.5).float()),
        v2.Lambda(lambda x: x.unsqueeze(0))
    ])
    
    return train_transform, val_transform, target_transform


def main():
    parser = argparse.ArgumentParser(description='Train DeepLabV3+ for Hand Segmentation')
    parser.add_argument('--data_root', type=str, default='./dataset', 
                       help='Root directory of dataset')
    parser.add_argument('--pretrained_path', type=str, 
                       default='./pretrained_weight/best_deeplabv3plus_mobilenet_cityscapes_os16.pth',
                       help='Path to pretrained weights')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay')
    parser.add_argument('--input_size', type=int, default=512, 
                       help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loader workers')
    # parser.add_argument('--output_dir', type=str, default='./checkpoints', 
    #                    help='Output directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', 
                       help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    # Create directories
    # os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create transforms
    train_transform, val_transform, target_transform = create_data_transforms(args.input_size)
    
    # Create datasets
    train_dataset = HandSegmentationDataset(
        data_dir=os.path.join(args.data_root, 'train'),
        transform=train_transform,
        target_transform=target_transform
    )
    
    val_dataset = HandSegmentationDataset(
        data_dir=os.path.join(args.data_root, 'test'),
        transform=val_transform,
        target_transform=target_transform
    )
    
    # Create data loaders
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True
    # )
    datamodule = HandSegDataModule(os.path.join(args.data_root), args.batch_size, args.batch_size,train_transform, val_transform, target_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = DeepLabLightningModule(
        num_classes=2,  # background and hand
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pretrained_path=args.pretrained_path,
        output_stride=8
    )
    
    # Setup callbacks
    # ModelCheckpoint for saving best model based on validation IoU
    checkpoint_callback = ModelCheckpoint(
        # dirpath=args.output_dir,
        filename='best_hand_segmentation_{epoch:02d}_{val/epoch_iou:.4f}',
        monitor='val/epoch_iou',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name='hand_segmentation',
        version=None
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
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
    print(f"\nStarting training for {args.num_epochs} epochs...")
    # print(f"Training on {len(train_loader)} batches per epoch")
    # print(f"Validation on {len(val_loader)} batches per epoch")
    print(f"Logs will be saved to: {logger.log_dir}")
    # print(f"Checkpoints will be saved to: {args.output_dir}")
    
    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model, datamodule=datamodule)
    print("\nTraining completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation IoU: {checkpoint_callback.best_model_score:.4f}")


if __name__ == '__main__':
    main()
