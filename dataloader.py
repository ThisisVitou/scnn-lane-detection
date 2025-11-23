"""
DataLoader creation for training
Author: ThisisVitou
Date: 2025-11-22
This code is from github copilot following this git repo "https://github.com/zkyseu/PPlanedet/tree/v6"
"""

import torch
from torch.utils.data import DataLoader
from dataset import CULaneDataset
from data_process import CULaneTransform


def create_dataloaders(data_root, batch_size=2, num_workers=2, 
                       img_size=(400, 144), cut_height=120):
    """
    Create training and validation dataloaders
    
    Args:
        data_root: Path to CULane dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: (width, height) tuple
        cut_height: Pixels to crop from top
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader (subset of train for quick validation)
    """
    # Training dataset
    print("Creating training dataset...")
    train_transform = CULaneTransform(
        split='train',
        img_size=img_size,
        cut_height=cut_height
    )
    train_dataset = CULaneDataset(
        data_root=data_root,
        split='train',
        transform=train_transform
    )
    
    # Create validation split (use first 1000 samples for quick validation)
    print("Creating validation dataset...")
    val_transform = CULaneTransform(
        split='train',  # Use train split but with test augmentation
        img_size=img_size,
        cut_height=cut_height
    )
    val_dataset = CULaneDataset(
        data_root=data_root,
        split='train',
        transform=val_transform
    )
    
    # Use subset for validation
    val_size = min(1000, len(val_dataset))
    val_dataset.data_list = val_dataset.data_list[:val_size]
    
    # Remove validation samples from training
    train_dataset.data_list = train_dataset.data_list[val_size:]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataloader
    print("Testing DataLoader...")
    
    data_root = 'D:\\code\\detection_lane\\culane'  # Change this
    
    train_loader, val_loader = create_dataloaders(
        data_root=data_root,
        batch_size=2,
        num_workers=2,
        img_size=(400, 144),
        cut_height=120
    )
    
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test loading one batch
    print("\nLoading first training batch...")
    for imgs, masks, lane_exist in train_loader:
        print(f"Images shape: {imgs.shape}")  # [2, 3, 144, 400]
        print(f"Masks shape: {masks.shape}")  # [2, 144, 400]
        print(f"Lane exist shape: {lane_exist.shape}")  # [2, 4]
        print(f"Images dtype: {imgs.dtype}")
        print(f"Masks dtype: {masks.dtype}")
        print(f"Lane exist dtype: {lane_exist.dtype}")
        break
    
    print("\n✓ DataLoader test passed!")