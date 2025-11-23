"""
CULane Dataset Loader
Author: ThisisVitou
Date: 2025-11-22
This code is from github copilot following this git repo "https://github.com/zkyseu/PPlanedet/tree/v6"
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CULaneDataset(Dataset):
    """
    CULane Dataset for lane detection
    """
    def __init__(self, data_root, split='train', transform=None):
        """
        Args:
            data_root: Path to CULane dataset root
            split: 'train' or 'test'
            transform: Data transformation pipeline
        """
        self.data_root = os.path.normpath(os.path.abspath(data_root))
        self.split = split
        self.transform = transform
        
        # List files
        list_files = {
            'train': 'list/train_gt.txt',
            'test': 'list/test.txt'
        }
        
        list_file = list_files[split].replace('/', os.sep)
        self.list_path = os.path.join(data_root, list_file)
        self.data_list = []
        
        # Load annotations
        print(f'Loading {split} list from {self.list_path}...')
        if not os.path.exists(self.list_path):
            raise FileNotFoundError(f'List file not found: {self.list_path}')
        
        with open(self.list_path, 'r') as f:
            for line in f:
                self.data_list.append(line.strip().split())
        
        print(f'Loaded {len(self.data_list)} samples for {split}')
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        Returns:
            If training:
                img: RGB image tensor [3, H, W]
                mask: Segmentation mask tensor [H, W]
                lane_exist: Binary lane existence tensor [4]
            If testing:
                img: RGB image tensor [3, H, W]
                img_path: Image path string
        """
        line = self.data_list[idx]
        
        # Parse line
        img_path = line[0]
        if img_path[0] == '/':
            img_path = img_path[1:]
        
        # Load image
        img_full_path = os.path.join(self.data_root, img_path)
        img = cv2.imread(img_full_path)
        
        if img is None:
            raise ValueError(f'Failed to load image: {img_full_path}')
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.split == 'train':
            # self.data_list = self.data_list[:10000] 
            # Load segmentation mask
            mask_path = line[1]
            if mask_path[0] == '/':
                mask_path = mask_path[1:]
            mask_full_path = os.path.join(self.data_root, mask_path)
            mask = cv2.imread(mask_full_path, cv2.IMREAD_UNCHANGED)
            
            if mask is None:
                raise ValueError(f'Failed to load mask: {mask_full_path}')
            
            # Handle multi-channel masks
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]
            
            # Load lane existence labels
            lane_exist = np.array([int(x) for x in line[2:]], dtype=np.float32)
            
            # Apply transformations
            if self.transform:
                img, mask, lane_exist = self.transform(img, mask, lane_exist)
            
            return img, mask, lane_exist
        else:
            # Test mode
            if self.transform:
                img = self.transform(img)
            
            return img, img_path


if __name__ == '__main__':
    # Test dataset loading
    print("Testing CULane Dataset...")
    
    from data_process import CULaneTransform
    
    data_root = 'D:/code/detection_lane/culane'  # Change this
    
    # Create transform
    transform = CULaneTransform(
        split='train',
        img_size=(400, 144),
        cut_height=120
    )
    
    # Create dataset
    dataset = CULaneDataset(
        data_root=data_root,
        split='train',
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    img, mask, lane_exist = dataset[0]
    
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Lane exist shape: {lane_exist.shape}")
    print(f"Lane exist values: {lane_exist}")
    
    print("\n✓ Dataset test passed!")