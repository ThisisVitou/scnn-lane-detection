"""
Data preprocessing and augmentation for CULane
Author: ThisisVitou
Date: 2025-11-22
This code is from github copilot following this git repo "https://github.com/zkyseu/PPlanedet/tree/v6"
"""

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CULaneTransform:
    """
    Data preprocessing and augmentation for CULane dataset
    """
    def __init__(self, split='train', img_size=(400, 144), cut_height=120):
        """
        Args:
            split: 'train' or 'test'
            img_size: (width, height) target size
            cut_height: Pixels to crop from top (removes sky)
        """
        self.split = split
        self.img_width, self.img_height = img_size
        self.cut_height = cut_height
        
        # ImageNet normalization (standard for pretrained models)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        if split == 'train':
            # Training augmentations
            self.transform = A.Compose([
                A.Rotate(limit=2, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.Resize(self.img_height, self.img_width, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            # Test/validation (no augmentation)
            self.transform = A.Compose([
                A.Resize(self.img_height, self.img_width, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
    
    def __call__(self, img, mask=None, lane_exist=None):
        """
        Apply transformations
        
        Args:
            img: RGB image (H, W, 3)
            mask: Segmentation mask (H, W) - optional
            lane_exist: Lane existence array (4,) - optional
        
        Returns:
            If training: img_tensor, mask_tensor, lane_exist_tensor
            If testing: img_tensor
        """
        # Crop top portion (remove sky/irrelevant areas)
        img = img[self.cut_height:, :, :]
        if mask is not None:
            mask = mask[self.cut_height:, :]
        
        if self.split == 'train':
            # Apply augmentation to both image and mask
            transformed = self.transform(image=img, mask=mask)
            img_tensor = transformed['image']
            mask_tensor = transformed['mask']
            
            # Convert mask to long tensor
            mask_tensor = mask_tensor.long()
            
            # Convert lane_exist to tensor
            lane_exist_tensor = torch.tensor(lane_exist, dtype=torch.float32)
            
            return img_tensor, mask_tensor, lane_exist_tensor
        else:
            # Test mode - only transform image
            transformed = self.transform(image=img)
            img_tensor = transformed['image']
            return img_tensor


class ManualCULaneTransform:
    """
    Manual data preprocessing without Albumentations
    Use this if you don't want to install Albumentations
    """
    def __init__(self, split='train', img_size=(400, 144), cut_height=120):
        self.split = split
        self.img_width, self.img_height = img_size
        self.cut_height = cut_height
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def rotate_image(self, img, mask, angle):
        """Rotate image and mask"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=0)
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (w, h), borderValue=255)
        return img, mask
    
    def horizontal_flip(self, img, mask):
        """Flip image and mask horizontally"""
        img = cv2.flip(img, 1)
        if mask is not None:
            mask = cv2.flip(mask, 1)
        return img, mask
    
    def resize(self, img, mask=None):
        """Resize image and mask"""
        img = cv2.resize(img, (self.img_width, self.img_height), 
                        interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (self.img_width, self.img_height),
                            interpolation=cv2.INTER_NEAREST)
        return img, mask
    
    def normalize(self, img):
        """Normalize image"""
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img
    
    def to_tensor(self, img, mask=None):
        """Convert to PyTorch tensor"""
        # HWC to CHW
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        if mask is not None:
            mask = torch.from_numpy(mask).long()
        return img, mask
    
    def __call__(self, img, mask=None, lane_exist=None):
        # Crop top
        img = img[self.cut_height:, :, :]
        if mask is not None:
            mask = mask[self.cut_height:, :]
        
        if self.split == 'train':
            # Random rotation
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-2, 2)
                img, mask = self.rotate_image(img, mask, angle)
            
            # Random horizontal flip
            if np.random.rand() < 0.5:
                img, mask = self.horizontal_flip(img, mask)
        
        # Resize
        img, mask = self.resize(img, mask)
        
        # Normalize
        img = self.normalize(img)
        
        # To tensor
        img, mask = self.to_tensor(img, mask)
        
        if self.split == 'train':
            lane_exist = torch.tensor(lane_exist, dtype=torch.float32)
            return img, mask, lane_exist
        else:
            return img


if __name__ == '__main__':
    # Test transformation
    print("Testing CULaneTransform...")
    
    # Create dummy data
    img = np.random.randint(0, 255, (590, 1640, 3), dtype=np.uint8)
    mask = np.random.randint(0, 5, (590, 1640), dtype=np.uint8)
    lane_exist = np.array([1, 1, 0, 0], dtype=np.float32)
    
    # Test training transform
    transform = CULaneTransform(split='train', img_size=(400, 144), cut_height=120)
    img_t, mask_t, exist_t = transform(img, mask, lane_exist)
    
    print(f"Transformed image shape: {img_t.shape}")  # [3, 144, 400]
    print(f"Transformed mask shape: {mask_t.shape}")  # [144, 400]
    print(f"Transformed exist shape: {exist_t.shape}")  # [4]
    
    print(f"Image dtype: {img_t.dtype}")
    print(f"Mask dtype: {mask_t.dtype}")
    print(f"Exist dtype: {exist_t.dtype}")
    
    print(f"Image range: [{img_t.min():.2f}, {img_t.max():.2f}]")
    
    print("\n✓ Transform test passed!")