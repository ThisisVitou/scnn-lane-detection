"""
Utility functions for training
Author: ThisisVitou
Date: 2025-11-22
This code is from github copilot following this git repo "https://github.com/zkyseu/PPlanedet/tree/v6"
"""

import os
import shutil
import torch


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """Save checkpoint and optionally the best model"""
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    # Save as latest
    latest_path = os.path.join(save_dir, 'latest.pth')
    shutil.copyfile(filepath, latest_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        shutil.copyfile(filepath, best_path)
        print(f'✓ New best model saved!')


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    return epoch, best_loss