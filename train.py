"""
SCNN Training Script for CULane Dataset
Optimized for RTX 3050Ti, 16GB RAM
Author: ThisisVitou
Date: 2025-11-22
This code is from github copilot following this git repo "https://github.com/zkyseu/PPlanedet/tree/v6"
"""

import os
import sys
import time
import argparse
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Assuming you have your dataset and model files
from dataset import CULaneDataset
from data_process import CULaneTransform
from model import SCNN  # Your SCNN model
from utils import AverageMeter, save_checkpoint, load_checkpoint


class CombinedLoss(nn.Module):
    """
    Combined loss for segmentation and lane existence
    """
    def __init__(self, seg_weight=1.0, exist_weight=0.1, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.seg_weight = seg_weight
        self.exist_weight = exist_weight
        
        # Class weights for imbalanced segmentation (background vs lanes)
        if class_weights is None:
            # CULane: [background, lane1, lane2, lane3, lane4]
            class_weights = torch.tensor([0.4, 1.0, 1.0, 1.0, 1.0])
        
        self.seg_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        self.exist_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, seg_pred, seg_target, exist_pred, exist_target):
        """
        Args:
            seg_pred: (B, 5, H, W) - segmentation predictions
            seg_target: (B, H, W) - segmentation ground truth
            exist_pred: (B, 4) - existence predictions (logits)
            exist_target: (B, 4) - existence ground truth
        """
        loss_seg = self.seg_loss(seg_pred, seg_target)
        loss_exist = self.exist_loss(exist_pred, exist_target)
        
        total_loss = self.seg_weight * loss_seg + self.exist_weight * loss_exist
        
        return total_loss, loss_seg, loss_exist


def get_optimizer(model, args):
    """
    Create optimizer with learning rate scheduling
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    return optimizer


def get_scheduler(optimizer, args):
    """
    Polynomial learning rate decay
    """
    total_iters = args.epochs * args.iters_per_epoch
    
    lambda_poly = lambda iteration: (1 - iteration / total_iters) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)
    
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, args, writer):
    """
    Train for one epoch
    """
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    seg_losses = AverageMeter()
    exist_losses = AverageMeter()
    
    end = time.time()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
    
    for batch_idx, (imgs, masks, lane_exist) in enumerate(pbar):
        data_time.update(time.time() - end)
        
        # Move to GPU
        imgs = imgs.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        lane_exist = lane_exist.cuda(non_blocking=True)
        
        # Forward pass
        seg_pred, exist_pred = model(imgs)
        
        # Compute loss
        loss, loss_seg, loss_exist = criterion(seg_pred, masks, exist_pred, lane_exist)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        losses.update(loss.item(), imgs.size(0))
        seg_losses.update(loss_seg.item(), imgs.size(0))
        exist_losses.update(loss_exist.item(), imgs.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Seg': f'{seg_losses.avg:.4f}',
            'Exist': f'{exist_losses.avg:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # TensorBoard logging
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('Train/Loss', losses.avg, global_step)
            writer.add_scalar('Train/Seg_Loss', seg_losses.avg, global_step)
            writer.add_scalar('Train/Exist_Loss', exist_losses.avg, global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
    
    return losses.avg


def validate(model, val_loader, criterion, epoch, args, writer):
    """
    Validation for one epoch
    """
    model.eval()
    
    losses = AverageMeter()
    seg_losses = AverageMeter()
    exist_losses = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for batch_idx, (imgs, masks, lane_exist) in enumerate(pbar):
            imgs = imgs.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            lane_exist = lane_exist.cuda(non_blocking=True)
            
            # Forward pass
            seg_pred, exist_pred = model(imgs)
            
            # Compute loss
            loss, loss_seg, loss_exist = criterion(seg_pred, masks, exist_pred, lane_exist)
            
            # Update metrics
            losses.update(loss.item(), imgs.size(0))
            seg_losses.update(loss_seg.item(), imgs.size(0))
            exist_losses.update(loss_exist.item(), imgs.size(0))
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Seg': f'{seg_losses.avg:.4f}',
                'Exist': f'{exist_losses.avg:.4f}'
            })
    
    # TensorBoard logging
    writer.add_scalar('Val/Loss', losses.avg, epoch)
    writer.add_scalar('Val/Seg_Loss', seg_losses.avg, epoch)
    writer.add_scalar('Val/Exist_Loss', exist_losses.avg, epoch)
    
    return losses.avg


def main():
    parser = argparse.ArgumentParser(description='SCNN Training on CULane')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to CULane dataset')
    parser.add_argument('--img_width', type=int, default=400)
    parser.add_argument('--img_height', type=int, default=144)
    parser.add_argument('--cut_height', type=int, default=120,
                       help='Pixels to crop from top (scaled from 240)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes (4 lanes + background)')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=50,
                       help='Log interval')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume')
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU id to use')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    
    print('='*60)
    print('SCNN Training Configuration')
    print('='*60)
    print(f'Data root: {args.data_root}')
    print(f'Image size: ({args.img_width}, {args.img_height})')
    print(f'Batch size: {args.batch_size}')
    print(f'Epochs: {args.epochs}')
    print(f'Learning rate: {args.lr}')
    print(f'GPU: {args.gpu}')
    print('='*60)
    
    # Create datasets
    print('\nLoading datasets...')
    train_transform = CULaneTransform(
        split='train',
        img_size=(args.img_width, args.img_height),
        cut_height=args.cut_height
    )
    train_dataset = CULaneDataset(
        data_root=args.data_root,
        split='train',
        transform=train_transform
    )
    
    # Create validation split (use a portion of test set or create val split)
    val_transform = CULaneTransform(
        split='test',
        img_size=(args.img_width, args.img_height),
        cut_height=args.cut_height
    )
    # For quick validation, use a subset
    val_dataset = CULaneDataset(
        data_root=args.data_root,
        split='test',
        transform=val_transform
    )
    # Sample 1000 images for validation
    val_dataset.data_list = val_dataset.data_list[:1000]
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    args.iters_per_epoch = len(train_loader)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Iterations per epoch: {args.iters_per_epoch}')
    
    # Create model
    print('\nCreating model...')
    model = SCNN(num_classes=args.num_classes)
    model = model.cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Load pretrained weights if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.pretrained:
        print(f'\nLoading pretrained weights from {args.pretrained}')
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Create criterion
    class_weights = torch.tensor([0.4, 1.0, 1.0, 1.0, 1.0]).cuda()
    criterion = CombinedLoss(
        seg_weight=1.0,
        exist_weight=0.1,
        class_weights=class_weights
    )
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    # Resume training if specified
    if args.resume:
        print(f'\nResuming from checkpoint: {args.resume}')
        start_epoch, best_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        print(f'Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}')
    
    # TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(args.log_dir, f'run_{timestamp}'))
    
    # Save configuration
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print('\n' + '='*60)
    print('Starting training...')
    print('='*60 + '\n')
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            epoch, args, writer
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, epoch, args, writer)
        
        epoch_time = time.time() - epoch_start
        
        print(f'\nEpoch {epoch}/{args.epochs} completed in {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Save checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_loss': best_loss,
            'args': args
        }, is_best, args.save_dir, f'epoch_{epoch}.pth')
        
        print(f'Checkpoint saved. Best loss: {best_loss:.4f}\n')
    
    writer.close()
    print('\n' + '='*60)
    print('Training completed!')
    print(f'Best validation loss: {best_loss:.4f}')
    print('='*60)


if __name__ == '__main__':
    main()