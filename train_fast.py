"""
SCNN Fast Training - FIXED GPU Version
"""
import os
import sys
# ADD THESE LINES RIGHT HERE - BEFORE ANY OTHER CODE
# Force use of RTX GPU (try 1 first, then 0 if it doesn't work)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Change to '0' if RTX is GPU 0

import time
import argparse
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from dataset import CULaneDataset
from data_process import CULaneTransform
from model import SCNN
from utils import AverageMeter, save_checkpoint, load_checkpoint


class CombinedLoss(nn.Module):
    def __init__(self, seg_weight=1.0, exist_weight=0.1, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.seg_weight = seg_weight
        self.exist_weight = exist_weight
        
        if class_weights is None:
            class_weights = torch.tensor([0.4, 1.0, 1.0, 1.0, 1.0])
        
        self.seg_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        self.exist_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, seg_pred, seg_target, exist_pred, exist_target):
        loss_seg = self.seg_loss(seg_pred, seg_target)
        loss_exist = self.exist_loss(exist_pred, exist_target)
        total_loss = self.seg_weight * loss_seg + self.exist_weight * loss_exist
        return total_loss, loss_seg, loss_exist


def get_optimizer(model, args):
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    return optimizer


def get_scheduler(optimizer, args):
    total_iters = args.epochs * args.iters_per_epoch
    lambda_poly = lambda iteration: (1 - iteration / total_iters) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, args, writer):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    seg_losses = AverageMeter()
    exist_losses = AverageMeter()
    
    end = time.time()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, (imgs, masks, lane_exist) in enumerate(pbar):
        data_time.update(time.time() - end)
        
        # Move to GPU with explicit device
        imgs = imgs.to(args.device, non_blocking=True)
        masks = masks.to(args.device, non_blocking=True)
        lane_exist = lane_exist.to(args.device, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast('cuda', enabled=args.use_amp):
            seg_pred, exist_pred = model(imgs)
            loss, loss_seg, loss_exist = criterion(seg_pred, masks, exist_pred, lane_exist)
        
        # Backward pass
        optimizer.zero_grad()
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
            'Time': f'{batch_time.avg:.2f}s',
            'GPU': f'{torch.cuda.memory_allocated(args.device)/1024**3:.1f}GB'  # Show GPU usage
        })
        
        # TensorBoard logging
        if batch_idx % args.log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', losses.avg, global_step)
            writer.add_scalar('Train/Seg_Loss', seg_losses.avg, global_step)
            writer.add_scalar('Train/Exist_Loss', exist_losses.avg, global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Train/BatchTime', batch_time.avg, global_step)
            writer.add_scalar('Train/GPU_Memory_GB', torch.cuda.memory_allocated(args.device)/1024**3, global_step)
    
    return losses.avg


def validate(model, val_loader, criterion, epoch, args, writer):
    model.eval()
    
    losses = AverageMeter()
    seg_losses = AverageMeter()
    exist_losses = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for batch_idx, (imgs, masks, lane_exist) in enumerate(pbar):
            imgs = imgs.to(args.device, non_blocking=True)
            masks = masks.to(args.device, non_blocking=True)
            lane_exist = lane_exist.to(args.device, non_blocking=True)
            
            with autocast('cuda', enabled=args.use_amp):
                seg_pred, exist_pred = model(imgs)
                loss, loss_seg, loss_exist = criterion(seg_pred, masks, exist_pred, lane_exist)
            
            losses.update(loss.item(), imgs.size(0))
            seg_losses.update(loss_seg.item(), imgs.size(0))
            exist_losses.update(loss_exist.item(), imgs.size(0))
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Seg': f'{seg_losses.avg:.4f}',
                'Exist': f'{exist_losses.avg:.4f}'
            })
    
    writer.add_scalar('Val/Loss', losses.avg, epoch)
    writer.add_scalar('Val/Seg_Loss', seg_losses.avg, epoch)
    writer.add_scalar('Val/Exist_Loss', exist_losses.avg, epoch)
    
    return losses.avg

def select_best_gpu():
    """Automatically select NVIDIA GPU, avoiding integrated graphics"""
    if not torch.cuda.is_available():
        return None
    
    print("\n" + "="*60)
    print("GPU Selection")
    print("="*60)
    
    nvidia_gpus = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        compute = torch.cuda.get_device_properties(i)
        
        print(f"GPU {i}: {name}")
        print(f"  Memory: {memory:.2f} GB")
        print(f"  Compute: {compute.major}.{compute.minor}")
        
        # Check if it's NVIDIA discrete GPU
        if any(keyword in name.lower() for keyword in ['nvidia', 'rtx', 'gtx', 'tesla', 'quadro']):
            nvidia_gpus.append((i, memory, name))
    
    if not nvidia_gpus:
        print("\n⚠️  Warning: No NVIDIA GPU detected!")
        return 0
    
    # Select GPU with most memory
    best_gpu = max(nvidia_gpus, key=lambda x: x[1])
    print(f"\n✓ Selected: GPU {best_gpu[0]} - {best_gpu[2]}")
    print("="*60)
    
    return best_gpu[0]

def main():
    parser = argparse.ArgumentParser(description='SCNN Fast Training')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--img_width', type=int, default=400)
    parser.add_argument('--img_height', type=int, default=144)
    parser.add_argument('--cut_height', type=int, default=120)
    parser.add_argument('--train_samples', type=int, default=20000)
    parser.add_argument('--val_samples', type=int, default=1000)
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Mixed precision
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--pretrained', type=str, default=None)
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume', type=str, default=None)
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=None)
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("="*60)
        print("❌ ERROR: CUDA is not available!")
        print("="*60)
        print("PyTorch is running in CPU-only mode.")
        print("\nTo fix this, reinstall PyTorch with CUDA support:")
        print("Visit: https://pytorch.org/get-started/locally/")
        print("\nFor RTX 3050Ti, use:")
        print("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("="*60)
        sys.exit(1)
    
    # Select GPU with auto-detection
    if args.gpu is None:
        selected_gpu = select_best_gpu()
        if selected_gpu is not None:
            args.gpu = selected_gpu
    else:
        # Verify user-selected GPU is NVIDIA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(args.gpu)
            if not any(k in gpu_name.lower() for k in ['nvidia', 'rtx', 'gtx']):
                print(f"\n⚠️  Warning: GPU {args.gpu} ({gpu_name}) may not be your RTX 3050Ti!")
                selected_gpu = select_best_gpu()
                response = input(f"Use GPU {selected_gpu} instead? (y/n): ")
                if response.lower() == 'y':
                    args.gpu = selected_gpu

    # Set device
    args.device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(args.device)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print('='*60)
    print('SCNN Fast Training Configuration')
    print('='*60)
    print(f'PyTorch Version: {torch.__version__}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(args.device)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(args.device).total_memory/1024**3:.2f} GB')
    print(f'Device: {args.device}')
    print('-'*60)
    print(f'Data root: {args.data_root}')
    print(f'Image size: ({args.img_width}, {args.img_height})')
    print(f'Training samples: {args.train_samples}')
    print(f'Validation samples: {args.val_samples}')
    print(f'Batch size: {args.batch_size}')
    print(f'Epochs: {args.epochs}')
    print(f'Learning rate: {args.lr}')
    print(f'Mixed Precision (FP16): {args.use_amp}')
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
    
    total_available = len(train_dataset.data_list)
    train_size = min(args.train_samples, total_available)
    val_size = min(args.val_samples, total_available)
    
    print(f'Dataset size: {total_available} samples')
    print(f'Using {train_size} for training, {val_size} for validation')
    
    # Create validation dataset
    val_transform = CULaneTransform(
        split='train',
        img_size=(args.img_width, args.img_height),
        cut_height=args.cut_height
    )
    val_dataset = CULaneDataset(
        data_root=args.data_root,
        split='train',
        transform=val_transform
    )
    
    # Split data
    val_dataset.data_list = val_dataset.data_list[:val_size]
    train_dataset.data_list = train_dataset.data_list[val_size:val_size + train_size]
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    args.iters_per_epoch = len(train_loader)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Iterations per epoch: {args.iters_per_epoch}')
    
    # Create model
    print('\nCreating model...')
    model = SCNN(
        num_classes=args.num_classes,
        input_size=(args.img_width, args.img_height),
        pretrained=True
    )
    model = model.to(args.device)
    
    # Verify model is on GPU
    print(f'Model device: {next(model.parameters()).device}')
    print(f'Model on CUDA: {next(model.parameters()).is_cuda}')
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # Load pretrained if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.pretrained:
        print(f'\nLoading pretrained weights from {args.pretrained}')
        checkpoint = torch.load(args.pretrained, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Create criterion
    class_weights = torch.tensor([0.4, 1.0, 1.0, 1.0, 1.0]).to(args.device)
    criterion = CombinedLoss(
        seg_weight=1.0,
        exist_weight=0.1,
        class_weights=class_weights
    )
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    scaler = GradScaler('cuda', enabled=args.use_amp)
    
    # Resume if specified
    if args.resume:
        print(f'\nResuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        if 'scaler' in checkpoint and args.use_amp:
            scaler.load_state_dict(checkpoint['scaler'])
        print(f'Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}')
    
    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(args.log_dir, f'run_{timestamp}'))
    
    # Save config
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=4)
    
    print('\n' + '='*60)
    print('Starting training...')
    print('='*60 + '\n')
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            epoch, args, writer
        )
        
        val_loss = validate(model, val_loader, criterion, epoch, args, writer)
        
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        
        print(f'\nEpoch {epoch+1}/{args.epochs} completed in {epoch_time/60:.1f} minutes')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'GPU Memory Used: {torch.cuda.max_memory_allocated(args.device)/1024**3:.2f} GB')
        print(f'Elapsed time: {elapsed_time/3600:.2f} hours')
        
        # ETA
        avg_epoch_time = elapsed_time / (epoch - start_epoch + 1)
        remaining_epochs = args.epochs - epoch - 1
        eta = (avg_epoch_time * remaining_epochs) / 3600
        print(f'ETA: {eta:.2f} hours')
        
        # Save checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_loss': best_loss,
            'args': vars(args)
        }, is_best, args.save_dir, f'epoch_{epoch+1}.pth')
        
        print(f'Checkpoint saved. Best loss: {best_loss:.4f}\n')
        
        # Reset GPU memory stats
        torch.cuda.reset_peak_memory_stats(args.device)
    
    total_time = time.time() - start_time
    writer.close()
    
    print('\n' + '='*60)
    print('Training completed!')
    print(f'Total training time: {total_time/3600:.2f} hours')
    print(f'Best validation loss: {best_loss:.4f}')
    print('='*60)


if __name__ == '__main__':
    main()