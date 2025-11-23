"""
SCNN Evaluation Script for CULane Dataset - FIXED
Evaluates model performance on test set
Author: ThisisVitou
Date: 2025-11-22
"""

import os
import sys
import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dataset import CULaneDataset
from data_process import CULaneTransform
from model import SCNN


class LaneEvaluator:
    """
    CULane evaluation metrics
    """
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.exist_correct = 0
        self.exist_total = 0
    
    def update(self, seg_pred, seg_target, exist_pred, exist_target):
        """
        Update metrics with batch predictions
        """
        # Segmentation metrics
        seg_pred = torch.argmax(seg_pred, dim=1).cpu().numpy()
        seg_target = seg_target.cpu().numpy()
        
        # Flatten and compute confusion matrix
        mask = (seg_target >= 0) & (seg_target < self.num_classes)
        label = self.num_classes * seg_target[mask].astype('int') + seg_pred[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)
        
        # Existence accuracy
        exist_pred = torch.sigmoid(exist_pred).cpu().numpy()
        exist_pred = (exist_pred > 0.5).astype('int')
        exist_target = exist_target.cpu().numpy().astype('int')
        
        self.exist_correct += np.sum(exist_pred == exist_target)
        self.exist_total += exist_target.size
    
    def get_metrics(self):
        """
        Compute final metrics
        """
        pos = self.confusion_matrix.sum(1)
        res = self.confusion_matrix.sum(0)
        tp = np.diag(self.confusion_matrix)
        
        # IoU for each class
        iou = tp / np.maximum(1.0, pos + res - tp)
        
        # Mean IoU (excluding background)
        miou = np.nanmean(iou[1:])
        
        # Pixel accuracy
        pixel_acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        
        # Existence accuracy
        exist_acc = self.exist_correct / self.exist_total if self.exist_total > 0 else 0
        
        return {
            'mIoU': miou,
            'pixel_acc': pixel_acc,
            'exist_acc': exist_acc,
            'iou_per_class': iou
        }


def evaluate(model, test_loader, args):
    """
    Evaluate model on test set
    """
    model.eval()
    evaluator = LaneEvaluator(num_classes=args.num_classes)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        
        for batch_idx, (imgs, masks, lane_exist) in enumerate(pbar):
            imgs = imgs.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            lane_exist = lane_exist.cuda(non_blocking=True)
            
            # Forward pass
            seg_pred, exist_pred = model(imgs)
            
            # Update metrics
            evaluator.update(seg_pred, masks, exist_pred, lane_exist)
            
            # Update progress bar
            if batch_idx % 100 == 0:
                metrics = evaluator.get_metrics()
                pbar.set_postfix({
                    'mIoU': f'{metrics["mIoU"]:.4f}',
                    'PixAcc': f'{metrics["pixel_acc"]:.4f}',
                    'ExistAcc': f'{metrics["exist_acc"]:.4f}'
                })
    
    # Get final metrics
    metrics = evaluator.get_metrics()
    
    return metrics


def save_predictions(model, test_loader, save_dir, args):
    """
    Save prediction visualizations
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Color map for visualization
    colors = [
        (0, 0, 0),       # Background - black
        (255, 0, 0),     # Lane 1 - red
        (0, 255, 0),     # Lane 2 - green
        (0, 0, 255),     # Lane 3 - blue
        (255, 255, 0)    # Lane 4 - yellow
    ]
    
    num_save = min(50, len(test_loader.dataset))
    saved = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Saving predictions')
        
        for batch_idx, (imgs, masks, lane_exist) in enumerate(pbar):
            if saved >= num_save:
                break
            
            imgs_cuda = imgs.cuda(non_blocking=True)
            
            # Forward pass
            seg_pred, exist_pred = model(imgs_cuda)
            seg_pred = torch.argmax(seg_pred, dim=1).cpu().numpy()
            
            # Process each image in batch
            for i in range(imgs.size(0)):
                if saved >= num_save:
                    break
                
                # Get image and prediction
                img = imgs[i].permute(1, 2, 0).numpy()
                pred = seg_pred[i]
                gt = masks[i].numpy()
                
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img * std + mean) * 255
                img = np.clip(img, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Create colored masks
                pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                gt_color = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
                
                for class_id, color in enumerate(colors):
                    pred_color[pred == class_id] = color
                    gt_color[gt == class_id] = color
                
                # Overlay on image
                pred_overlay = cv2.addWeighted(img, 0.6, pred_color, 0.4, 0)
                gt_overlay = cv2.addWeighted(img, 0.6, gt_color, 0.4, 0)
                
                # Concatenate images
                result = np.hstack([img, gt_overlay, pred_overlay])
                
                # Add text
                cv2.putText(result, 'Original', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result, 'Ground Truth', (img.shape[1] + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result, 'Prediction', (img.shape[1]*2 + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save
                save_path = os.path.join(save_dir, f'result_{saved:04d}.png')
                cv2.imwrite(save_path, result)
                
                saved += 1


def main():
    parser = argparse.ArgumentParser(description='SCNN Evaluation on CULane')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to CULane dataset')
    parser.add_argument('--img_width', type=int, default=400)
    parser.add_argument('--img_height', type=int, default=144)
    parser.add_argument('--cut_height', type=int, default=120)
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--eval_samples', type=int, default=5000,
                       help='Number of samples to evaluate (default: 5000)')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction visualizations')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Directory to save results')
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    
    print('='*60)
    print('SCNN Evaluation')
    print('='*60)
    print(f'Data root: {args.data_root}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Image size: ({args.img_width}, {args.img_height})')
    print(f'Evaluation samples: {args.eval_samples}')
    print('='*60)
    
    # Create dataset
    print('\nLoading dataset...')
    test_transform = CULaneTransform(
        split='train',
        img_size=(args.img_width, args.img_height),
        cut_height=args.cut_height
    )
    test_dataset = CULaneDataset(
        data_root=args.data_root,
        split='train',
        transform=test_transform
    )
    
    # Limit evaluation samples
    test_dataset.data_list = test_dataset.data_list[:args.eval_samples]
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'Test samples: {len(test_dataset)}')
    
    # Create model
    print('\nLoading model...')
    model = SCNN(
        num_classes=args.num_classes,
        input_size=(args.img_width, args.img_height),
        pretrained=False
    )
    
    # Load checkpoint - FIXED: Added weights_only=False
    print(f'Loading checkpoint from: {args.checkpoint}')
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            epoch = checkpoint.get('epoch', 'N/A')
            best_loss = checkpoint.get('best_loss', 'N/A')
            print(f'Loaded checkpoint from epoch {epoch}')
            print(f'Best training loss: {best_loss}')
        else:
            model.load_state_dict(checkpoint)
            print('Loaded state dict directly')
    except Exception as e:
        print(f'Error loading checkpoint: {e}')
        sys.exit(1)
    
    model = model.cuda()
    model.eval()
    
    print('Model loaded successfully')
    
    # Evaluate
    print('\n' + '='*60)
    print('Starting evaluation...')
    print('='*60 + '\n')
    
    metrics = evaluate(model, test_loader, args)
    
    # Print results
    print('\n' + '='*60)
    print('Evaluation Results')
    print('='*60)
    print(f'Mean IoU: {metrics["mIoU"]:.4f}')
    print(f'Pixel Accuracy: {metrics["pixel_acc"]:.4f}')
    print(f'Existence Accuracy: {metrics["exist_acc"]:.4f}')
    print('\nPer-class IoU:')
    class_names = ['Background', 'Lane 1', 'Lane 2', 'Lane 3', 'Lane 4']
    for i, (name, iou) in enumerate(zip(class_names, metrics['iou_per_class'])):
        print(f'  {name}: {iou:.4f}')
    print('='*60)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        results_dict = {
            'checkpoint': args.checkpoint,
            'eval_samples': args.eval_samples,
            'mIoU': float(metrics['mIoU']),
            'pixel_acc': float(metrics['pixel_acc']),
            'exist_acc': float(metrics['exist_acc']),
            'iou_per_class': {
                name: float(iou) 
                for name, iou in zip(class_names, metrics['iou_per_class'])
            }
        }
        json.dump(results_dict, f, indent=4)
    
    print(f'\nResults saved to {results_path}')
    
    # Save predictions if requested
    if args.save_predictions:
        print('\nSaving prediction visualizations...')
        pred_dir = os.path.join(args.output_dir, 'predictions')
        save_predictions(model, test_loader, pred_dir, args)
        print(f'Predictions saved to {pred_dir}')
    
    print('\nEvaluation completed!')


if __name__ == '__main__':
    main()