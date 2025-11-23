"""
SCNN Testing/Inference Script - FIXED
Run inference on test images or videos
Author: ThisisVitou
Date: 2025-11-22
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

from model import SCNN


class LaneDetector:
    """
    Lane detection inference class
    """
    def __init__(self, model_path, img_size=(400, 144), cut_height=120, gpu=0):
        self.img_width, self.img_height = img_size
        self.cut_height = cut_height
        self.device = torch.device(f'cuda:{gpu}')
        
        # Normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Load model
        self.model = SCNN(
            num_classes=5,
            input_size=(self.img_width, self.img_height),
            pretrained=False
        )
        
        # FIXED: Load checkpoint with weights_only=False
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Color map for lanes
        self.colors = [
            (0, 0, 0),       # Background
            (255, 0, 0),     # Lane 1 - red
            (0, 255, 0),     # Lane 2 - green
            (0, 0, 255),     # Lane 3 - blue
            (255, 255, 0)    # Lane 4 - yellow
        ]
        
        print('Lane detector initialized')
    
    def preprocess(self, img):
        """Preprocess image for inference"""
        ori_h, ori_w = img.shape[:2]
        
        # Crop top
        img_cropped = img[self.cut_height:, :, :]
        
        # Resize
        img_resized = cv2.resize(img_cropped, (self.img_width, self.img_height))
        
        # Normalize
        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std
        
        # To tensor
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor, ori_h, ori_w
    
    def postprocess(self, seg_pred, exist_pred, ori_h, ori_w):
        """Postprocess predictions"""
        # Get segmentation mask
        seg_pred = torch.argmax(seg_pred, dim=1).squeeze().cpu().numpy()
        
        # Resize to original size
        seg_pred = cv2.resize(
            seg_pred.astype(np.uint8),
            (ori_w, ori_h - self.cut_height),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Pad top portion
        seg_full = np.zeros((ori_h, ori_w), dtype=np.uint8)
        seg_full[self.cut_height:, :] = seg_pred
        
        # Get existence predictions
        exist_pred = torch.sigmoid(exist_pred).squeeze().cpu().numpy()
        lane_exist = (exist_pred > 0.5).astype(bool)
        
        return seg_full, lane_exist, exist_pred
    
    def detect(self, img):
        """Run lane detection on image"""
        ori_h, ori_w = img.shape[:2]
        
        # Preprocess
        img_tensor, ori_h, ori_w = self.preprocess(img)
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            seg_pred, exist_pred = self.model(img_tensor)
        
        # Postprocess
        seg_mask, lane_exist, exist_conf = self.postprocess(
            seg_pred, exist_pred, ori_h, ori_w
        )
        
        return seg_mask, lane_exist, exist_conf
    
    def visualize(self, img, seg_mask, lane_exist, exist_conf, alpha=0.5):
        """Visualize detection results"""
        vis = img.copy()
        color_mask = np.zeros_like(img)
        
        for class_id in range(1, 5):
            mask = seg_mask == class_id
            if np.any(mask):
                color_mask[mask] = self.colors[class_id]
        
        # Overlay
        vis = cv2.addWeighted(vis, 1 - alpha, color_mask, alpha, 0)
        
        # Add lane existence information
        y_offset = 30
        for i in range(4):
            status = "✓" if lane_exist[i] else "✗"
            color = (0, 255, 0) if lane_exist[i] else (0, 0, 255)
            conf = exist_conf[i]
            text = f"Lane {i+1}: {status} ({conf:.2f})"
            cv2.putText(vis, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        return vis


def process_image(detector, image_path, output_dir, show=False):
    """Process a single image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f'Failed to read image: {image_path}')
        return
    
    # Detect lanes
    seg_mask, lane_exist, exist_conf = detector.detect(img)
    
    # Visualize
    vis = detector.visualize(img, seg_mask, lane_exist, exist_conf)
    
    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, vis)
    
    # Show
    if show:
        cv2.imshow('Lane Detection', vis)
        cv2.waitKey(0)
    
    return vis


def process_video(detector, video_path, output_path, show=False):
    """Process a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Failed to open video: {video_path}')
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f'Processing video: {video_path}')
    print(f'Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}')
    
    # Create video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    pbar = tqdm(total=total_frames, desc='Processing')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect lanes
        seg_mask, lane_exist, exist_conf = detector.detect(frame)
        
        # Visualize
        vis = detector.visualize(frame, seg_mask, lane_exist, exist_conf)
        
        # Write frame
        if output_path:
            out.write(vis)
        
        # Show
        if show:
            cv2.imshow('Lane Detection', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        pbar.update(1)
    
    pbar.close()
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f'Video saved to: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='SCNN Lane Detection Inference')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--img_width', type=int, default=400)
    parser.add_argument('--img_height', type=int, default=144)
    parser.add_argument('--cut_height', type=int, default=120)
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                       help='Input image, folder, or video path')
    parser.add_argument('--output', type=str, default='./test_results',
                       help='Output directory or video path')
    parser.add_argument('--show', action='store_true',
                       help='Show results')
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    print('='*60)
    print('SCNN Lane Detection Inference')
    print('='*60)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Input: {args.input}')
    print(f'Output: {args.output}')
    print('='*60 + '\n')
    
    # Initialize detector
    detector = LaneDetector(
        model_path=args.checkpoint,
        img_size=(args.img_width, args.img_height),
        cut_height=args.cut_height,
        gpu=args.gpu
    )
    
    # Check input type
    input_path = Path(args.input)
    
    if input_path.is_file():
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        
        if input_path.suffix.lower() in video_exts:
            print('Processing video...')
            process_video(detector, str(input_path), args.output, args.show)
        elif input_path.suffix.lower() in image_exts:
            print('Processing image...')
            process_image(detector, str(input_path), args.output, args.show)
        else:
            print(f'Unsupported file format: {input_path.suffix}')
    
    elif input_path.is_dir():
        print('Processing image directory...')
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(glob.glob(os.path.join(str(input_path), ext)))
        
        print(f'Found {len(image_paths)} images')
        
        for img_path in tqdm(image_paths, desc='Processing'):
            process_image(detector, img_path, args.output, args.show)
        
        print(f'\nResults saved to: {args.output}')
    
    else:
        print(f'Invalid input path: {args.input}')
    
    print('\nInference completed!')


if __name__ == '__main__':
    main()

#video_example/05081544_0305/05081544_0305-000001.jpg
#video_example/05081544_0305/05081544_0305-000009.jpgvideo_example/05081544_0305/05081544_0305-005400.jpg