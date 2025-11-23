"""
SCNN Model with ResNet-18 Backbone - FIXED for Mixed Precision
Author: ThisisVitou
Date: 2025-11-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SCNN(nn.Module):
    """
    Spatial CNN for lane detection - Fixed in-place operations
    """
    def __init__(self, num_classes=5, input_size=(400, 144), pretrained=True):
        super(SCNN, self).__init__()
        
        self.input_width, self.input_height = input_size
        self.num_classes = num_classes
        
        # Load pretrained ResNet-18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
        
        # Extract ResNet layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # SCNN layers
        self.scnn_d = nn.Conv2d(512, 512, (1, 9), padding=(0, 4), bias=False)
        self.scnn_u = nn.Conv2d(512, 512, (1, 9), padding=(0, 4), bias=False)
        self.scnn_r = nn.Conv2d(512, 512, (9, 1), padding=(4, 0), bias=False)
        self.scnn_l = nn.Conv2d(512, 512, (9, 1), padding=(4, 0), bias=False)
        
        # Segmentation head
        self.seg_conv = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        
        # Existence head
        self.exist_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )
    
    def forward(self, x):
        # ResNet backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # SCNN message passing - FIXED: No in-place operations
        # Process down
        for i in range(1, x.size(2)):
            x_new = x[:, :, i:i+1, :] + F.relu(self.scnn_d(x[:, :, i-1:i, :]))
            x = torch.cat([x[:, :, :i, :], x_new, x[:, :, i+1:, :]], dim=2)
        
        # Process up
        for i in range(x.size(2) - 2, -1, -1):
            x_new = x[:, :, i:i+1, :] + F.relu(self.scnn_u(x[:, :, i+1:i+2, :]))
            x = torch.cat([x[:, :, :i, :], x_new, x[:, :, i+1:, :]], dim=2)
        
        # Process right
        for i in range(1, x.size(3)):
            x_new = x[:, :, :, i:i+1] + F.relu(self.scnn_r(x[:, :, :, i-1:i]))
            x = torch.cat([x[:, :, :, :i], x_new, x[:, :, :, i+1:]], dim=3)
        
        # Process left
        for i in range(x.size(3) - 2, -1, -1):
            x_new = x[:, :, :, i:i+1] + F.relu(self.scnn_l(x[:, :, :, i+1:i+2]))
            x = torch.cat([x[:, :, :, :i], x_new, x[:, :, :, i+1:]], dim=3)
        
        # Segmentation output
        seg_out = self.seg_conv(x)
        seg_out = F.interpolate(
            seg_out,
            size=(self.input_height, self.input_width),
            mode='bilinear',
            align_corners=True
        )
        
        # Existence output
        exist_out = self.exist_conv(x)
        
        return seg_out, exist_out


if __name__ == '__main__':
    print("Testing SCNN Model...")
    model = SCNN(num_classes=5, input_size=(400, 144), pretrained=True)
    
    x = torch.randn(2, 3, 144, 400)
    seg, exist = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Segmentation output: {seg.shape}")
    print(f"Existence output: {exist.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n✓ Model test passed!")