# SCNN Lane Detection for CULane Dataset

A PyTorch implementation of Spatial CNN (SCNN) for lane detection on the CULane dataset, optimized for RTX 3050 Ti.

**Author:** ThisisVitou  
**Date:** 2025-11-23

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Common Issues](#common-issues)
- [Citation](#citation)

## 🎯 Overview

This project implements lane detection using SCNN (Spatial Convolutional Neural Network) trained on the CULane dataset. It includes:
- Fast training script with GPU optimization
- Comprehensive evaluation metrics
- Real-time inference on images and videos
- Support for RTX 3050 Ti and other NVIDIA GPUs

## ✨ Features

- ✅ **GPU Auto-Detection**: Automatically selects NVIDIA GPU (avoids Intel integrated graphics)
- ✅ **Mixed Precision Training (FP16)**: 2x faster training with lower memory usage
- ✅ **Overnight Training Mode**: Optimized settings for long training runs
- ✅ **Visual Predictions**: Save side-by-side comparison images
- ✅ **Video Processing**: Process entire videos for lane detection
- ✅ **TensorBoard Integration**: Monitor training progress in real-time

## 📦 Requirements

### Hardware
- NVIDIA GPU with CUDA support (minimum 4GB VRAM)
- 8GB+ system RAM recommended
- 50GB+ disk space for CULane dataset

### Software
- Python 3.8+
- CUDA 11.8 or 12.1
- PyTorch 2.0+ with CUDA support

### trained model
I'll provide a link to google drive for existing model and evaluation. I can't upload anything over 100mb.

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ThisisVitou/scnn-lane-detection.git
cd scnn-lane-detection
```

### 2. Install PyTorch with CUDA
```bash
# For CUDA 11.8 (recommended for RTX 3050 Ti)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Other Dependencies
```bash
pip install opencv-python numpy tqdm tensorboard scikit-learn
```

### 4. Verify CUDA Installation
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
```

### 5. Download CULane Dataset
Download from [CULane official website](https://xingangpan.github.io/projects/CULane.html) and extract to `./culane/`

Expected structure:
```
culane/
├── driver_23_30frame/
├── driver_37_30frame/
├── driver_100_30frame/
├── driver_161_90frame/
├── driver_182_30frame/
└── list/
    ├── train_gt.txt
    ├── test.txt
    └── ...
```

## 🏃 Quick Start

### Training (Quick Test - 6 minutes)
```bash
python train_fast.py \
  --data_root culane \
  --train_samples 5000 \
  --epochs 6 \
  --batch_size 12 \
  --num_workers 2 \
  --use_amp
```

### Training (Overnight - Best Results)
```bash
python train_fast.py \
  --data_root culane \
  --train_samples 80000 \
  --epochs 24 \
  --batch_size 12 \
  --num_workers 2 \
  --lr 0.01 \
  --use_amp \
  --img_width 800 \
  --img_height 288
```

### Evaluate Model
```bash
python evaluate.py \
  --data_root culane \
  --checkpoint checkpoints/best_model.pth \
  --eval_samples 5000 \
  --batch_size 12 \
  --save_predictions
```

### Test on Image
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input path/to/image.jpg \
  --output results/ \
  --show
```

### Test on Video
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input path/to/video.mp4 \
  --output results/output_video.mp4
```

## 📚 Documentation

Detailed documentation for each component:

- **[TRAINING.md](TRAINING.md)** - Complete training guide with all parameters
- **[EVALUATION.md](EVALUATION.md)** - How to evaluate and interpret metrics
- **[TESTING.md](TESTING.md)** - Inference on images and videos
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

## 📁 Project Structure

```
scnn-lane-detection/
├── train_fast.py          # Main training script
├── evaluate.py            # Model evaluation script
├── test.py               # Inference script
├── model.py              # SCNN model architecture
├── dataset.py            # CULane dataset loader
├── data_process.py       # Data augmentation & transforms
├── utils.py              # Helper functions
├── checkpoints/          # Saved model checkpoints
│   ├── best_model.pth
│   └── epoch_*.pth
├── logs/                 # TensorBoard logs
├── eval_results/         # Evaluation outputs
│   ├── predictions/      # Visual predictions
│   └── evaluation_results.json
└── culane/              # CULane dataset
```

## ⚙️ Common Parameters

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | Required | Path to CULane dataset |
| `--train_samples` | 20000 | Number of training samples |
| `--val_samples` | 1000 | Number of validation samples |
| `--epochs` | 12 | Number of training epochs |
| `--batch_size` | 2 | Batch size (increase for faster training) |
| `--lr` | 0.005 | Learning rate |
| `--use_amp` | True | Use mixed precision (FP16) |
| `--img_width` | 400 | Input image width |
| `--img_height` | 144 | Input image height |
| `--num_workers` | 2 | Data loading workers |

### Evaluation Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--eval_samples` | 5000 | Number of samples to evaluate |
| `--save_predictions` | False | Save visual predictions |
| `--batch_size` | 2 | Batch size |

### Testing Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | Required | Path to model checkpoint |
| `--input` | Required | Image, folder, or video path |
| `--output` | ./test_results | Output directory |
| `--show` | False | Display results in window |

## 🔧 Common Issues

### CUDA Not Available
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Using Intel GPU Instead of NVIDIA
The script automatically detects and uses your NVIDIA GPU. If issues persist, check [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Out of Memory Error
Reduce batch size:
```bash
python train_fast.py --batch_size 8  # or lower
```

### Laptop Overheating
Use conservative settings:
```bash
python train_fast.py --batch_size 8 --num_workers 1
```

## 📊 Expected Results

| Training Duration | Samples | Epochs | mIoU | Quality |
|-------------------|---------|--------|------|---------|
| 6 minutes | 5,000 | 6 | 0.35-0.40 | Testing |
| 2 hours | 20,000 | 12 | 0.45-0.50 | Good |
| 10 hours | 80,000 | 24 | 0.55-0.60 | Excellent |
| 30 hours | 88,880 | 50 | 0.60+ | Best |

## 📈 Monitoring Training

### Using TensorBoard
```bash
tensorboard --logdir=./logs
```
Open: http://localhost:6006

### Using nvidia-smi
```bash
# Monitor GPU usage in real-time
nvidia-smi -l 1
```

## 🎓 Citation

If you use this code, please cite:

```bibtex
@article{pan2018scnn,
  title={Spatial as deep: Spatial cnn for traffic scene understanding},
  author={Pan, Xingang and Shi, Jianping and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  journal={AAAI Conference on Artificial Intelligence},
  year={2018}
}
```

## 📝 License

This project is for educational purposes. Please check the original SCNN and CULane licenses.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📧 Contact

**Author:** ThisisVitou  
**GitHub:** [@ThisisVitou](https://github.com/ThisisVitou)

---

**Last Updated:** 2025-11-23