# Training Guide

Complete guide for training the SCNN lane detection model.

## Table of Contents
- [Quick Start](#quick-start)
- [Training Modes](#training-modes)
- [All Parameters](#all-parameters)
- [Optimization Tips](#optimization-tips)
- [Monitoring Training](#monitoring-training)

## Quick Start

### Minimal Command
```bash
python train_fast.py --data_root culane
```
This uses all default values.

### Recommended Quick Test (6 minutes)
```bash
python train_fast.py \
  --data_root culane \
  --train_samples 5000 \
  --val_samples 1000 \
  --epochs 6 \
  --batch_size 12 \
  --num_workers 2 \
  --use_amp
```

## Training Modes

### 1. Quick Test (6 minutes)
**Purpose:** Verify everything works  
**Time:** ~6 minutes  
**Quality:** Low (for testing only)

```bash
python train_fast.py \
  --data_root culane \
  --train_samples 5000 \
  --epochs 6 \
  --batch_size 12 \
  --num_workers 2 \
  --use_amp
```

### 2. Fast Training (2 hours)
**Purpose:** Get a decent model quickly  
**Time:** ~2 hours  
**Quality:** Good

```bash
python train_fast.py \
  --data_root culane \
  --train_samples 20000 \
  --val_samples 2000 \
  --epochs 12 \
  --batch_size 12 \
  --num_workers 2 \
  --lr 0.01 \
  --use_amp \
  --img_width 640 \
  --img_height 224
```

### 3. Overnight Training (10 hours) ⭐ RECOMMENDED
**Purpose:** Best balance of time vs quality  
**Time:** ~10 hours  
**Quality:** Excellent

```bash
python train_fast.py \
  --data_root culane \
  --train_samples 80000 \
  --val_samples 8880 \
  --epochs 24 \
  --batch_size 12 \
  --num_workers 2 \
  --lr 0.01 \
  --use_amp \
  --img_width 800 \
  --img_height 288
```

### 4. Weekend Training (24-30 hours)
**Purpose:** Production-quality model  
**Time:** ~30 hours  
**Quality:** Best possible

```bash
python train_fast.py \
  --data_root culane \
  --train_samples 88880 \
  --val_samples 9675 \
  --epochs 50 \
  --batch_size 10 \
  --num_workers 2 \
  --lr 0.01 \
  --weight_decay 1e-4 \
  --use_amp \
  --img_width 800 \
  --img_height 288 \
  --log_interval 100
```

### 5. Conservative Mode (Cooler Laptop)
**Purpose:** Reduce heat for long runs  
**Time:** ~14 hours  
**Quality:** Good

```bash
python train_fast.py \
  --data_root culane \
  --train_samples 80000 \
  --val_samples 8880 \
  --epochs 24 \
  --batch_size 8 \
  --num_workers 1 \
  --lr 0.01 \
  --use_amp \
  --img_width 640 \
  --img_height 224
```

## All Parameters

### Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data_root` | str | **Required** | Path to CULane dataset directory |
| `--train_samples` | int | 20000 | Number of training samples to use |
| `--val_samples` | int | 1000 | Number of validation samples |
| `--img_width` | int | 400 | Input image width (pixels) |
| `--img_height` | int | 144 | Input image height (pixels) |
| `--cut_height` | int | 120 | Pixels to crop from top of image |

**What to change:**
- `--train_samples`: Increase for better accuracy (max: 88880)
- `--img_width/height`: Larger = better but slower (try 800x288)

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--epochs` | int | 12 | Number of training epochs |
| `--batch_size` | int | 2 | Batch size per GPU |
| `--lr` | float | 0.005 | Initial learning rate |
| `--momentum` | float | 0.9 | SGD momentum |
| `--weight_decay` | float | 1e-4 | Weight decay (L2 regularization) |
| `--num_workers` | int | 2 | Data loading workers |

**What to change:**
- `--batch_size`: **IMPORTANT!** Increase to 12-16 for faster training
- `--epochs`: More epochs = better convergence (try 24-50)
- `--lr`: Start with 0.01 for full dataset training

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_classes` | int | 5 | Number of lane classes (bg + 4 lanes) |
| `--pretrained` | str | None | Path to pretrained checkpoint |
| `--resume` | str | None | Resume training from checkpoint |

**What to change:**
- `--resume`: Use to continue interrupted training
- `--pretrained`: Load weights from previous training

### Performance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use_amp` | flag | True | Enable mixed precision (FP16) training |
| `--no_amp` | flag | False | Disable mixed precision |
| `--gpu` | int | None (auto) | GPU index to use |

**What to change:**
- `--use_amp`: Keep enabled for 2x speedup
- `--gpu`: Usually auto-detected correctly

### Logging Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save_dir` | str | ./checkpoints | Directory to save checkpoints |
| `--log_dir` | str | ./logs | Directory for TensorBoard logs |
| `--log_interval` | int | 50 | Log metrics every N batches |

## Optimization Tips

### For RTX 3050 Ti (4GB VRAM)

| Resolution | Batch Size | Speed | Memory Usage |
|------------|------------|-------|--------------|
| 400x144 | 16 | Fastest | ~3.5GB |
| 640x224 | 12 | Fast | ~3.8GB |
| 800x288 | 10 | Balanced | ~3.9GB |
| 1024x384 | 6 | Slow | ~4.0GB |

**Recommended:** `800x288` with `batch_size=12`

### Adjust Batch Size

```bash
# Try these in order until you get "Out of Memory":
--batch_size 16  # If this works, great!
--batch_size 12  # Usually works well
--batch_size 8   # Conservative
--batch_size 4   # Very safe
```

### Speed vs Quality Trade-offs

**Fastest (2 hours):**
- Small images: 400x144
- Few samples: 20,000
- Few epochs: 12

**Balanced (10 hours):** ⭐
- Medium images: 800x288
- Many samples: 80,000
- Moderate epochs: 24

**Best Quality (30 hours):**
- Large images: 800x288
- All samples: 88,880
- Many epochs: 50

## Monitoring Training

### Check GPU Usage
```bash
# In a separate terminal
nvidia-smi -l 1
```

You should see:
- **GPU Utilization:** 60-85%
- **Memory Used:** 2-4 GB
- **Temperature:** 65-75°C

### View TensorBoard
```bash
tensorboard --logdir=./logs
```
Open: http://localhost:6006

Metrics to watch:
- **Train/Loss** - should decrease steadily
- **Val/Loss** - should decrease (if increasing, overfitting)
- **Train/LR** - learning rate schedule

### Check Progress
Training prints every epoch:
```
Epoch 12/24 completed in 25.3 minutes
Train Loss: 0.2543 | Val Loss: 0.2456
GPU Memory Used: 3.52 GB
Elapsed time: 5.06 hours
ETA: 5.12 hours
✓ New best model saved!
```

## Resume Training

If training is interrupted:

```bash
python train_fast.py \
  --data_root culane \
  --resume checkpoints/checkpoint.pth \
  --epochs 24 \
  # ... other parameters must match original training
```

## Create Training Script

**Windows (train_overnight.bat):**
```batch
@echo off
echo Starting Overnight Training
python train_fast.py ^
  --data_root culane ^
  --train_samples 80000 ^
  --val_samples 8880 ^
  --epochs 24 ^
  --batch_size 12 ^
  --num_workers 2 ^
  --lr 0.01 ^
  --use_amp ^
  --img_width 800 ^
  --img_height 288
pause
```

**Linux/Mac (train_overnight.sh):**
```bash
#!/bin/bash
echo "Starting Overnight Training"
python train_fast.py \
  --data_root culane \
  --train_samples 80000 \
  --val_samples 8880 \
  --epochs 24 \
  --batch_size 12 \
  --num_workers 2 \
  --lr 0.01 \
  --use_amp \
  --img_width 800 \
  --img_height 288
```

## Troubleshooting

### Out of Memory
**Solution:** Reduce batch size
```bash
--batch_size 8  # or even 4
```

### Too Slow
**Solution:** Increase batch size or reduce image size
```bash
--batch_size 16 --img_width 400 --img_height 144
```

### Laptop Overheating
**Solution:** Reduce workers and batch size
```bash
--batch_size 8 --num_workers 1
```

### Not Using NVIDIA GPU
**Solution:** Check with nvidia-smi
```bash
nvidia-smi -l 1
```
Should show 60-85% GPU usage during training.

---

**Next Steps:** After training completes, see [EVALUATION.md](EVALUATION.md)