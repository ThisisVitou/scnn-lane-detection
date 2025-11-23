# Evaluation Guide

Complete guide for evaluating your trained SCNN model.

## Table of Contents
- [Quick Start](#quick-start)
- [All Parameters](#all-parameters)
- [Understanding Metrics](#understanding-metrics)
- [Visual Predictions](#visual-predictions)
- [Compare Models](#compare-models)

## Quick Start

### Basic Evaluation
```bash
python evaluate.py \
  --data_root culane \
  --checkpoint checkpoints/best_model.pth \
  --eval_samples 5000
```

### With Visual Predictions (Recommended)
```bash
python evaluate.py \
  --data_root culane \
  --checkpoint checkpoints/best_model.pth \
  --eval_samples 5000 \
  --batch_size 12 \
  --num_workers 2 \
  --save_predictions \
  --output_dir eval_results
```

## All Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--data_root` | str | Path to CULane dataset |
| `--checkpoint` | str | Path to model checkpoint (.pth file) |

### Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--img_width` | int | 400 | Must match training width |
| `--img_height` | int | 144 | Must match training height |
| `--cut_height` | int | 120 | Must match training value |
| `--eval_samples` | int | 5000 | Number of test samples |

**⚠️ Important:** Image dimensions must match training!

### Evaluation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--batch_size` | int | 2 | Evaluation batch size |
| `--num_workers` | int | 2 | Data loading workers |
| `--num_classes` | int | 5 | Number of classes (bg + 4 lanes) |

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output_dir` | str | ./eval_results | Output directory |
| `--save_predictions` | flag | False | Save visual predictions |
| `--gpu` | int | 0 | GPU index to use |

## Evaluation Modes

### 1. Quick Evaluation (2 minutes)
Test model performance quickly:
```bash
python evaluate.py \
  --data_root culane \
  --checkpoint checkpoints/best_model.pth \
  --eval_samples 1000 \
  --batch_size 12 \
  --save_predictions
```

### 2. Standard Evaluation (10 minutes) ⭐
Reliable performance estimate:
```bash
python evaluate.py \
  --data_root culane \
  --checkpoint checkpoints/best_model.pth \
  --eval_samples 5000 \
  --batch_size 12 \
  --num_workers 2 \
  --save_predictions
```

### 3. Full Evaluation (1 hour)
Most accurate results:
```bash
python evaluate.py \
  --data_root culane \
  --checkpoint checkpoints/best_model.pth \
  --eval_samples 30000 \
  --batch_size 12 \
  --num_workers 2 \
  --img_width 800 \
  --img_height 288 \
  --save_predictions
```

## Understanding Metrics

### Example Output
```
============================================================
Evaluation Results
============================================================
Mean IoU: 0.5834
Pixel Accuracy: 0.9523
Existence Accuracy: 0.9756

Per-class IoU:
  Background: 0.9812
  Lane 1: 0.6123
  Lane 2: 0.5987
  Lane 3: 0.5745
  Lane 4: 0.5632
============================================================
```

### Metric Explanations

#### 1. Mean IoU (Intersection over Union)
**Range:** 0.0 - 1.0 (higher is better)

| mIoU Score | Quality | Description |
|------------|---------|-------------|
| < 0.40 | Poor | Model needs more training |
| 0.40 - 0.50 | Decent | Acceptable for basic use |
| 0.50 - 0.60 | Good | Suitable for most applications |
| 0.60 - 0.70 | Excellent | Professional quality |
| > 0.70 | Outstanding | State-of-the-art |

**Formula:** IoU = (Prediction ∩ Ground Truth) / (Prediction ∪ Ground Truth)

#### 2. Pixel Accuracy
**Range:** 0.0 - 1.0 (higher is better)

| Accuracy | Quality |
|----------|---------|
| < 0.85 | Poor |
| 0.85 - 0.90 | Decent |
| 0.90 - 0.95 | Good |
| > 0.95 | Excellent |

**What it means:** Percentage of pixels correctly classified

#### 3. Existence Accuracy
**Range:** 0.0 - 1.0 (higher is better)

| Accuracy | Quality |
|----------|---------|
| < 0.90 | Needs improvement |
| 0.90 - 0.95 | Good |
| > 0.95 | Excellent |

**What it means:** How well the model detects whether lanes exist in the image

### Per-Class IoU
- **Background:** Usually high (>0.95) because most pixels are background
- **Lane 1-4:** Individual lane detection accuracy
- Lower values indicate difficulty detecting that specific lane

## Visual Predictions

### Viewing Predictions
After running with `--save_predictions`, check:
```
eval_results/
├── evaluation_results.json
└── predictions/
    ├── result_0000.png
    ├── result_0001.png
    ├── result_0002.png
    └── ...
```

Each image shows:
```
┌─────────────┬──────────────┬──────────────┐
│   Original  │ Ground Truth │  Prediction  │
└─────────────┴──────────────┴──────────────┘
```

### Color Coding
- **Black:** Background
- **Red:** Lane 1
- **Green:** Lane 2
- **Blue:** Lane 3
- **Yellow:** Lane 4

### What to Look For

**Good Predictions:**
- ✅ Lanes match ground truth closely
- ✅ Smooth, continuous lane lines
- ✅ Correct number of lanes detected
- ✅ Few false positives

**Bad Predictions:**
- ❌ Broken or discontinuous lanes
- ❌ Missing lanes
- ❌ Extra false lanes
- ❌ Lanes in wrong positions

## Compare Models

### Evaluate Multiple Checkpoints
```bash
# Evaluate epoch 6
python evaluate.py \
  --checkpoint checkpoints/epoch_6.pth \
  --data_root culane \
  --eval_samples 1000 \
  --output_dir eval_results/epoch_6

# Evaluate epoch 12
python evaluate.py \
  --checkpoint checkpoints/epoch_12.pth \
  --data_root culane \
  --eval_samples 1000 \
  --output_dir eval_results/epoch_12

# Evaluate best model
python evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --data_root culane \
  --eval_samples 1000 \
  --output_dir eval_results/best
```

### Compare Results
```bash
# View JSON results
cat eval_results/epoch_6/evaluation_results.json
cat eval_results/epoch_12/evaluation_results.json
cat eval_results/best/evaluation_results.json
```

Example comparison:
| Checkpoint | mIoU | Pixel Acc | Exist Acc |
|------------|------|-----------|-----------|
| epoch_6.pth | 0.4523 | 0.9234 | 0.9456 |
| epoch_12.pth | 0.5234 | 0.9445 | 0.9678 |
| best_model.pth | 0.5834 | 0.9523 | 0.9756 |

## Evaluation Script

Create `eval_model.bat` (Windows) or `eval_model.sh` (Linux/Mac):

**Windows:**
```batch
@echo off
echo Evaluating SCNN Model
python evaluate.py ^
  --data_root culane ^
  --checkpoint checkpoints/best_model.pth ^
  --eval_samples 5000 ^
  --batch_size 12 ^
  --num_workers 2 ^
  --save_predictions ^
  --output_dir eval_results
pause
```

**Linux/Mac:**
```bash
#!/bin/bash
echo "Evaluating SCNN Model"
python evaluate.py \
  --data_root culane \
  --checkpoint checkpoints/best_model.pth \
  --eval_samples 5000 \
  --batch_size 12 \
  --num_workers 2 \
  --save_predictions \
  --output_dir eval_results
```

## Interpreting Results

### Scenario 1: Good Results
```
mIoU: 0.5834
Pixel Accuracy: 0.9523
Existence Accuracy: 0.9756
```
**Interpretation:** Model is working well! Ready for deployment.  
**Next Step:** Test on real images/videos ([TESTING.md](TESTING.md))

### Scenario 2: Low mIoU (<0.40)
```
mIoU: 0.3234
Pixel Accuracy: 0.8923
Existence Accuracy: 0.9156
```
**Possible Issues:**
- Not enough training epochs
- Too few training samples
- Learning rate too high/low
- Model underfitting

**Solutions:**
- Train longer (more epochs)
- Use more training data
- Adjust learning rate
- Check training logs for issues

### Scenario 3: High Pixel Acc, Low mIoU
```
mIoU: 0.4012
Pixel Accuracy: 0.9734
Existence Accuracy: 0.9856
```
**Interpretation:** Model detects background well but struggles with lanes

**Solutions:**
- Increase class weights for lanes
- Train on more diverse samples
- Use data augmentation

### Scenario 4: Low Exist Acc
```
mIoU: 0.5534
Pixel Accuracy: 0.9523
Existence Accuracy: 0.8456
```
**Interpretation:** Model struggles to detect lane presence

**Solutions:**
- Increase `exist_weight` in loss function
- Add more training epochs
- Check if dataset is imbalanced

## Benchmarks

### CULane Dataset (State-of-the-art)
| Method | mIoU | Notes |
|--------|------|-------|
| SCNN (Paper) | 0.71 | Original implementation |
| This Implementation | 0.55-0.65 | Depends on training time |
| Your Quick Model | 0.35-0.40 | 6 epochs, 5k samples |
| Your Overnight Model | 0.55-0.60 | 24 epochs, 80k samples |
| Your Weekend Model | 0.60+ | 50 epochs, full dataset |

## Troubleshooting

### Error: Image size mismatch
**Problem:** Different image size than training
**Solution:** Use same `--img_width` and `--img_height` as training

### Error: Checkpoint not found
**Problem:** Wrong path to checkpoint
**Solution:** Check path with `ls checkpoints/`

### Low Performance
**Problem:** Model performs poorly
**Solutions:**
1. Train longer (more epochs)
2. Use more data (increase `--train_samples`)
3. Use larger images (800x288 instead of 400x144)
4. Check if training converged (view logs)

---

**Next Steps:** Test your model on real images - see [TESTING.md](TESTING.md)