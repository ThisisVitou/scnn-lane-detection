# Troubleshooting Guide

Common issues and solutions for SCNN lane detection.

## Table of Contents
- [Installation Issues](#installation-issues)
- [GPU Issues](#gpu-issues)
- [Training Issues](#training-issues)
- [Evaluation Issues](#evaluation-issues)
- [Testing Issues](#testing-issues)

## Installation Issues

### CUDA Not Available
**Symptom:**
```
❌ ERROR: CUDA is not available!
```

**Solution 1:** Install PyTorch with CUDA
```bash
# Uninstall CPU version
pip uninstall torch torchvision torchaudio

# Install CUDA version (for RTX 3050 Ti)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Solution 2:** Check NVIDIA drivers
```bash
nvidia-smi
```
If this fails, install/update NVIDIA drivers from:
https://www.nvidia.com/download/index.aspx

**Solution 3:** Verify installation
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Version:', torch.version.cuda)"
```

Expected:
```
CUDA: True
Version: 11.8
```

### Import Errors
**Symptom:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**
```bash
pip install opencv-python numpy tqdm tensorboard scikit-learn
```

### Dataset Not Found
**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'culane/list/train_gt.txt'
```

**Solution:**
1. Download CULane dataset
2. Extract to `./culane/` folder
3. Verify structure:
```
culane/
├── driver_23_30frame/
├── driver_37_30frame/
└── list/
    └── train_gt.txt
```

## GPU Issues

### Using Intel GPU Instead of NVIDIA
**Symptom:**
```
GPU: Intel(R) UHD Graphics
```
But you have RTX 3050 Ti installed.

**Solution 1:** Script auto-detection (already implemented)
The training script now automatically selects NVIDIA GPU.

**Solution 2:** Force GPU selection
```bash
# Try GPU 0
CUDA_VISIBLE_DEVICES=0 python train_fast.py --data_root culane

# If that doesn't work, try GPU 1
CUDA_VISIBLE_DEVICES=1 python train_fast.py --data_root culane
```

**Solution 3:** Check which GPU is NVIDIA
```python
# check_gpu.py
import torch
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

Run it:
```bash
python check_gpu.py
```

Use the index of your NVIDIA GPU:
```bash
python train_fast.py --data_root culane --gpu 1  # if NVIDIA is GPU 1
```

### Task Manager Shows No GPU Usage
**Symptom:** Task Manager shows 0% GPU usage but training is running.

**Solution:** This is a Windows bug. Check actual usage with:
```bash
nvidia-smi -l 1
```

You should see:
- GPU Utilization: 60-85%
- Memory Used: 1000-3000 MiB
- Temperature: 65-75°C

**Trust `nvidia-smi`, not Task Manager!**

### Out of Memory Error
**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solution 1:** Reduce batch size
```bash
python train_fast.py --batch_size 8  # try 8, 6, or 4
```

**Solution 2:** Reduce image size
```bash
python train_fast.py --img_width 400 --img_height 144 --batch_size 16
```

**Solution 3:** Clear GPU cache
```python
import torch
torch.cuda.empty_cache()
```

**Memory Requirements (RTX 3050 Ti - 4GB):**
| Resolution | Batch Size | Memory Used |
|------------|------------|-------------|
| 400x144 | 16 | ~3.5GB |
| 640x224 | 12 | ~3.8GB |
| 800x288 | 10 | ~3.9GB |
| 1024x384 | 6 | ~4.0GB |

### GPU Temperature Too High
**Symptom:** GPU reaches 85°C+ during training

**Solution 1:** Reduce batch size (reduces power)
```bash
python train_fast.py --batch_size 8 --num_workers 1
```

**Solution 2:** Improve cooling
- Use laptop cooling pad
- Elevate laptop back
- Clean dust from vents
- Ensure good airflow

**Solution 3:** Limit GPU power (Linux)
```bash
nvidia-smi -pl 35  # Limit to 35W (adjust as needed)
```

**Safe temperatures:**
- < 75°C: Perfect ✅
- 75-83°C: Normal ✅
- 83-87°C: Warm ⚠️
- > 87°C: Too hot 🔥 (reduce load)

## Training Issues

### Training is Very Slow
**Symptom:** < 5 iterations/second

**Possible Causes & Solutions:**

**1. Batch size too small**
```bash
# Current (slow)
--batch_size 2

# Better (faster)
--batch_size 12
```

**2. Not using AMP**
```bash
# Make sure this flag is set
--use_amp
```

**3. Too many workers**
```bash
# Try reducing
--num_workers 2  # or even 1
```

**4. CPU bottleneck**
- Close other applications
- Reduce `--num_workers`
- Use SSD instead of HDD for dataset

**5. Using CPU instead of GPU**
Check with:
```bash
nvidia-smi -l 1
```
Should show 60-85% GPU usage.

### Loss Not Decreasing
**Symptom:** Loss stays high or increases

**Solution 1:** Lower learning rate
```bash
python train_fast.py --lr 0.001  # instead of 0.01
```

**Solution 2:** Train longer
```bash
python train_fast.py --epochs 24  # instead of 6
```

**Solution 3:** Use more data
```bash
python train_fast.py --train_samples 80000  # instead of 5000
```

**Solution 4:** Check data loading
Verify dataset loads correctly:
```python
from dataset import CULaneDataset
from data_process import CULaneTransform

transform = CULaneTransform('train', (400, 144))
dataset = CULaneDataset('culane', 'train', transform)
img, mask, exist = dataset[0]
print(f"Image shape: {img.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Lane exist: {exist}")
```

### Training Crashes/Freezes
**Symptom:** Training stops without error or freezes

**Solution 1:** Reduce workers
```bash
python train_fast.py --num_workers 1
```

**Solution 2:** Disable pin_memory
Edit `train_fast.py` and change:
```python
pin_memory=False  # instead of True
```

**Solution 3:** Check disk space
```bash
df -h  # Linux/Mac
dir   # Windows
```
Need at least 10GB free for checkpoints and logs.

**Solution 4:** Monitor system resources
```bash
# Linux
htop

# Windows
Task Manager → Performance
```

### ValueError: Not Enough Values to Unpack
**Symptom:**
```
ValueError: not enough values to unpack (expected 3, got 2)
```

**Solution:** Check dataset format. Make sure dataset returns (image, mask, lane_exist):
```python
# dataset.py should return:
return img, mask, lane_exist
```

### Permission Denied Error
**Symptom:**
```
PermissionError: [Errno 13] Permission denied: 'checkpoints/best_model.pth'
```

**Solution 1:** Close other programs using the file
- Close TensorBoard
- Close Python processes

**Solution 2:** Check file permissions
```bash
# Linux/Mac
chmod 777 checkpoints/

# Windows
# Right-click folder → Properties → Security → Edit permissions
```

## Evaluation Issues

### Image Size Mismatch
**Symptom:**
```
RuntimeError: size mismatch, expected X, got Y
```

**Solution:** Use same image size as training
```bash
# If trained with 800x288
python evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --data_root culane \
  --img_width 800 \
  --img_height 288
```

### Checkpoint Not Loading
**Symptom:**
```
RuntimeError: Error(s) in loading state_dict
```

**Solution 1:** Check checkpoint path
```bash
ls checkpoints/
```

**Solution 2:** Verify checkpoint is valid
```python
import torch
checkpoint = torch.load('checkpoints/best_model.pth')
print(checkpoint.keys())
```

Should show: `dict_keys(['state_dict', 'optimizer', 'epoch', 'best_loss', ...])`

**Solution 3:** Use strict=False
Edit `evaluate.py`:
```python
model.load_state_dict(checkpoint['state_dict'], strict=False)
```

### Low mIoU (<0.40)
**Symptom:** Poor evaluation metrics

**Causes & Solutions:**

**1. Insufficient training**
- Train for more epochs (24-50)
- Use more training samples (80k+)

**2. Wrong hyperparameters**
- Adjust learning rate
- Increase batch size

**3. Data issues**
- Check if dataset is corrupted
- Verify image preprocessing

**4. Model issues**
- Try different model architecture
- Check if model is overfitting (val loss > train loss)

## Testing Issues

### No Lanes Detected
**Symptom:** Output image has no colored lanes

**Solution 1:** Check model performance
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --data_root culane --eval_samples 1000
```
If mIoU < 0.40, retrain model.

**Solution 2:** Check input image
- Make sure image contains roads/lanes
- Check if image is readable:
```python
import cv2
img = cv2.imread('test.jpg')
print(img.shape if img is not None else "Failed to read")
```

**Solution 3:** Adjust confidence threshold
Edit `test.py`:
```python
# Lower threshold
lane_exist = (exist_pred > 0.3).astype(bool)  # instead of 0.5
```

### Video Won't Save
**Symptom:** Video processing completes but file is 0 bytes

**Solution 1:** Change codec
Edit `test.py`:
```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # instead of 'mp4v'
```

**Solution 2:** Convert with ffmpeg
```bash
ffmpeg -i output.mp4 -c:v libx264 output_fixed.mp4
```

**Solution 3:** Check output path
```bash
# Make sure directory exists
mkdir -p test_results
python test.py --input video.mp4 --output test_results/output.mp4
```

### Slow Video Processing
**Symptom:** < 5 FPS processing speed

**Solution 1:** Use smaller input size
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input video.mp4 \
  --output output.mp4 \
  --img_width 400 \
  --img_height 144
```

**Solution 2:** Disable display
```bash
# Don't use --show flag for faster processing
python test.py --input video.mp4 --output output.mp4
```

**Solution 3:** Batch process frames
Modify `test.py` to process multiple frames at once (advanced).

## General Issues

### Python Version Too Old
**Symptom:**
```
SyntaxError: f-string expression part cannot include a backslash
```

**Solution:** Upgrade Python
```bash
python --version  # Should be 3.8+
```

If older, install Python 3.8+ from python.org

### Conda Environment Issues
**Symptom:** Packages installed but not found

**Solution:**
```bash
# Make sure you're in the right environment
conda activate your_env_name

# Verify Python path
which python  # Linux/Mac
where python  # Windows
```

### Disk Space Full
**Symptom:**
```
OSError: [Errno 28] No space left on device
```

**Solution:**
```bash
# Check space
df -h  # Linux/Mac

# Clean up
rm -rf logs/old_runs/
rm checkpoints/epoch_*.pth  # Keep only best_model.pth
```

---

Still having issues? Check:
1. [GitHub Issues](https://github.com/ThisisVitou/scnn-lane-detection/issues)
2. Error messages carefully - Google the exact error
3. PyTorch documentation
4. CUDA toolkit documentation

**For more help, open an issue with:**
- Full error message
- Your command
- System info (Python version, GPU, OS)
- Steps to reproduce