# Testing/Inference Guide

Complete guide for running inference with your trained SCNN model on images and videos.

## Table of Contents
- [Quick Start](#quick-start)
- [All Parameters](#all-parameters)
- [Testing Modes](#testing-modes)
- [Understanding Results](#understanding-results)
- [Batch Processing](#batch-processing)

## Quick Start

### Test Single Image
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input path/to/image.jpg \
  --output test_results/ \
  --show
```

### Test Folder of Images
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input path/to/images/ \
  --output test_results/
```

### Test Video
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input path/to/video.mp4 \
  --output test_results/output.mp4
```

## All Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--checkpoint` | str | Path to trained model (.pth file) |
| `--input` | str | Image file, folder, or video file |

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--img_width` | int | 400 | Must match training width |
| `--img_height` | int | 144 | Must match training height |
| `--cut_height` | int | 120 | Pixels to crop from top |

**⚠️ Important:** Must match your training parameters!

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output` | str | ./test_results | Output directory or video path |
| `--show` | flag | False | Display results in window |
| `--gpu` | int | 0 | GPU index to use |

## Testing Modes

### 1. Single Image with Display
Preview results immediately:
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input test_image.jpg \
  --show
```
Press any key to close the window.

### 2. Batch Process Images
Process entire folder:
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input images_folder/ \
  --output results/
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### 3. Process Video
Convert entire video:
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input input_video.mp4 \
  --output output_video.mp4
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`

### 4. Real-time Video Preview
Watch results while processing:
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input video.mp4 \
  --output output.mp4 \
  --show
```
Press `q` to stop processing early.

## Understanding Results

### Visual Output

Each result image shows:
- **Original image** with transparent overlay
- **Colored lane lines**
- **Lane existence information** (top-left corner)

### Lane Colors
- 🔴 **Red:** Lane 1 (leftmost)
- 🟢 **Green:** Lane 2
- 🔵 **Blue:** Lane 3
- 🟡 **Yellow:** Lane 4 (rightmost)

### Lane Information Display
```
Lane 1: ✓ (0.95)  ← Lane detected with 95% confidence
Lane 2: ✓ (0.87)  ← Lane detected with 87% confidence
Lane 3: ✗ (0.23)  ← No lane detected (23% confidence)
Lane 4: ✗ (0.15)  ← No lane detected (15% confidence)
```

### Good Detection
✅ Smooth, continuous lane lines  
✅ Correct number of lanes  
✅ High confidence scores (>0.8)  
✅ Lines follow actual road markings

### Poor Detection
❌ Broken or discontinuous lanes  
❌ False lane detections  
❌ Low confidence scores (<0.5)  
❌ Lines don't match road markings

## Examples

### Example 1: Highway Image
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input examples/highway.jpg \
  --output results/highway_result.jpg \
  --show
```

**Expected Output:**
- 4 lanes detected (all marked)
- High confidence (>0.9)
- Clean, straight lines

### Example 2: Urban Road
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input examples/city_road.jpg \
  --output results/city_result.jpg \
  --show
```

**Expected Output:**
- 2-3 lanes detected
- Medium confidence (0.7-0.9)
- Some occlusions possible

### Example 3: Night Scene
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input examples/night.jpg \
  --output results/night_result.jpg \
  --show
```

**Challenges:**
- Lower confidence scores
- Possible missed lanes
- More false positives

### Example 4: Dashboard Camera Video
```bash
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input dashcam_footage.mp4 \
  --output dashcam_result.mp4
```

**Processing Time (RTX 3050 Ti):**
- **400x144 input:** ~30-40 FPS
- **800x288 input:** ~15-20 FPS
- **1080p video:** ~10-12 FPS

## Batch Processing Script

### Process All Images in Folder

**Windows (process_images.bat):**
```batch
@echo off
echo Processing Images...
python test.py ^
  --checkpoint checkpoints/best_model.pth ^
  --input test_images/ ^
  --output results/
echo Done! Check results/ folder
pause
```

**Linux/Mac (process_images.sh):**
```bash
#!/bin/bash
echo "Processing Images..."
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input test_images/ \
  --output results/
echo "Done! Check results/ folder"
```

### Process Video

**Windows (process_video.bat):**
```batch
@echo off
set INPUT_VIDEO=%1
set OUTPUT_VIDEO=%2
echo Processing video: %INPUT_VIDEO%
python test.py ^
  --checkpoint checkpoints/best_model.pth ^
  --input %INPUT_VIDEO% ^
  --output %OUTPUT_VIDEO%
echo Done! Output: %OUTPUT_VIDEO%
pause
```

Usage:
```bash
process_video.bat input.mp4 output.mp4
```

## Performance Tips

### Speed vs Quality

| Input Size | FPS (RTX 3050Ti) | Quality | Use Case |
|------------|------------------|---------|----------|
| 400x144 | 30-40 | Basic | Real-time preview |
| 640x224 | 20-25 | Good | Balanced |
| 800x288 | 15-20 | Better | Offline processing |
| 1024x384 | 10-12 | Best | High quality |

### Optimize for Speed
```bash
# Use smaller input size
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input video.mp4 \
  --output output.mp4 \
  --img_width 400 \
  --img_height 144
```

### Optimize for Quality
```bash
# Use larger input size (must retrain model)
python test.py \
  --checkpoint checkpoints/best_model_large.pth \
  --input video.mp4 \
  --output output.mp4 \
  --img_width 800 \
  --img_height 288
```

## Use Cases

### 1. Autonomous Driving Dataset
```bash
# Process dataset images
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input autonomous_driving_dataset/ \
  --output predictions/
```

### 2. Dashboard Camera Analysis
```bash
# Analyze dashcam footage
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input dashcam_trip.mp4 \
  --output analyzed_trip.mp4
```

### 3. Quality Assurance
```bash
# Test model on validation set
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input validation_images/ \
  --output qa_results/ \
  --show
```

### 4. Demo/Presentation
```bash
# Real-time demo
python test.py \
  --checkpoint checkpoints/best_model.pth \
  --input demo_video.mp4 \
  --show
```

## Python API Usage

For integration into your own code:

```python
from test import LaneDetector
import cv2

# Initialize detector
detector = LaneDetector(
    model_path='checkpoints/best_model.pth',
    img_size=(800, 288),
    cut_height=120,
    gpu=0
)

# Load image
img = cv2.imread('test.jpg')

# Detect lanes
seg_mask, lane_exist, exist_conf = detector.detect(img)

# Visualize
vis = detector.visualize(img, seg_mask, lane_exist, exist_conf)

# Save or display
cv2.imwrite('result.jpg', vis)
cv2.imshow('Result', vis)
cv2.waitKey(0)
```

### Process Video Programmatically
```python
import cv2
from test import LaneDetector

detector = LaneDetector('checkpoints/best_model.pth')

cap = cv2.VideoCapture('input.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (1920, 1080))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    seg_mask, lane_exist, exist_conf = detector.detect(frame)
    vis = detector.visualize(frame, seg_mask, lane_exist, exist_conf)
    
    out.write(vis)
    cv2.imshow('Processing', vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

## Troubleshooting

### No Lanes Detected
**Possible causes:**
- Model not trained well (low mIoU)
- Wrong image size parameters
- Very different scene from training data

**Solutions:**
- Check evaluation metrics
- Verify `--img_width` and `--img_height` match training
- Train on more diverse data

### False Lane Detections
**Possible causes:**
- Model overfitting
- Low confidence threshold
- Similar patterns to lanes

**Solutions:**
- Use better trained model
- Filter by confidence (>0.7)
- Retrain with more data

### Slow Processing
**Solutions:**
- Reduce input image size
- Use smaller model
- Process on more powerful GPU
- Disable `--show` flag

### Video Output Issues
**Problem:** Output video won't play
**Solutions:**
- Try different codec: modify `test.py` fourcc to `'XVID'` or `'H264'`
- Convert with ffmpeg: `ffmpeg -i output.mp4 -c:v libx264 final.mp4`

---

**Related Guides:**
- [TRAINING.md](TRAINING.md) - Train your own model
- [EVALUATION.md](EVALUATION.md) - Evaluate model performance
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and fixes