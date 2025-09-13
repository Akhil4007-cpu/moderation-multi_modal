# YOLO Detection System - Testing Guide

## 🚀 Quick Start Testing

### Prerequisites
```bash
cd integrated_yolo_runner
# Activate virtual environment (if using one)
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate    # Linux/Mac

# Install dependencies (if not done)
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📸 Testing Single Images

### Basic Image Testing
```bash
# Test accident detection on an image
python run.py --task accident --source path/to/your/image.jpg

# Test weapon detection with custom confidence
python run.py --task weapon --source path/to/image.jpg --conf 0.6

# Test fire detection and show results
python run.py --task fire_s --source path/to/image.jpg --show

# Test NSFW content (segmentation mode)
python run.py --task nsfw_seg --source path/to/image.jpg --conf 0.7
```

### High-Level Category Testing
```bash
# Use category shortcuts (easier to remember)
python run.py --category accident --source image.jpg
python run.py --category fight --source image.jpg
python run.py --category weapon --source image.jpg
python run.py --category fire --source image.jpg
python run.py --category nsfw --source image.jpg
```

## 🎥 Testing Videos

### Basic Video Testing
```bash
# Test fight detection on video
python run.py --task fight_small --source path/to/video.mp4

# Sample every 2 seconds for faster processing
python run.py --task accident --source video.mp4 --sample_secs 2.0

# Show progress during video processing
python run.py --task fire_s --source video.mp4 --progress
```

### Video with Filtered Results
```bash
# Generate filtered JSON with high-confidence results only
python run.py --task weapon --source video.mp4 --filter_output --conf 0.7
```

## 📹 Testing Live Camera/Webcam

```bash
# Test with webcam (camera index 0)
python run.py --task accident --source 0 --show

# Test with different camera index
python run.py --task fight_small --source 1 --show

# Real-time weapon detection with display
python run.py --category weapon --source 0 --show --conf 0.8
```

## 🔄 Batch Testing (Multiple Models)

### Run All Detection Models
```bash
# Run all 6 categories on single image/video
python run.py --run_all --source path/to/media.jpg --filter_output

# Run specific categories only
python run.py --categories "weapon,fight,fire" --source video.mp4
```

### Batch with Custom Settings
```bash
# Batch processing with strict video preset
python run.py --run_all --source video.mp4 --preset video_strict --filter_output

# Batch with relaxed image preset
python run.py --run_all --source image.jpg --preset image_relaxed --filter_output
```

## 📁 Testing Multiple Files

```bash
# Test entire folder of images
python run.py --task accident --source path/to/image/folder/

# Process folder with custom output location
python run.py --task weapon --source folder/ --project d:/results --name weapon_batch
```

## 🎯 Advanced Testing Options

### Performance Optimization
```bash
# Use GPU for faster processing
python run.py --task fire_s --source video.mp4 --device cuda

# Reduce image size for speed
python run.py --task accident --source video.mp4 --imgsz 416 --device cuda

# CPU-only processing
python run.py --task weapon --source image.jpg --device cpu
```

### Custom Confidence Thresholds
```bash
# Category-specific confidence settings
python run.py --task fire_s --source image.jpg --fire_conf 0.8 --fire_hard 0.95

# Fight detection with strict settings
python run.py --task fight_small --source video.mp4 --fight_conf 0.7 --fight_hard 0.9

# Weapon detection with area filtering
python run.py --task weapon --source image.jpg --weapon_conf 0.6 --weapon_min_area 0.01
```

## 📊 Understanding Results

### Output Locations
Results are saved to: `d:/akhil/integrated_runs/<task>/predict*/`

**Files Generated:**
- `image.jpg` or `video.mp4` - Annotated media with bounding boxes
- `results.json` - Complete detection data
- `filtered_results.json` - High-confidence results only (if `--filter_output` used)

### JSON Result Structure
```json
{
  "summary": {
    "task": "weapon",
    "frames": 1,
    "detections": 2,
    "by_class": {"weapon": 1, "person": 1}
  },
  "items": [
    {
      "image": "path/to/image.jpg",
      "type": "detection",
      "detections": [
        {
          "class_name": "weapon",
          "confidence": 0.85,
          "bbox_xyxy": [100, 50, 200, 150]
        }
      ]
    }
  ]
}
```

### Filtered Results Structure
```json
{
  "category": "weapon",
  "detected": true,
  "objects": [
    {
      "class_name": "weapon",
      "confidence": 0.85,
      "bbox_xyxy": [100, 50, 200, 150],
      "image": "path/to/image.jpg"
    }
  ],
  "category_confidence": 0.85,
  "safety_tag": "unsafe"
}
```

## 🧪 Example Test Commands

### Test with Sample Data
```bash
# Create test directory
mkdir test_media
cd test_media

# Download sample images/videos or use your own files
# Then test:

# Single image test
python ../integrated_yolo_runner/run.py --task accident --source accident_scene.jpg --show

# Video test with progress
python ../integrated_yolo_runner/run.py --task fight_small --source fight_video.mp4 --progress --filter_output

# Batch test all models
python ../integrated_yolo_runner/run.py --run_all --source test_image.jpg --filter_output
```

### Real-World Testing Scenarios
```bash
# Security camera feed
python run.py --category weapon --source rtsp://camera_ip:port/stream --conf 0.8

# Traffic monitoring
python run.py --category accident --source traffic_video.mp4 --accident_require_vehicle

# Content moderation
python run.py --category nsfw --source user_upload.jpg --nsfw_mode seg --conf 0.9

# Fire safety monitoring
python run.py --category fire --source security_cam.mp4 --fire_conf 0.7 --progress
```

## 🔧 Troubleshooting

### Common Issues
```bash
# Model not found error
python run.py --task custom --weights d:/akhil/models/accident/weights/yolov8s.pt --source image.jpg

# Memory issues (reduce image size)
python run.py --task accident --source large_video.mp4 --imgsz 320

# Slow processing (use GPU)
python run.py --task fire_s --source video.mp4 --device 0

# No detections (lower confidence)
python run.py --task weapon --source image.jpg --conf 0.3
```

### Verification Commands
```bash
# Check if models are loaded correctly
python -c "from pathlib import Path; print([f.name for f in Path('d:/akhil/models').rglob('*.pt')])"

# Test GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Verify installation
python -c "import ultralytics, cv2; print('ultralytics:', ultralytics.__version__, 'cv2:', cv2.__version__)"
```

## 📈 Performance Tips

1. **For Images**: Use `--preset image_relaxed` for better detection
2. **For Videos**: Use `--preset video_strict` for accuracy with temporal consistency
3. **For Speed**: Reduce `--imgsz` and use `--device cuda`
4. **For Accuracy**: Increase confidence thresholds and use larger models
5. **For Batch**: Use `--clean_outputs` to avoid storage buildup

## ⚠️ Important: Confidence Threshold Guidelines

### Understanding Detection Sensitivity

The system uses different confidence thresholds that can dramatically affect results:

#### **Preset Comparison**
```bash
# STRICT (High Precision, May Miss Detections)
python run.py --run_all --source video.mp4 --preset video_strict
# Requires ~90%+ confidence, good for production/security

# RELAXED (High Sensitivity, More Detections)  
python run.py --run_all --source image.jpg --preset image_relaxed
# Accepts ~30%+ confidence, good for testing/analysis
```

#### **Manual Confidence Control**
```bash
# Lower thresholds for better detection (recommended for testing)
python run.py --category weapon --source media.jpg --weapon_conf 0.3 --weapon_hard 0.5

# Higher thresholds for production use
python run.py --category weapon --source media.jpg --weapon_conf 0.8 --weapon_hard 0.9
```

### 🎯 **Recommended Settings by Use Case**

| Use Case | Media Type | Preset | Additional Flags | Why |
|----------|------------|--------|------------------|-----|
| **Testing/Analysis** | Images | `image_relaxed` | `--objects_conf 0.3 --weapon_conf 0.3` | Catch more detections |
| **Testing/Analysis** | Videos | `image_relaxed` | `--objects_conf 0.3 --weapon_conf 0.3 --sample_secs 1.0` | Better detection sensitivity |
| **Security/Production** | Images | `video_strict` | Default thresholds | High precision required |
| **Security/Production** | Videos | `video_strict` | `--sample_secs 2.0` | Temporal consistency |
| **Content Moderation** | Images | `image_relaxed` | `--nsfw_conf 0.4 --fight_conf 0.4` | Better coverage |
| **Content Moderation** | Videos | `image_relaxed` | `--nsfw_conf 0.4 --sample_secs 1.5` | Comprehensive scanning |
| **Real-time Monitoring** | Both | Custom | `--conf 0.5 --imgsz 416` | Balance speed/accuracy |

### 🚨 **Common Issues & Solutions**

#### "No Detections Found" - Images
```bash
# Problem: Thresholds too high
python run.py --run_all --source image.jpg --preset video_strict
# Result: {} (empty)

# Solution: Lower confidence thresholds
python run.py --run_all --source image.jpg --preset image_relaxed --objects_conf 0.3 --weapon_conf 0.3
# Result: Detections found!
```

#### "No Detections Found" - Videos
```bash
# Problem: Video strict preset filtering detections
python run.py --run_all --source video.mp4 --preset video_strict
# Result: {} (empty)

# Solution: Use relaxed preset for videos too
python run.py --run_all --source video.mp4 --preset image_relaxed --objects_conf 0.3 --weapon_conf 0.3 --sample_secs 1.0
# Result: Detections found in video frames!
```

#### "Too Many False Positives" - Images
```bash
# Problem: Thresholds too low
python run.py --run_all --source image.jpg --fire_conf 0.1 --weapon_conf 0.1
# Result: Many false detections

# Solution: Increase thresholds
python run.py --run_all --source image.jpg --fire_conf 0.7 --weapon_conf 0.8
# Result: Only high-confidence detections
```

#### "Too Many False Positives" - Videos
```bash
# Problem: Low confidence + frequent sampling
python run.py --run_all --source video.mp4 --preset image_relaxed --sample_secs 0.5
# Result: Too many detections across frames

# Solution: Higher confidence + less frequent sampling
python run.py --run_all --source video.mp4 --weapon_conf 0.6 --sample_secs 2.0
# Result: More reliable detections
```

## 🎯 Quick Test Script

Create a test script `quick_test.py`:
```python
import subprocess
import sys

def test_detection(task, source, show_results=True):
    cmd = [
        sys.executable, "run.py",
        "--task", task,
        "--source", source,
        "--filter_output"
    ]
    if show_results:
        cmd.append("--show")
    
    result = subprocess.run(cmd, cwd="integrated_yolo_runner", capture_output=True, text=True)
    print(f"Testing {task}: {'✅ Success' if result.returncode == 0 else '❌ Failed'}")
    return result.returncode == 0

# Test all models
test_detection("accident", "test_image.jpg")
test_detection("weapon", "test_image.jpg") 
test_detection("fire_s", "test_image.jpg")
```

Run with: `python quick_test.py`
