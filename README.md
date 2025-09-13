# YOLO Multi-Model Detection System

A comprehensive YOLO-based detection system for multiple safety and security applications including accident detection, violence detection, weapon detection, fire detection, and NSFW content detection. Now includes **video frame processing** and **single image analysis** capabilities.

## 🚀 Features

- **Accident Detection**: Detect vehicle accidents and crashes
- **Violence/Fight Detection**: Identify violent behavior and fights
- **Weapon Detection**: Detect various weapons including guns, knives, etc.
- **Fire Detection**: Identify fire, smoke, and flames
- **NSFW Content Detection**: Classification and segmentation of inappropriate content
- **Object Detection**: General COCO object detection
- **Video Frame Processing**: Extract frames from videos and analyze with all models
- **Single Image Analysis**: Comprehensive multi-model analysis for single images
- **Batch Processing**: Run multiple detection models sequentially
- **Advanced Filtering**: Filter results by confidence and generate summary reports
- **Weapon Detection Analysis**: Specialized analysis for weapon detections
- **Comprehensive Category Analysis**: Detailed breakdown of all detection categories
- **Automated Cleanup**: Built-in test result cleanup to prevent storage issues

## 📁 Project Structure

```
d:/akhil/
├── integrated_yolo_runner/          # Main unified runner
│   ├── run.py                       # Primary execution script
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # Detailed usage instructions
├── models/                          # Organized model directory
│   ├── accident/                    # Accident detection
│   │   ├── weights/yolov8s.pt
│   │   └── config/data.yaml
│   ├── fight/                       # Violence detection
│   │   └── weights/
│   │       ├── nano_weights.pt
│   │       └── small_weights.pt
│   ├── weapon/                      # Weapon detection
│   │   └── weights/weapon_detection.pt
│   ├── fire/                        # Fire detection
│   │   └── weights/
│   │       ├── yolov8n.pt
│   │       └── yolov8s.pt
│   ├── nsfw/                        # NSFW detection
│   │   └── weights/
│   │       ├── classification_model.pt
│   │       └── segmentation_model.pt
│   ├── model_registry.json          # Model configuration
│   └── ADD_NEW_MODELS.md           # Guide for adding models
├── video_analysis_results/          # Video processing results
├── video_frame_processor.py         # Video frame extraction & analysis
├── single_image_analyzer.py         # Single image multi-model analysis
├── create_filtered_results.py       # Results filtering script
├── weapon_detection_analysis.py     # Weapon-specific analysis
├── comprehensive_category_analysis.py # All-category analysis
├── demo_video_processor.py          # Demo script for video processing
├── VIDEO_PROCESSOR_GUIDE.md         # Video processing guide
├── TESTING_GUIDE.md                 # Testing instructions
├── cleanup_tests.py                 # Automated test cleanup script
├── project_structure.py             # Project optimization tools
└── .gitignore                       # Git ignore rules
```

## 🛠️ Installation

### Quick Setup (Automated)

```bash
# Clone repository
git clone https://github.com/Akhil4007-cpu/moderation-multi_modal.git
cd moderation-multi_modal

# Run automated setup
python setup.py
```

### Manual Installation

#### 1. Prerequisites
- **Python 3.8+** (3.9-3.11 recommended)
- **FFmpeg** (required for video processing)
- **Git** (for cloning repository)

#### 2. Install FFmpeg
```bash
# Windows: Download from https://ffmpeg.org/download.html and add to PATH
# Linux:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg

# Verify installation:
ffmpeg -version
```

#### 3. Clone Repository
```bash
git clone https://github.com/Akhil4007-cpu/moderation-multi_modal.git
cd moderation-multi_modal
```

#### 4. Set Up Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 5. Install Dependencies
```bash
# Install PyTorch first (choose based on your system)
# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 (for NVIDIA GPUs):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

#### 6. Verify Installation
```bash
python -c "import ultralytics, cv2, torch; print('✅ All packages imported successfully')"
ffmpeg -version
```

For detailed installation instructions, see **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)**

## 🎯 Quick Start

### Video Frame Processing (NEW!)

```bash
# Extract frames from video and analyze with all models
python video_frame_processor.py "path/to/video.mp4" --conf 0.5

# Keep extracted frames for inspection
python video_frame_processor.py "path/to/video.mp4" --keep-frames

# Custom output directory
python video_frame_processor.py "path/to/video.mp4" --output "results/" --temp "temp_frames/"
```

### Single Image Analysis (NEW!)

```bash
# Analyze single image with all 9 models
python single_image_analyzer.py "path/to/image.jpg" --conf 0.5

# Save results to specific file
python single_image_analyzer.py "path/to/image.jpg" --output "analysis.json"

# Silent mode (no display, only save)
python single_image_analyzer.py "path/to/image.jpg" --no-display
```

### Results Analysis & Filtering (NEW!)

```bash
# Filter video results by confidence
python create_filtered_results.py "detailed_results.json" --min-conf 0.7 --min-prob 0.7

# Analyze weapon detections specifically
python weapon_detection_analysis.py "detailed_results.json"

# Comprehensive analysis of all categories
python comprehensive_category_analysis.py "detailed_results.json" all
```

### Traditional Single Model Detection

```bash
# Accident detection from webcam
cd integrated_yolo_runner
python run.py --task accident --source 0 --show

# Weapon detection from image
python run.py --task weapon --source path/to/image.jpg --conf 0.5

# Fire detection from video
python run.py --task fire_s --source path/to/video.mp4 --device cuda
```

### Batch Detection (All Models)

```bash
# Run all detection models sequentially
python run.py --run_all --source path/to/media --filter_output

# Run specific categories
python run.py --categories "weapon,fight,fire" --source path/to/media
```

## 📊 Available Models

| Category | Task Key | Model File | Description |
|----------|----------|------------|-------------|
| Accident | `accident` | `yolov8s.pt` | Vehicle accident detection |
| Fight | `fight_nano` | `nano_weights.pt` | Violence detection (fast) |
| Fight | `fight_small` | `small_weights.pt` | Violence detection (accurate) |
| Weapon | `weapon` | `weapon_detection.pt` | Weapon detection |
| Fire | `fire_n` | `yolov8n.pt` | Fire detection (fast) |
| Fire | `fire_s` | `yolov8s.pt` | Fire detection (accurate) |
| NSFW | `nsfw_cls` | `classification_model.pt` | NSFW classification |
| NSFW | `nsfw_seg` | `segmentation_model.pt` | NSFW segmentation |
| Objects | `objects` | `yolov8n.pt` | General object detection |

## 🔧 Configuration Options

### Confidence Thresholds
- `--conf`: General confidence threshold (default: 0.25)
- `--fire_conf`: Fire detection confidence (default: 0.5)
- `--fight_conf`: Fight detection confidence (default: 0.5)
- `--weapon_conf`: Weapon detection confidence (default: 0.5)

### Presets
- `--preset image_relaxed`: Optimized for single images
- `--preset video_strict`: Optimized for videos with temporal consistency

### Output Options
- `--filter_output`: Generate filtered JSON with high-confidence results only
- `--clean_outputs`: Remove previous results before running
- `--nosave`: Don't save visual outputs

## 🧹 Maintenance

### Automatic Test Cleanup

The system includes automated cleanup to prevent storage confusion:

```bash
# Clean test results (keep only 1 latest run per category)
python cleanup_tests.py --keep 1

# Clean test results and temporary files
python cleanup_tests.py --keep 1 --temp

# Clean specific path
python cleanup_tests.py --path d:/akhil/integrated_runs --keep 2
```

### Project Optimization

```bash
# Analyze project structure
python project_structure.py --analyze

# Optimize for deployment (removes virtual environments, etc.)
python project_structure.py --optimize
```

## 🎥 Video Processing Workflow

### Complete Video Analysis Pipeline

1. **Extract and Analyze Frames**:
   ```bash
   python video_frame_processor.py "video.mp4" --conf 0.5
   ```

2. **Filter Results by Confidence**:
   ```bash
   python create_filtered_results.py "detailed_results.json" --min-conf 0.7
   ```

3. **Analyze Specific Categories**:
   ```bash
   # Weapon-specific analysis
   python weapon_detection_analysis.py "detailed_results.json"
   
   # All categories breakdown
   python comprehensive_category_analysis.py "detailed_results.json" all
   ```

### Output Files Generated
- `*_detailed_results.json`: Complete frame-by-frame analysis
- `*_summary.json`: High-level summary with safety assessment
- `*_filtered_final_*.json`: Confidence-filtered results
- `*_weapon_analysis_*.json`: Weapon detection breakdown
- `*_comprehensive_analysis_*.json`: All-category analysis

## 📤 Deployment

### Prepare for Git Upload

1. **Clean the project**:
   ```bash
   python cleanup_tests.py --keep 0 --temp
   python project_structure.py --optimize
   ```

2. **Add and commit changes**:
   ```bash
   git add .
   git commit -m "Updated multi-model detection system with video processing"
   ```

3. **Push to repository**:
   ```bash
   git push origin main
   ```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY integrated_yolo_runner/ .
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

CMD ["python", "run.py", "--help"]
```

## 🔍 Output Format

### Standard Output
Results are saved to `d:/akhil/integrated_runs/<task>/predict*/`:
- Annotated images/videos
- `results.json`: Complete detection results
- `filtered_results.json`: High-confidence results only (if `--filter_output` used)

### JSON Structure
```json
{
  "summary": {
    "task": "weapon",
    "frames": 1,
    "detections": 2
  },
  "items": [
    {
      "image": "path/to/image.jpg",
      "type": "detection",
      "detections": [
        {
          "class_name": "weapon",
          "confidence": 0.85,
          "bbox_xyxy": [x1, y1, x2, y2]
        }
      ]
    }
  ]
}
```

## ⚠️ Important Notes

1. **Model Files**: Ensure all model files (*.pt) are present in their respective directories
2. **Storage Management**: Run cleanup scripts regularly to prevent storage issues
3. **GPU Usage**: Install appropriate PyTorch version for your GPU
4. **Test Results**: Test outputs are automatically cleaned to prevent confusion
5. **Virtual Environments**: Recreate virtual environments on each deployment



## 🏗️ New Model Structure (v2.0)

The project has been restructured for better scalability and organization:

### Benefits of New Structure
- **Scalable**: Easy to add new models without cluttering
- **Organized**: All models follow the same directory pattern  
- **Maintainable**: Clear separation of weights and configuration
- **Future-ready**: Supports multiple model variants per category

### Adding New Models
See `models/ADD_NEW_MODELS.md` for detailed instructions on adding new detection models.

### Model Registry
All models are now registered in `models/model_registry.json` with metadata including:
- Model category and description
- Weight file locations
- Supported input types
- Default confidence thresholds

## 🐛 Troubleshooting

### Common Issues

1. **Missing model files**: Check if all *.pt files exist in their expected locations
2. **CUDA errors**: Ensure PyTorch CUDA version matches your GPU drivers
3. **Permission errors**: Run with appropriate permissions for file operations
4. **Storage full**: Run `python cleanup_tests.py --temp` to free space

### Performance Optimization

- Use `--device cuda` for GPU acceleration
- Adjust `--imgsz` based on your hardware capabilities
- Use appropriate confidence thresholds for your use case
- Consider using nano models for real-time applications

## 📝 License

This project contains multiple YOLO models for safety and security applications. Please ensure compliance with respective model licenses and usage terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run cleanup scripts before committing
5. Submit a pull request

---

For detailed usage instructions, see `integrated_yolo_runner/README.md`.
