# YOLO Multi-Model Detection System

A comprehensive YOLO-based detection system for multiple safety and security applications including accident detection, violence detection, weapon detection, fire detection, and NSFW content detection.

## 🚀 Features

- **Accident Detection**: Detect vehicle accidents and crashes
- **Violence/Fight Detection**: Identify violent behavior and fights
- **Weapon Detection**: Detect various weapons including guns, knives, etc.
- **Fire Detection**: Identify fire, smoke, and flames
- **NSFW Content Detection**: Classification and segmentation of inappropriate content
- **Object Detection**: General COCO object detection
- **Batch Processing**: Run multiple detection models sequentially
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
├── cleanup_tests.py                # Automated test cleanup script
├── project_structure.py            # Project optimization tools
└── .gitignore                      # Git ignore rules
```

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd akhil
```

### 2. Set Up Virtual Environment

```bash
cd integrated_yolo_runner
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# Install PyTorch (choose based on your system)
# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 (adjust for your GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🎯 Quick Start

### Single Model Detection

```bash
# Accident detection from webcam
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

### High-Level Categories

```bash
# Use category shortcuts
python run.py --category fight --source path/to/video.mp4
python run.py --category nsfw --nsfw_mode seg --source path/to/image.jpg
```

## 📊 Available Models

| Category | Task Key | Model File | Description |
|----------|----------|------------|-------------|
| Accident | `accident` | `yolov8s.pt` | Vehicle accident detection |
| Fight | `fight_nano` | `Yolo_nano_weights.pt` | Violence detection (fast) |
| Fight | `fight_small` | `yolo_small_weights.pt` | Violence detection (accurate) |
| Weapon | `weapon` | `best (3).pt` | Weapon detection |
| Fire | `fire_n` | `yolov8n.pt` | Fire detection (fast) |
| Fire | `fire_s` | `yolov8s (1).pt` | Fire detection (accurate) |
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

## 📤 Deployment

### Prepare for Git Upload

1. **Clean the project**:
   ```bash
   python cleanup_tests.py --keep 0 --temp
   python project_structure.py --optimize
   ```

2. **Initialize Git repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: YOLO multi-model detection system"
   ```

3. **Add remote and push**:
   ```bash
   git remote add origin <your-repository-url>
   git push -u origin main
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
