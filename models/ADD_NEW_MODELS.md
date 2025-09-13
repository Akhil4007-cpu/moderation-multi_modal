# Adding New Models to YOLO Detection System

## Quick Start for New Models

### 1. Create Model Directory Structure

```bash
mkdir models/your_model_name
mkdir models/your_model_name/weights
mkdir models/your_model_name/config  # if needed
```

### 2. Add Model Files

Place your model weights in the appropriate directory:
- `models/your_model_name/weights/model.pt` - Main model file
- `models/your_model_name/config/data.yaml` - Configuration (if needed)

### 3. Update Model Registry

Edit `models/model_registry.json` and add your model:

```json
{
  "your_model_name": {
    "category": "safety|security|content_moderation|custom",
    "description": "Brief description of what the model detects",
    "weights": {
      "default": "weights/model.pt"
    },
    "config": "config/data.yaml",  // optional
    "input_types": ["image", "video", "stream"],
    "confidence_threshold": 0.5
  }
}
```

### 4. Update Runner Configuration

Edit `integrated_yolo_runner/run.py` and add to MODEL_REGISTRY:

```python
MODEL_REGISTRY = {
    # ... existing models ...
    "your_model_name": WORKSPACE / "models/your_model_name/weights/model.pt",
}
```

### 5. Test Your Model

```bash
cd integrated_yolo_runner
python run.py --task your_model_name --source path/to/test/image.jpg
```

## Model Categories

- **safety**: Accident, fire, hazard detection
- **security**: Weapon, violence, intrusion detection  
- **content_moderation**: NSFW, inappropriate content
- **custom**: Domain-specific models

## File Naming Conventions

- Use lowercase with underscores: `fire_detection`, `weapon_scanner`
- Model files: `model_name.pt` or descriptive names like `yolov8s.pt`
- Config files: `data.yaml`, `classes.yaml`

## Best Practices

1. **Model Size**: Keep models under 100MB when possible
2. **Documentation**: Add model description and source
3. **Testing**: Always test with sample data before deployment
4. **Versioning**: Use subdirectories for model versions if needed
5. **Cleanup**: Remove old model versions regularly

## Example: Adding a "Traffic" Detection Model

```bash
# 1. Create structure
mkdir models/traffic
mkdir models/traffic/weights

# 2. Copy model file
cp your_traffic_model.pt models/traffic/weights/yolov8_traffic.pt

# 3. Update model_registry.json
{
  "traffic": {
    "category": "safety",
    "description": "Traffic violation and congestion detection",
    "weights": {
      "default": "weights/yolov8_traffic.pt"
    },
    "input_types": ["video", "stream"],
    "confidence_threshold": 0.6
  }
}

# 4. Update run.py MODEL_REGISTRY
"traffic": WORKSPACE / "models/traffic/weights/yolov8_traffic.pt",

# 5. Test
python run.py --task traffic --source traffic_video.mp4
```

## Troubleshooting

- **Model not found**: Check file paths in MODEL_REGISTRY
- **Low accuracy**: Adjust confidence threshold in model_registry.json
- **Memory issues**: Use smaller model variants or reduce image size
- **Performance**: Consider model quantization for deployment

For more details, see the main README.md file.
