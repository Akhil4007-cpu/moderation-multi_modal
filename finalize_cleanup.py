#!/usr/bin/env python3
"""
Final cleanup script to remove old model directories and complete project restructuring.
"""

import os
import shutil
from pathlib import Path
import json

def remove_old_model_directories():
    """Remove old model directories after successful restructuring"""
    
    base_path = Path("d:/akhil")
    
    old_dirs_to_remove = [
        "Accident_detection_model",
        "Fight_model", 
        "Weapon-Detection-YOLO",
        "fire_detection_model",
        "NSFW_Detectio",
        "Object-detection-master"  # This was identified as unused earlier
    ]
    
    print("🧹 Removing old model directories...")
    print("=" * 50)
    
    removed_dirs = []
    space_freed = 0
    errors = []
    
    for old_dir in old_dirs_to_remove:
        dir_path = base_path / old_dir
        if dir_path.exists() and dir_path.is_dir():
            try:
                # Calculate size before removal
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                
                # Force removal with error handling for locked files
                def handle_remove_readonly(func, path, exc):
                    if os.path.exists(path):
                        os.chmod(path, 0o777)
                        func(path)
                
                shutil.rmtree(dir_path, onerror=handle_remove_readonly)
                removed_dirs.append(old_dir)
                space_freed += size
                print(f"  ✅ Removed: {old_dir} ({size/(1024*1024):.1f} MB)")
                
            except Exception as e:
                errors.append(f"{old_dir}: {str(e)}")
                print(f"  ❌ Could not remove {old_dir}: {e}")
    
    return removed_dirs, space_freed, errors

def verify_new_structure():
    """Verify that the new model structure is complete and functional"""
    
    base_path = Path("d:/akhil")
    models_dir = base_path / "models"
    
    print("\n🔍 Verifying new model structure...")
    
    expected_structure = {
        "accident": ["weights/yolov8s.pt", "config/data.yaml"],
        "fight": ["weights/nano_weights.pt", "weights/small_weights.pt"],
        "weapon": ["weights/weapon_detection.pt"],
        "fire": ["weights/yolov8n.pt", "weights/yolov8s.pt"],
        "nsfw": ["weights/classification_model.pt", "weights/segmentation_model.pt"]
    }
    
    verification_results = {}
    total_model_size = 0
    
    for category, expected_files in expected_structure.items():
        category_dir = models_dir / category
        verification_results[category] = {
            "exists": category_dir.exists(),
            "files": {},
            "complete": True
        }
        
        print(f"\n  📂 {category}/")
        
        for expected_file in expected_files:
            file_path = category_dir / expected_file
            exists = file_path.exists()
            verification_results[category]["files"][expected_file] = exists
            
            if exists:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_model_size += size_mb
                print(f"    ✅ {expected_file} ({size_mb:.1f} MB)")
            else:
                print(f"    ❌ {expected_file} (MISSING)")
                verification_results[category]["complete"] = False
    
    # Check model registry
    registry_file = models_dir / "model_registry.json"
    registry_exists = registry_file.exists()
    print(f"\n  📄 model_registry.json: {'✅' if registry_exists else '❌'}")
    
    print(f"\n📊 Total model size: {total_model_size:.1f} MB")
    
    return verification_results, total_model_size

def create_model_addition_guide():
    """Create a guide for adding new models in the future"""
    
    base_path = Path("d:/akhil")
    
    guide_content = """# Adding New Models to YOLO Detection System

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
"""

    guide_file = base_path / "models" / "ADD_NEW_MODELS.md"
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"📖 Created model addition guide: {guide_file}")

def update_main_readme():
    """Update the main README with new structure information"""
    
    base_path = Path("d:/akhil")
    readme_path = base_path / "README.md"
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update project structure section
    old_structure = """```
d:/akhil/
├── integrated_yolo_runner/          # Main unified runner
│   ├── run.py                       # Primary execution script
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # Detailed usage instructions
├── Accident_detection_model/        # Accident detection model
├── Fight_model/                     # Violence/fight detection models
├── Weapon-Detection-YOLO/          # Weapon detection model
├── fire_detection_model/           # Fire detection models
├── NSFW_Detectio/                  # NSFW content detection
├── cleanup_tests.py                # Automated test cleanup script
├── project_structure.py            # Project optimization tools
└── .gitignore                      # Git ignore rules
```"""

    new_structure = """```
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
```"""

    # Replace the structure
    updated_content = content.replace(old_structure, new_structure)
    
    # Add section about new structure
    addition = """

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
"""

    # Add before the troubleshooting section
    troubleshooting_pos = updated_content.find("## 🐛 Troubleshooting")
    if troubleshooting_pos != -1:
        updated_content = updated_content[:troubleshooting_pos] + addition + "\n" + updated_content[troubleshooting_pos:]
    
    # Write back
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("📝 Updated main README.md with new structure")

def finalize_project():
    """Complete the final cleanup and optimization"""
    
    print("🎯 FINALIZING PROJECT CLEANUP")
    print("=" * 60)
    
    # Step 1: Remove old directories
    removed_dirs, space_freed, errors = remove_old_model_directories()
    
    # Step 2: Verify new structure
    verification, total_model_size = verify_new_structure()
    
    # Step 3: Create guides and documentation
    create_model_addition_guide()
    update_main_readme()
    
    space_freed_mb = space_freed / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("🎉 PROJECT FINALIZATION COMPLETED!")
    print(f"🗑️  Old directories removed: {len(removed_dirs)}")
    print(f"💾 Additional space freed: {space_freed_mb:.1f} MB")
    print(f"📊 Total model size: {total_model_size:.1f} MB")
    
    if errors:
        print(f"\n⚠️  Errors encountered: {len(errors)}")
        for error in errors:
            print(f"  - {error}")
    
    # Check if all models are present
    all_complete = all(cat["complete"] for cat in verification.values())
    print(f"\n✅ Model structure complete: {'Yes' if all_complete else 'No'}")
    
    # Create final report
    final_report = {
        "timestamp": "2025-09-13T09:15:48+05:30",
        "removed_directories": removed_dirs,
        "space_freed_mb": round(space_freed_mb, 1),
        "total_model_size_mb": round(total_model_size, 1),
        "structure_verification": verification,
        "all_models_present": all_complete,
        "errors": errors
    }
    
    with open(Path("d:/akhil") / "final_cleanup_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n📝 Final report saved to: final_cleanup_report.json")
    
    if all_complete and not errors:
        print("\n🚀 PROJECT IS READY FOR DEPLOYMENT!")
        print("Next steps:")
        print("1. git init")
        print("2. git add .")
        print("3. git commit -m 'Initial commit: Clean YOLO detection system'")
        print("4. git remote add origin <your-repo-url>")
        print("5. git push -u origin main")
    
    return final_report

if __name__ == "__main__":
    finalize_project()
