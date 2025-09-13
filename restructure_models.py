#!/usr/bin/env python3
"""
Model structure reorganization script for YOLO detection system.
Creates a clean, scalable structure for current and future models.
"""

import os
import shutil
from pathlib import Path
import json

def analyze_model_folders():
    """Analyze each model folder and identify what to keep/remove"""
    
    base_path = Path("d:/akhil")
    analysis = {
        "accident": {
            "current_path": "Accident_detection_model/Accident-Detection-Model",
            "essential_files": ["yolov8s.pt", "data.yaml"],
            "removable_files": ["Accident detection demo.mp4", "yolo.ipynb"],
            "removable_dirs": [".git", "data", "runs"]
        },
        "fight": {
            "current_path": "Fight_model/Fight-Violence-detection-yolov8", 
            "essential_files": ["Yolo_nano_weights.pt", "yolo_small_weights.pt"],
            "removable_files": ["test_detect.py", "testing_code.ipynb"],
            "removable_dirs": [".git"]
        },
        "weapon": {
            "current_path": "Weapon-Detection-YOLO",
            "essential_files": ["best (3).pt"],
            "removable_files": ["SECURITY.md", "evaluate.py", "weapon-detection.ipynb"],
            "removable_dirs": [".git"]
        },
        "fire": {
            "current_path": "fire_detection_model",
            "essential_files": ["yolov8n.pt", "yolov8s (1).pt"],
            "removable_files": [],
            "removable_dirs": []
        },
        "nsfw": {
            "current_path": "NSFW_Detectio/nsfw_detector_annotator",
            "essential_files": ["models/classification_model.pt", "models/segmentation_model.pt"],
            "removable_files": [".gitignore", "LICENSE", "packages.txt", "requirements.txt"],
            "removable_dirs": [".devcontainer", ".git", "app", "assets", "out", "scripts"]
        }
    }
    
    return analysis

def create_new_structure():
    """Create new standardized model structure"""
    
    base_path = Path("d:/akhil")
    models_dir = base_path / "models"
    
    # Create main models directory
    models_dir.mkdir(exist_ok=True)
    
    # Define new structure
    new_structure = {
        "accident": {
            "weights": ["yolov8s.pt"],
            "config": ["data.yaml"]
        },
        "fight": {
            "weights": ["nano_weights.pt", "small_weights.pt"],
            "config": []
        },
        "weapon": {
            "weights": ["weapon_detection.pt"],
            "config": []
        },
        "fire": {
            "weights": ["yolov8n.pt", "yolov8s.pt"],
            "config": []
        },
        "nsfw": {
            "weights": ["classification_model.pt", "segmentation_model.pt"],
            "config": []
        }
    }
    
    print("📁 Creating new standardized model structure...")
    
    for category, structure in new_structure.items():
        category_dir = models_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Create weights subdirectory
        weights_dir = category_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        
        # Create config subdirectory if needed
        if structure["config"]:
            config_dir = category_dir / "config"
            config_dir.mkdir(exist_ok=True)
        
        print(f"  ✅ Created: models/{category}/")
    
    return new_structure

def move_essential_files():
    """Move essential files to new structure and remove unnecessary ones"""
    
    base_path = Path("d:/akhil")
    models_dir = base_path / "models"
    analysis = analyze_model_folders()
    
    moved_files = []
    removed_items = []
    space_freed = 0
    
    print("\n🚚 Moving essential files and removing unnecessary ones...")
    
    # Mapping of old files to new locations
    file_mappings = {
        # Accident model
        "Accident_detection_model/Accident-Detection-Model/yolov8s.pt": "models/accident/weights/yolov8s.pt",
        "Accident_detection_model/Accident-Detection-Model/data.yaml": "models/accident/config/data.yaml",
        
        # Fight models
        "Fight_model/Fight-Violence-detection-yolov8/Yolo_nano_weights.pt": "models/fight/weights/nano_weights.pt",
        "Fight_model/Fight-Violence-detection-yolov8/yolo_small_weights.pt": "models/fight/weights/small_weights.pt",
        
        # Weapon model
        "Weapon-Detection-YOLO/best (3).pt": "models/weapon/weights/weapon_detection.pt",
        
        # Fire models
        "fire_detection_model/yolov8n.pt": "models/fire/weights/yolov8n.pt",
        "fire_detection_model/yolov8s (1).pt": "models/fire/weights/yolov8s.pt",
        
        # NSFW models
        "NSFW_Detectio/nsfw_detector_annotator/models/classification_model.pt": "models/nsfw/weights/classification_model.pt",
        "NSFW_Detectio/nsfw_detector_annotator/models/segmentation_model.pt": "models/nsfw/weights/segmentation_model.pt"
    }
    
    # Move essential files
    for old_path, new_path in file_mappings.items():
        old_file = base_path / old_path
        new_file = base_path / new_path
        
        if old_file.exists():
            try:
                # Create parent directories
                new_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file (don't move yet, in case something goes wrong)
                shutil.copy2(old_file, new_file)
                moved_files.append(f"{old_path} → {new_path}")
                print(f"  ✅ Moved: {old_path} → {new_path}")
                
            except Exception as e:
                print(f"  ❌ Failed to move {old_path}: {e}")
        else:
            print(f"  ⚠️  File not found: {old_path}")
    
    # Remove unnecessary files and directories
    for category, info in analysis.items():
        category_path = base_path / info["current_path"]
        
        # Remove unnecessary files
        for file_name in info["removable_files"]:
            file_path = category_path / file_name
            if file_path.exists():
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    removed_items.append(f"File: {info['current_path']}/{file_name}")
                    space_freed += size
                    print(f"  🗑️  Removed file: {info['current_path']}/{file_name}")
                except Exception as e:
                    print(f"  ❌ Could not remove {file_path}: {e}")
        
        # Remove unnecessary directories
        for dir_name in info["removable_dirs"]:
            dir_path = category_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                try:
                    size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    shutil.rmtree(dir_path)
                    removed_items.append(f"Directory: {info['current_path']}/{dir_name}")
                    space_freed += size
                    print(f"  🗑️  Removed directory: {info['current_path']}/{dir_name}")
                except Exception as e:
                    print(f"  ❌ Could not remove {dir_path}: {e}")
    
    return moved_files, removed_items, space_freed

def create_model_registry_config():
    """Create a configuration file for the new model structure"""
    
    base_path = Path("d:/akhil")
    
    model_config = {
        "version": "2.0",
        "base_path": "d:/akhil/models",
        "models": {
            "accident": {
                "category": "safety",
                "description": "Vehicle accident detection",
                "weights": {
                    "default": "weights/yolov8s.pt"
                },
                "config": "config/data.yaml",
                "input_types": ["image", "video", "stream"],
                "confidence_threshold": 0.25
            },
            "fight": {
                "category": "security", 
                "description": "Violence and fight detection",
                "weights": {
                    "nano": "weights/nano_weights.pt",
                    "small": "weights/small_weights.pt",
                    "default": "weights/small_weights.pt"
                },
                "input_types": ["image", "video", "stream"],
                "confidence_threshold": 0.5
            },
            "weapon": {
                "category": "security",
                "description": "Weapon detection",
                "weights": {
                    "default": "weights/weapon_detection.pt"
                },
                "input_types": ["image", "video", "stream"],
                "confidence_threshold": 0.5
            },
            "fire": {
                "category": "safety",
                "description": "Fire and smoke detection", 
                "weights": {
                    "nano": "weights/yolov8n.pt",
                    "small": "weights/yolov8s.pt",
                    "default": "weights/yolov8s.pt"
                },
                "input_types": ["image", "video", "stream"],
                "confidence_threshold": 0.5
            },
            "nsfw": {
                "category": "content_moderation",
                "description": "NSFW content detection",
                "weights": {
                    "classification": "weights/classification_model.pt",
                    "segmentation": "weights/segmentation_model.pt", 
                    "default": "weights/segmentation_model.pt"
                },
                "input_types": ["image"],
                "confidence_threshold": 0.7
            }
        }
    }
    
    config_file = base_path / "models" / "model_registry.json"
    with open(config_file, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"📝 Created model registry: {config_file}")
    return model_config

def cleanup_old_structure():
    """Remove old model directories after successful migration"""
    
    base_path = Path("d:/akhil")
    
    old_dirs = [
        "Accident_detection_model",
        "Fight_model", 
        "Weapon-Detection-YOLO",
        "fire_detection_model",
        "NSFW_Detectio"
    ]
    
    print("\n🧹 Cleaning up old directory structure...")
    
    removed_dirs = []
    space_freed = 0
    
    for old_dir in old_dirs:
        dir_path = base_path / old_dir
        if dir_path.exists() and dir_path.is_dir():
            try:
                # Calculate size before removal
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                
                # Check if directory is mostly empty (only has essential files we already moved)
                remaining_files = list(dir_path.rglob('*'))
                essential_count = sum(1 for f in remaining_files if f.suffix == '.pt')
                
                if essential_count <= 2:  # Only a few model files left
                    shutil.rmtree(dir_path)
                    removed_dirs.append(old_dir)
                    space_freed += size
                    print(f"  ✅ Removed: {old_dir}")
                else:
                    print(f"  ⚠️  Skipped: {old_dir} (contains {essential_count} model files)")
                    
            except Exception as e:
                print(f"  ❌ Could not remove {old_dir}: {e}")
    
    return removed_dirs, space_freed

def update_runner_for_new_structure():
    """Update the integrated runner to use new model structure"""
    
    base_path = Path("d:/akhil")
    run_py_path = base_path / "integrated_yolo_runner" / "run.py"
    
    print("\n🔧 Updating runner for new model structure...")
    
    # Read current file
    with open(run_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the MODEL_REGISTRY section
    old_registry_start = "MODEL_REGISTRY = {"
    old_registry_end = "}"
    
    new_registry = '''MODEL_REGISTRY = {
    # Accident detection
    "accident": WORKSPACE / "models/accident/weights/yolov8s.pt",
    
    # Fight detection
    "fight_nano": WORKSPACE / "models/fight/weights/nano_weights.pt",
    "fight_small": WORKSPACE / "models/fight/weights/small_weights.pt",
    
    # Weapon detection
    "weapon": WORKSPACE / "models/weapon/weights/weapon_detection.pt",
    
    # Fire detection
    "fire_n": WORKSPACE / "models/fire/weights/yolov8n.pt",
    "fire_s": WORKSPACE / "models/fire/weights/yolov8s.pt",
    
    # NSFW detection
    "nsfw_cls": WORKSPACE / "models/nsfw/weights/classification_model.pt",
    "nsfw_seg": WORKSPACE / "models/nsfw/weights/segmentation_model.pt",
    
    # Generic object detection (auto-download)
    "objects": Path("yolov8n.pt"),
}'''
    
    # Find and replace the registry
    start_idx = content.find(old_registry_start)
    if start_idx != -1:
        # Find the end of the registry (look for the closing brace)
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(content[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        # Replace the registry
        updated_content = content[:start_idx] + new_registry + content[end_idx:]
        
        # Write back
        with open(run_py_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print("  ✅ Updated MODEL_REGISTRY in run.py")
    else:
        print("  ⚠️  Could not find MODEL_REGISTRY in run.py")

def restructure_project():
    """Main function to restructure the entire project"""
    
    print("🏗️  RESTRUCTURING YOLO DETECTION PROJECT")
    print("=" * 60)
    
    # Step 1: Create new structure
    new_structure = create_new_structure()
    
    # Step 2: Move files and clean up
    moved_files, removed_items, space_freed = move_essential_files()
    
    # Step 3: Create model registry
    model_config = create_model_registry_config()
    
    # Step 4: Update runner
    update_runner_for_new_structure()
    
    # Step 5: Clean up old structure (commented out for safety)
    # removed_dirs, old_space_freed = cleanup_old_structure()
    
    space_freed_mb = space_freed / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("🎉 PROJECT RESTRUCTURING COMPLETED!")
    print(f"📁 Files moved: {len(moved_files)}")
    print(f"🗑️  Items removed: {len(removed_items)}")
    print(f"💾 Space freed: {space_freed_mb:.2f} MB")
    
    print(f"\n📊 New structure created:")
    print(f"  📂 models/")
    for category in model_config["models"].keys():
        print(f"    📂 {category}/")
        print(f"      📂 weights/")
        if category == "accident":
            print(f"      📂 config/")
    
    # Create restructure report
    report = {
        "timestamp": "2025-09-13T09:15:48+05:30",
        "moved_files": moved_files,
        "removed_items": removed_items,
        "space_freed_mb": round(space_freed_mb, 2),
        "new_structure": model_config
    }
    
    with open(Path("d:/akhil") / "restructure_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📝 Restructure report saved to: restructure_report.json")
    
    return report

if __name__ == "__main__":
    restructure_project()
