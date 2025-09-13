#!/usr/bin/env python3
"""
Project structure analyzer and optimizer for YOLO detection system.
Identifies unused files and optimizes the project for deployment.
"""

import os
import shutil
from pathlib import Path
import json

def analyze_project_structure():
    """Analyze the current project structure and identify optimization opportunities"""
    
    base_path = Path("d:/akhil")
    
    # Essential directories and files to keep
    essential_dirs = {
        "integrated_yolo_runner",
        "Accident_detection_model/Accident-Detection-Model", 
        "Fight_model/Fight-Violence-detection-yolov8",
        "Weapon-Detection-YOLO",
        "fire_detection_model",
        "NSFW_Detectio/nsfw_detector_annotator"
    }
    
    # Essential model files
    essential_models = {
        "Accident_detection_model/Accident-Detection-Model/yolov8s.pt",
        "Fight_model/Fight-Violence-detection-yolov8/Yolo_nano_weights.pt", 
        "Fight_model/Fight-Violence-detection-yolov8/yolo_small_weights.pt",
        "Weapon-Detection-YOLO/best (3).pt",
        "fire_detection_model/yolov8n.pt",
        "fire_detection_model/yolov8s (1).pt",
        "NSFW_Detectio/nsfw_detector_annotator/models/classification_model.pt",
        "NSFW_Detectio/nsfw_detector_annotator/models/segmentation_model.pt"
    }
    
    analysis = {
        "essential_files": [],
        "removable_dirs": [],
        "large_files": [],
        "duplicate_files": [],
        "total_size_mb": 0
    }
    
    print("Analyzing project structure...")
    
    # Check for large virtual environment directories that can be removed
    venv_dirs = list(base_path.glob("**/venv")) + list(base_path.glob("**/.venv"))
    for venv_dir in venv_dirs:
        if venv_dir.exists() and venv_dir.is_dir():
            try:
                size = sum(f.stat().st_size for f in venv_dir.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                if size_mb > 10:  # Only report large venv dirs
                    analysis["removable_dirs"].append({
                        "path": str(venv_dir.relative_to(base_path)),
                        "type": "virtual_environment",
                        "size_mb": round(size_mb, 2),
                        "reason": "Virtual environments should be recreated on deployment"
                    })
            except Exception as e:
                print(f"Error analyzing {venv_dir}: {e}")
    
    # Check for duplicate model files
    model_files = list(base_path.glob("**/*.pt"))
    model_names = {}
    for model_file in model_files:
        name = model_file.name
        if name in model_names:
            analysis["duplicate_files"].append({
                "original": str(model_names[name].relative_to(base_path)),
                "duplicate": str(model_file.relative_to(base_path))
            })
        else:
            model_names[name] = model_file
    
    # Check for large files that might not be needed
    for file_path in base_path.rglob("*"):
        if file_path.is_file():
            try:
                size = file_path.stat().st_size
                size_mb = size / (1024 * 1024)
                analysis["total_size_mb"] += size_mb
                
                if size_mb > 50:  # Files larger than 50MB
                    analysis["large_files"].append({
                        "path": str(file_path.relative_to(base_path)),
                        "size_mb": round(size_mb, 2),
                        "extension": file_path.suffix
                    })
            except Exception:
                pass
    
    # Mark essential files
    for essential in essential_models:
        essential_path = base_path / essential
        if essential_path.exists():
            analysis["essential_files"].append({
                "path": essential,
                "size_mb": round(essential_path.stat().st_size / (1024 * 1024), 2),
                "type": "model_weights"
            })
    
    return analysis

def optimize_project_for_deployment():
    """Remove unnecessary files and optimize project structure for deployment"""
    
    base_path = Path("d:/akhil")
    removed_items = []
    space_freed = 0
    
    print("Optimizing project for deployment...")
    
    # Remove virtual environment directories (they should be recreated on deployment)
    venv_patterns = ["**/venv", "**/.venv", "**/env", "**/.env"]
    for pattern in venv_patterns:
        for venv_dir in base_path.glob(pattern):
            if venv_dir.is_dir() and venv_dir.name in ["venv", ".venv", "env"]:
                try:
                    size = sum(f.stat().st_size for f in venv_dir.rglob('*') if f.is_file())
                    shutil.rmtree(venv_dir)
                    removed_items.append(f"Virtual environment: {venv_dir.relative_to(base_path)}")
                    space_freed += size
                except Exception as e:
                    print(f"Could not remove {venv_dir}: {e}")
    
    # Remove unnecessary directories that are not part of core functionality
    unnecessary_dirs = [
        "Object-detection-master",  # Seems to be a separate project
    ]
    
    for dir_name in unnecessary_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                shutil.rmtree(dir_path)
                removed_items.append(f"Unnecessary directory: {dir_name}")
                space_freed += size
            except Exception as e:
                print(f"Could not remove {dir_path}: {e}")
    
    space_freed_mb = space_freed / (1024 * 1024)
    
    print(f"\nOptimization completed:")
    print(f"Items removed: {len(removed_items)}")
    print(f"Space freed: {space_freed_mb:.2f} MB")
    
    for item in removed_items:
        print(f"  - {item}")
    
    return removed_items, space_freed_mb

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze and optimize project structure")
    parser.add_argument("--analyze", action="store_true", help="Only analyze, don't remove anything")
    parser.add_argument("--optimize", action="store_true", help="Optimize project for deployment")
    
    args = parser.parse_args()
    
    if args.analyze or not args.optimize:
        analysis = analyze_project_structure()
        
        print("\n" + "="*60)
        print("PROJECT STRUCTURE ANALYSIS")
        print("="*60)
        
        print(f"\nTotal project size: {analysis['total_size_mb']:.2f} MB")
        
        print(f"\nEssential model files ({len(analysis['essential_files'])}):")
        for file_info in analysis['essential_files']:
            print(f"  ✓ {file_info['path']} ({file_info['size_mb']} MB)")
        
        if analysis['removable_dirs']:
            print(f"\nRemovable directories ({len(analysis['removable_dirs'])}):")
            for dir_info in analysis['removable_dirs']:
                print(f"  🗑️  {dir_info['path']} ({dir_info['size_mb']} MB) - {dir_info['reason']}")
        
        if analysis['large_files']:
            print(f"\nLarge files ({len(analysis['large_files'])}):")
            for file_info in analysis['large_files']:
                print(f"  📁 {file_info['path']} ({file_info['size_mb']} MB)")
        
        if analysis['duplicate_files']:
            print(f"\nPotential duplicate files ({len(analysis['duplicate_files'])}):")
            for dup in analysis['duplicate_files']:
                print(f"  🔄 {dup['original']} ↔️ {dup['duplicate']}")
    
    if args.optimize:
        removed_items, space_freed = optimize_project_for_deployment()
