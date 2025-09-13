#!/usr/bin/env python3
"""
Deployment preparation script for YOLO multi-model detection system.
Automates the process of preparing the project for Git upload and deployment.
"""

import os
import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime

def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_git_status():
    """Check if git is initialized and get status"""
    base_path = Path("d:/akhil")
    
    # Check if git is initialized
    git_dir = base_path / ".git"
    if not git_dir.exists():
        print("Git repository not initialized. Initializing...")
        success, stdout, stderr = run_command("git init", cwd=base_path)
        if not success:
            print(f"Failed to initialize git: {stderr}")
            return False
        print("Git repository initialized successfully.")
    
    # Check git status
    success, stdout, stderr = run_command("git status --porcelain", cwd=base_path)
    if success:
        changes = stdout.strip().split('\n') if stdout.strip() else []
        return len(changes)
    return 0

def prepare_for_deployment():
    """Prepare the project for deployment"""
    base_path = Path("d:/akhil")
    
    print("🚀 Preparing YOLO Detection System for Deployment")
    print("=" * 60)
    
    # Step 1: Clean test results and temporary files
    print("\n1. Cleaning test results and temporary files...")
    success, stdout, stderr = run_command("python cleanup_tests.py --keep 0 --temp", cwd=base_path)
    if success:
        print("✅ Cleanup completed successfully")
    else:
        print(f"⚠️  Cleanup had some issues: {stderr}")
    
    # Step 2: Optimize project structure
    print("\n2. Optimizing project structure...")
    success, stdout, stderr = run_command("python project_structure.py --optimize", cwd=base_path)
    if success:
        print("✅ Project optimization completed")
    else:
        print(f"⚠️  Optimization had some issues: {stderr}")
    
    # Step 3: Verify essential files exist
    print("\n3. Verifying essential files...")
    essential_files = [
        "integrated_yolo_runner/run.py",
        "integrated_yolo_runner/requirements.txt",
        "integrated_yolo_runner/README.md",
        "cleanup_tests.py",
        "project_structure.py",
        ".gitignore",
        "README.md"
    ]
    
    missing_files = []
    for file_path in essential_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing essential files: {missing_files}")
        return False
    
    # Step 4: Verify model files
    print("\n4. Verifying model files...")
    model_files = [
        "Accident_detection_model/Accident-Detection-Model/yolov8s.pt",
        "Fight_model/Fight-Violence-detection-yolov8/Yolo_nano_weights.pt",
        "Fight_model/Fight-Violence-detection-yolov8/yolo_small_weights.pt",
        "Weapon-Detection-YOLO/best (3).pt",
        "fire_detection_model/yolov8n.pt",
        "fire_detection_model/yolov8s (1).pt",
        "NSFW_Detectio/nsfw_detector_annotator/models/classification_model.pt",
        "NSFW_Detectio/nsfw_detector_annotator/models/segmentation_model.pt"
    ]
    
    missing_models = []
    total_model_size = 0
    for model_path in model_files:
        full_path = base_path / model_path
        if not full_path.exists():
            missing_models.append(model_path)
        else:
            size_mb = full_path.stat().st_size / (1024 * 1024)
            total_model_size += size_mb
            print(f"✅ {model_path} ({size_mb:.1f} MB)")
    
    if missing_models:
        print(f"❌ Missing model files: {missing_models}")
        print("⚠️  Some models are missing. The system may not work correctly.")
    
    print(f"\n📊 Total model size: {total_model_size:.1f} MB")
    
    # Step 5: Check project size
    print("\n5. Calculating project size...")
    success, stdout, stderr = run_command('Get-ChildItem -Recurse | Measure-Object -Property Length -Sum | Select-Object @{Name="Size(MB)";Expression={[math]::Round($_.Sum/1MB,2)}}', cwd=base_path)
    if success and "Size(MB)" in stdout:
        print(f"📁 Current project size: {stdout.strip()}")
    
    # Step 6: Git preparation
    print("\n6. Preparing Git repository...")
    changes_count = check_git_status()
    print(f"📝 Files to be committed: {changes_count}")
    
    # Create deployment summary
    deployment_info = {
        "timestamp": datetime.now().isoformat(),
        "essential_files_present": len(missing_files) == 0,
        "model_files_present": len(missing_models) == 0,
        "total_model_size_mb": round(total_model_size, 1),
        "missing_files": missing_files,
        "missing_models": missing_models,
        "git_changes": changes_count,
        "ready_for_deployment": len(missing_files) == 0
    }
    
    with open(base_path / "deployment_status.json", "w") as f:
        json.dump(deployment_info, f, indent=2)
    
    print("\n" + "=" * 60)
    if deployment_info["ready_for_deployment"]:
        print("🎉 Project is ready for deployment!")
        print("\nNext steps:")
        print("1. git add .")
        print("2. git commit -m 'Prepare for deployment: YOLO multi-model detection system'")
        print("3. git remote add origin <your-repository-url>")
        print("4. git push -u origin main")
    else:
        print("❌ Project is NOT ready for deployment.")
        print("Please resolve the missing files before proceeding.")
    
    return deployment_info["ready_for_deployment"]

def git_commit_and_prepare():
    """Add files to git and prepare for upload"""
    base_path = Path("d:/akhil")
    
    print("\n7. Adding files to Git...")
    
    # Add all files
    success, stdout, stderr = run_command("git add .", cwd=base_path)
    if not success:
        print(f"❌ Failed to add files: {stderr}")
        return False
    
    # Check what will be committed
    success, stdout, stderr = run_command("git status --porcelain", cwd=base_path)
    if success:
        files_to_commit = stdout.strip().split('\n') if stdout.strip() else []
        print(f"📝 Files staged for commit: {len(files_to_commit)}")
        
        # Show first few files
        for i, file_line in enumerate(files_to_commit[:10]):
            print(f"   {file_line}")
        if len(files_to_commit) > 10:
            print(f"   ... and {len(files_to_commit) - 10} more files")
    
    # Commit
    commit_message = f"Prepare for deployment: YOLO multi-model detection system - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    success, stdout, stderr = run_command(f'git commit -m "{commit_message}"', cwd=base_path)
    
    if success:
        print("✅ Files committed successfully!")
        print(f"📝 Commit message: {commit_message}")
        
        print("\n🚀 Ready for Git upload!")
        print("To upload to a remote repository:")
        print("1. git remote add origin <your-repository-url>")
        print("2. git branch -M main")
        print("3. git push -u origin main")
        
        return True
    else:
        if "nothing to commit" in stderr:
            print("ℹ️  No changes to commit (already up to date)")
            return True
        else:
            print(f"❌ Failed to commit: {stderr}")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare YOLO detection system for deployment")
    parser.add_argument("--commit", action="store_true", help="Also commit changes to git")
    
    args = parser.parse_args()
    
    # Prepare for deployment
    ready = prepare_for_deployment()
    
    # Commit if requested and ready
    if args.commit and ready:
        git_commit_and_prepare()
    elif args.commit and not ready:
        print("\n❌ Cannot commit: Project is not ready for deployment")
        sys.exit(1)
