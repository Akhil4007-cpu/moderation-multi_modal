#!/usr/bin/env python3
"""
Automated test cleanup script for YOLO detection system.
This script removes test result files to prevent storage confusion and maintains clean workspace.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def cleanup_test_results(base_path="d:/akhil/integrated_runs", keep_latest=1):
    """
    Clean up test result directories, keeping only the most recent results.
    
    Args:
        base_path: Base directory containing test results
        keep_latest: Number of latest test runs to keep per category
    """
    base_path = Path(base_path)
    if not base_path.exists():
        print(f"Base path {base_path} does not exist")
        return
    
    total_removed = 0
    total_size_freed = 0
    
    print(f"Starting cleanup of test results in {base_path}")
    print(f"Keeping {keep_latest} latest run(s) per category")
    print("-" * 50)
    
    # Process each category directory
    for category_dir in base_path.iterdir():
        if not category_dir.is_dir():
            continue
            
        print(f"\nProcessing category: {category_dir.name}")
        
        # Find all predict directories
        predict_dirs = []
        for item in category_dir.iterdir():
            if item.is_dir() and item.name.startswith('predict'):
                try:
                    # Get modification time for sorting
                    mtime = item.stat().st_mtime
                    predict_dirs.append((item, mtime))
                except Exception as e:
                    print(f"  Warning: Could not get stats for {item}: {e}")
        
        # Sort by modification time (newest first)
        predict_dirs.sort(key=lambda x: x[1], reverse=True)
        
        # Remove older directories
        dirs_to_remove = predict_dirs[keep_latest:]
        
        for dir_path, _ in dirs_to_remove:
            try:
                # Calculate size before removal
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                
                shutil.rmtree(dir_path)
                total_removed += 1
                total_size_freed += size
                
                print(f"  Removed: {dir_path.name} ({size_mb:.2f} MB)")
                
            except Exception as e:
                print(f"  Error removing {dir_path}: {e}")
    
    # Clean up any empty category directories
    for category_dir in base_path.iterdir():
        if category_dir.is_dir():
            try:
                if not any(category_dir.iterdir()):
                    category_dir.rmdir()
                    print(f"Removed empty directory: {category_dir.name}")
            except Exception as e:
                print(f"Could not remove empty directory {category_dir}: {e}")
    
    print("\n" + "=" * 50)
    print(f"Cleanup completed!")
    print(f"Directories removed: {total_removed}")
    print(f"Space freed: {total_size_freed / (1024 * 1024):.2f} MB")
    
    # Create cleanup log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "directories_removed": total_removed,
        "space_freed_mb": round(total_size_freed / (1024 * 1024), 2),
        "keep_latest": keep_latest
    }
    
    log_file = base_path / "cleanup_log.json"
    logs = []
    
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
    
    logs.append(log_entry)
    
    # Keep only last 10 log entries
    logs = logs[-10:]
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    return total_removed, total_size_freed

def cleanup_temp_files():
    """Remove temporary files and caches"""
    temp_patterns = [
        "**/__pycache__",
        "**/*.pyc", 
        "**/*.pyo",
        "**/.pytest_cache",
        "**/runs/detect/train*",  # Remove training artifacts
        "**/runs/detect/val*",    # Remove validation artifacts
    ]
    
    base_path = Path("d:/akhil")
    removed_count = 0
    
    print("\nCleaning temporary files...")
    
    for pattern in temp_patterns:
        for item in base_path.glob(pattern):
            try:
                if item.is_file():
                    item.unlink()
                    removed_count += 1
                    print(f"  Removed file: {item.relative_to(base_path)}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    removed_count += 1
                    print(f"  Removed directory: {item.relative_to(base_path)}")
            except Exception as e:
                print(f"  Error removing {item}: {e}")
    
    print(f"Removed {removed_count} temporary items")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up YOLO test results")
    parser.add_argument("--keep", type=int, default=1, help="Number of latest runs to keep per category")
    parser.add_argument("--path", default="d:/akhil/integrated_runs", help="Base path for test results")
    parser.add_argument("--temp", action="store_true", help="Also clean temporary files")
    
    args = parser.parse_args()
    
    # Clean test results
    cleanup_test_results(args.path, args.keep)
    
    # Clean temporary files if requested
    if args.temp:
        cleanup_temp_files()
    
    print("\nCleanup completed successfully!")
