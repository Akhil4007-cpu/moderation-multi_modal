#!/usr/bin/env python
"""
Setup script for YOLO Multi-Model Detection System
Automated installation and dependency management
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found. Please install FFmpeg and add to PATH")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        print("   Linux: sudo apt install ffmpeg")
        print("   macOS: brew install ffmpeg")
        return False

def install_pytorch():
    """Install PyTorch based on system capabilities"""
    print("🔄 Detecting system capabilities for PyTorch installation...")
    
    try:
        import torch
        print(f"✅ PyTorch already installed: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError:
        pass
    
    # Try to detect CUDA
    cuda_available = False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        cuda_available = True
        print("✅ NVIDIA GPU detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ℹ️  No NVIDIA GPU detected, installing CPU version")
    
    if cuda_available:
        # Install CUDA version
        pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        description = "Installing PyTorch with CUDA 12.1 support"
    else:
        # Install CPU version
        pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        description = "Installing PyTorch (CPU version)"
    
    return run_command(pytorch_cmd, description)

def main():
    """Main setup function"""
    print("🚀 YOLO Multi-Model Detection System Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_ffmpeg():
        print("⚠️  FFmpeg is required for video processing. Please install it first.")
        response = input("Continue without FFmpeg? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Upgrade pip
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Failed to upgrade pip, continuing anyway...")
    
    # Install PyTorch first
    if not install_pytorch():
        print("❌ Failed to install PyTorch")
        sys.exit(1)
    
    # Install other requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Verify installation
    print("\n🔍 Verifying installation...")
    verification_script = """
import ultralytics, cv2, torch
print('✅ ultralytics:', ultralytics.__version__)
print('✅ opencv-python:', cv2.__version__)
print('✅ torch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
print('🎉 All core dependencies verified!')
"""
    
    if run_command(f'python -c "{verification_script}"', "Verifying core dependencies"):
        print("\n🎉 Setup completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Read README.md for feature overview")
        print("   2. Check TESTING_GUIDE.md for usage examples")
        print("   3. Run: python single_image_analyzer.py --help")
        print("   4. Run: python video_frame_processor.py --help")
    else:
        print("❌ Installation verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
