# Installation Guide - YOLO Multi-Model Detection System

## 🚀 Quick Installation

### 1. System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 8GB RAM (16GB recommended)
- 10GB free disk space
- Windows 10/11, Linux, or macOS

**Recommended for GPU Acceleration:**
- NVIDIA GPU with CUDA support
- CUDA 11.8+ or 12.x
- 16GB+ RAM
- 20GB+ free disk space

### 2. Clone Repository

```bash
git clone https://github.com/Akhil4007-cpu/moderation-multi_modal.git
cd moderation-multi_modal
```

### 3. Install FFmpeg (Required for Video Processing)

FFmpeg is essential for video frame extraction. Install based on your system:

#### Windows
1. **Download FFmpeg:**
   - Go to https://ffmpeg.org/download.html
   - Download Windows build (static version recommended)
   - Extract to `C:\ffmpeg\`

2. **Add to PATH:**
   ```powershell
   # Add to system PATH
   $env:PATH += ";C:\ffmpeg\bin"
   
   # Or permanently add via System Properties > Environment Variables
   # Add C:\ffmpeg\bin to PATH variable
   ```

3. **Verify Installation:**
   ```powershell
   ffmpeg -version
   ```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
ffmpeg -version
```

#### macOS
```bash
# Using Homebrew
brew install ffmpeg
ffmpeg -version
```

### 4. Python Environment Setup

#### Option A: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using Conda
```bash
# Create conda environment
conda create -n yolo-detection python=3.9
conda activate yolo-detection

# Install dependencies
pip install -r requirements.txt
```

### 5. Install PyTorch

Choose the appropriate PyTorch installation for your system:

#### CPU Only (No GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### CUDA 11.8 (Older GPUs)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1 (Most Common)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CUDA 12.4 (Latest)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 6. Verify Installation

Run the verification script:

```bash
python -c "
import ultralytics, cv2, torch, ffmpeg
print('✅ ultralytics:', ultralytics.__version__)
print('✅ opencv-python:', cv2.__version__)
print('✅ torch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ FFmpeg available: True')
print('🎉 All dependencies installed successfully!')
"
```

Check FFmpeg separately:
```bash
ffmpeg -version
```

## 🎯 Quick Test

### Test Single Image Analysis
```bash
# Download a test image or use your own
python single_image_analyzer.py "path/to/test_image.jpg" --conf 0.5
```

### Test Video Processing (if you have a video)
```bash
python video_frame_processor.py "path/to/test_video.mp4" --conf 0.5
```

### Run Demo Script
```bash
# Creates test video and processes it
python demo_video_processor.py
```

## 🔧 Troubleshooting

### Common Issues

#### 1. FFmpeg Not Found
```
Error: ffmpeg not found in PATH
```
**Solution:** Ensure FFmpeg is installed and added to system PATH. Restart terminal after installation.

#### 2. CUDA Not Available
```
CUDA available: False
```
**Solution:** 
- Install NVIDIA GPU drivers
- Install appropriate CUDA toolkit
- Reinstall PyTorch with correct CUDA version

#### 3. Model Files Missing
```
Model not found: models/weapon/weights/weapon_detection.pt
```
**Solution:** Ensure all model files are present in the models directory structure.

#### 4. Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce batch size or image size
- Use CPU instead: `--device cpu`
- Close other GPU applications

#### 5. Permission Errors
```
PermissionError: [Errno 13] Permission denied
```
**Solution:**
- Run as administrator (Windows) or with sudo (Linux)
- Check file permissions
- Ensure write access to output directories

### Dependency Conflicts

If you encounter package conflicts:

```bash
# Create fresh environment
python -m venv fresh_env
# Windows:
fresh_env\Scripts\activate
# Linux/macOS:
source fresh_env/bin/activate

# Install step by step
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.3.58
pip install opencv-python==4.10.0.84
pip install -r requirements.txt
```

## 🚀 Advanced Setup

### GPU Optimization

For maximum performance with NVIDIA GPUs:

1. **Install CUDA Toolkit:**
   - Download from https://developer.nvidia.com/cuda-downloads
   - Choose version matching your PyTorch installation

2. **Install cuDNN:**
   - Download from https://developer.nvidia.com/cudnn
   - Follow NVIDIA installation guide

3. **Verify GPU Setup:**
   ```bash
   python -c "
   import torch
   print('CUDA Version:', torch.version.cuda)
   print('GPU Count:', torch.cuda.device_count())
   print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
   "
   ```

### Docker Setup (Optional)

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .
CMD ["python", "single_image_analyzer.py", "--help"]
```

Build and run:
```bash
docker build -t yolo-detection .
docker run -v $(pwd)/test_data:/app/test_data yolo-detection python single_image_analyzer.py test_data/image.jpg
```

## 📋 Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Git installed and configured
- [ ] Repository cloned
- [ ] FFmpeg installed and in PATH
- [ ] Virtual environment created and activated
- [ ] Dependencies installed from requirements.txt
- [ ] PyTorch installed with appropriate CUDA version
- [ ] Installation verified with test script
- [ ] Model files present in models directory
- [ ] Test run completed successfully

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs:** Most scripts provide detailed error messages
2. **Verify dependencies:** Run the verification script
3. **Check system requirements:** Ensure your system meets minimum requirements
4. **Update packages:** Try updating to latest versions
5. **Create issue:** Report bugs on the GitHub repository

## 📚 Next Steps

After successful installation:

1. Read `README.md` for feature overview
2. Check `TESTING_GUIDE.md` for usage examples
3. Review `VIDEO_PROCESSOR_GUIDE.md` for video processing
4. Start with single image analysis for testing
5. Progress to video processing for advanced use cases
