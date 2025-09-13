#!/usr/bin/env python
"""
Demo script to test the video frame processor with a sample video
Creates a simple test video if none exists and runs the processor
"""

import os
import sys
import subprocess
from pathlib import Path
import tempfile

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed and working")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ FFmpeg not found!")
    print("Please install FFmpeg:")
    print("1. Download from: https://ffmpeg.org/download.html")
    print("2. Add to system PATH")
    print("3. Or install via: choco install ffmpeg")
    return False

def create_test_video(output_path):
    """Create a simple test video using FFmpeg"""
    try:
        print(f"🎬 Creating test video: {output_path}")
        
        # Create a 10-second test video with moving colored rectangles
        cmd = [
            'ffmpeg', '-f', 'lavfi', '-i', 
            'testsrc=duration=10:size=640x480:rate=30',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-y', str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Test video created successfully")
            return True
        else:
            print(f"❌ Failed to create test video: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating test video: {e}")
        return False

def run_demo():
    """Run the video processor demo"""
    
    print("🚀 Video Frame Processor Demo")
    print("=" * 50)
    
    # Check FFmpeg
    if not check_ffmpeg():
        return False
    
    # Create test video
    test_video = Path("test_video.mp4")
    if not test_video.exists():
        if not create_test_video(test_video):
            return False
    else:
        print(f"✅ Using existing test video: {test_video}")
    
    # Run the video processor
    print(f"\n🔍 Running video frame processor...")
    
    try:
        cmd = [
            sys.executable, 'video_frame_processor.py', 
            str(test_video),
            '--conf', '0.3',
            '--output', './demo_results'
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print("\n" + "=" * 50)
        
        # Run with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ Demo completed successfully!")
            print(f"📁 Check results in: ./demo_results/")
            return True
        else:
            print(f"\n❌ Demo failed with exit code: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False
    
    finally:
        # Cleanup test video
        if test_video.exists():
            try:
                test_video.unlink()
                print(f"🧹 Cleaned up test video")
            except:
                pass

if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
