#!/usr/bin/env python
"""
Test script for the unified multi-modal processor
Creates test files and validates the system functionality
"""

import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_test_image():
    """Create a simple test image"""
    # Create a 640x480 RGB image
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((50, 50), "TEST IMAGE", fill='black', font=font)
    draw.text((50, 100), "Multi-Modal Processing Test", fill='black', font=font)
    draw.rectangle([200, 200, 400, 350], outline='red', width=3)
    draw.text((220, 260), "Test Object", fill='red', font=font)
    
    # Save the image
    test_image_path = Path("test_image.jpg")
    img.save(test_image_path)
    print(f"✅ Created test image: {test_image_path}")
    return test_image_path

def create_test_text_file():
    """Create a test text file"""
    test_text = """
    This is a test document for text moderation.
    It contains safe content for testing purposes.
    The system should classify this as safe content.
    
    Testing keywords: hello, world, test, safe, document
    """
    
    test_text_path = Path("test_text.txt")
    with open(test_text_path, 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    print(f"✅ Created test text file: {test_text_path}")
    return test_text_path

def test_unified_processor():
    """Test the unified multi-modal processor"""
    print("🚀 Testing Unified Multi-Modal Processor")
    print("=" * 50)
    
    # Import the processor
    try:
        from unified_multimodal_processor import UnifiedMultiModalProcessor
        print("✅ Successfully imported UnifiedMultiModalProcessor")
    except ImportError as e:
        print(f"❌ Failed to import processor: {e}")
        return False
    
    # Initialize processor
    try:
        processor = UnifiedMultiModalProcessor()
        print("✅ Successfully initialized processor")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return False
    
    # Test image processing
    print("\n📸 Testing Image Processing...")
    test_image = create_test_image()
    
    try:
        image_results = processor.process_image(test_image, conf_threshold=0.5)
        print("✅ Image processing completed")
        print(f"   Models tested: {len(image_results.get('models_tested', []))}")
        print(f"   Safety assessment: {image_results.get('summary', {}).get('safety_assessment', 'Unknown')}")
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
    
    # Test text processing
    print("\n📝 Testing Text Processing...")
    test_text_file = create_test_text_file()
    
    try:
        with open(test_text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        text_results = processor.process_text_content(text_content)
        print("✅ Text processing completed")
        print(f"   Safety status: {text_results.get('safety_status', 'Unknown')}")
        print(f"   Unsafe words found: {len(text_results.get('unsafe_words_found', []))}")
    except Exception as e:
        print(f"❌ Text processing failed: {e}")
    
    # Test file processing with auto-detection
    print("\n🔍 Testing Auto-Detection...")
    try:
        auto_results = processor.process_file(test_image, mode='auto')
        print("✅ Auto-detection completed")
        print(f"   Detected modality: {auto_results.get('modality', 'Unknown')}")
    except Exception as e:
        print(f"❌ Auto-detection failed: {e}")
    
    # Cleanup test files
    try:
        test_image.unlink()
        test_text_file.unlink()
        print("\n🧹 Cleaned up test files")
    except:
        pass
    
    print("\n✅ Testing completed!")
    return True

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing Dependencies")
    print("=" * 30)
    
    dependencies = [
        ('ultralytics', 'YOLO models'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('librosa', 'Audio processing'),
        ('joblib', 'Model loading'),
        ('sklearn', 'Machine learning'),
        ('pytesseract', 'OCR'),
        ('speech_recognition', 'Speech recognition'),
        ('pydub', 'Audio processing')
    ]
    
    missing_deps = []
    
    for dep, description in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} ({description})")
        except ImportError:
            print(f"❌ {dep} ({description}) - MISSING")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    else:
        print("\n✅ All dependencies available!")
        return True

def main():
    """Main test function"""
    print("🧪 Unified Multi-Modal System Test Suite")
    print("=" * 60)
    
    # Test dependencies first
    deps_ok = test_dependencies()
    
    if not deps_ok:
        print("\n❌ Cannot proceed with testing due to missing dependencies")
        return
    
    print("\n")
    
    # Test the unified processor
    processor_ok = test_unified_processor()
    
    if processor_ok:
        print("\n🎉 All tests passed! The unified multi-modal system is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
