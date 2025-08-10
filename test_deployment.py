#!/usr/bin/env python3
"""
Test script to verify deployment readiness
This script checks if all required dependencies and modules can be imported
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    # Test basic Python packages
    basic_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'sklearn',
        'PIL'
    ]
    
    for package in basic_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ tensorflow {tf.__version__}")
    except ImportError as e:
        print(f"❌ tensorflow: {e}")
    
    # Test Streamlit
    try:
        import streamlit as st
        print(f"✅ streamlit {st.__version__}")
    except ImportError as e:
        print(f"❌ streamlit: {e}")
    
    # Test streamlit-drawable-canvas
    try:
        from streamlit_drawable_canvas import st_canvas
        print("✅ streamlit-drawable-canvas")
    except ImportError as e:
        print(f"❌ streamlit-drawable-canvas: {e}")
    
    # Test custom module
    try:
        from improved_digit_recognizer import ImprovedDigitRecognizer
        print("✅ improved_digit_recognizer")
    except ImportError as e:
        print(f"❌ improved_digit_recognizer: {e}")

def test_files():
    """Test if all required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'streamlit_app.py',
        'improved_digit_recognizer.py',
        'requirements.txt',
        '.streamlit/config.toml'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")

def test_model():
    """Test if model can be loaded"""
    print("\n🤖 Testing model loading...")
    
    try:
        from improved_digit_recognizer import ImprovedDigitRecognizer
        
        recognizer = ImprovedDigitRecognizer()
        print("✅ ImprovedDigitRecognizer instantiated")
        
        # Check if model file exists
        if os.path.exists('improved_digit_model.h5'):
            print("✅ Model file exists")
            try:
                recognizer.load_model('improved_digit_model.h5')
                print("✅ Model loaded successfully")
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
        else:
            print("⚠️ Model file not found (will be created during first run)")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")

def main():
    """Main test function"""
    print("🚀 Testing deployment readiness...\n")
    
    test_imports()
    test_files()
    test_model()
    
    print("\n" + "="*50)
    print("🎯 Deployment Test Summary")
    print("="*50)
    print("If you see any ❌ errors above, fix them before deploying.")
    print("If all tests pass (✅), your app should deploy successfully!")
    print("\nTo deploy:")
    print("1. Push to GitHub")
    print("2. Connect to Streamlit Cloud")
    print("3. Deploy automatically")

if __name__ == "__main__":
    main()
