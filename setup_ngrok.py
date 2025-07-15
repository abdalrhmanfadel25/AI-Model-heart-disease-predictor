#!/usr/bin/env python3
"""
Ngrok Setup Script for Heart Disease Prediction App
===================================================

This script helps download and set up Ngrok for local deployment.
"""

import os
import requests
import zipfile
import subprocess
import sys
from pathlib import Path

def download_ngrok():
    """Download Ngrok for Windows."""
    print("📥 Downloading Ngrok...")
    
    # Ngrok download URL for Windows
    ngrok_url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
    
    try:
        # Download Ngrok
        response = requests.get(ngrok_url, stream=True)
        response.raise_for_status()
        
        # Save to current directory
        zip_path = "ngrok.zip"
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("✅ Ngrok downloaded successfully!")
        
        # Extract Ngrok
        print("📁 Extracting Ngrok...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Remove zip file
        os.remove(zip_path)
        print("✅ Ngrok extracted successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading Ngrok: {e}")
        return False

def setup_ngrok():
    """Set up Ngrok configuration."""
    print("\n🔧 Setting up Ngrok...")
    
    # Check if ngrok.exe exists
    if not os.path.exists("ngrok.exe"):
        print("❌ ngrok.exe not found. Please download it manually from https://ngrok.com/download")
        return False
    
    # Test Ngrok
    try:
        result = subprocess.run(["./ngrok.exe", "version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ngrok is working!")
            return True
        else:
            print("❌ Ngrok test failed")
            return False
    except Exception as e:
        print(f"❌ Error testing Ngrok: {e}")
        return False

def create_ngrok_script():
    """Create a batch script to run Ngrok easily."""
    print("\n📝 Creating Ngrok startup script...")
    
    script_content = """@echo off
echo Starting Ngrok tunnel for Heart Disease Prediction App...
echo.
echo Make sure your Streamlit app is running on port 8501 first!
echo.
ngrok.exe http 8501
pause
"""
    
    with open("start_ngrok.bat", 'w') as f:
        f.write(script_content)
    
    print("✅ Created start_ngrok.bat")

def create_streamlit_script():
    """Create a batch script to run Streamlit easily."""
    print("\n📝 Creating Streamlit startup script...")
    
    script_content = """@echo off
echo Starting Heart Disease Prediction App...
echo.
cd ui
python -m streamlit run app.py --server.port 8501
pause
"""
    
    with open("start_streamlit.bat", 'w') as f:
        f.write(script_content)
    
    print("✅ Created start_streamlit.bat")

def main():
    """Main setup function."""
    print("🚀 Ngrok Setup for Heart Disease Prediction App")
    print("=" * 50)
    
    # Download Ngrok
    if download_ngrok():
        # Set up Ngrok
        if setup_ngrok():
            # Create scripts
            create_ngrok_script()
            create_streamlit_script()
            
            print("\n" + "=" * 50)
            print("🎉 Ngrok setup complete!")
            print("\n📋 How to use:")
            print("1. Run 'start_streamlit.bat' to start your app")
            print("2. Run 'start_ngrok.bat' in a new terminal to create tunnel")
            print("3. Share the Ngrok URL with your friends!")
            print("\n💡 The Ngrok URL will look like: https://abc123.ngrok-free.app")
        else:
            print("\n❌ Ngrok setup failed. Please try manual installation.")
    else:
        print("\n❌ Ngrok download failed. Please download manually from https://ngrok.com/download")

if __name__ == "__main__":
    main() 