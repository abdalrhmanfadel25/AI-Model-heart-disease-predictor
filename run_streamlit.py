#!/usr/bin/env python3
"""
Streamlit App Launcher
======================

Simple script to launch the Heart Disease Prediction Streamlit app.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application."""
    print("🚀 Starting Heart Disease Prediction App...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("ui/app.py"):
        print("❌ Error: ui/app.py not found!")
        print("Please run this script from the Heart_Disease_Project directory.")
        return
    
    # Check if model files exist
    if not os.path.exists("models/heart_disease_pipeline.pkl"):
        print("❌ Error: Model file not found!")
        print("Please run the model export script first:")
        print("python model_export_deployment.py")
        return
    
    print("✅ All files found!")
    print("🌐 Starting Streamlit server...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Change to ui directory and run streamlit
        os.chdir("ui")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit server stopped.")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

if __name__ == "__main__":
    main() 