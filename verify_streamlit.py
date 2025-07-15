#!/usr/bin/env python3
"""
Streamlit App Verification Script
================================

Simple script to verify that the Streamlit app is working correctly.
"""

import time
import requests
import webbrowser
from pathlib import Path

def check_streamlit_app():
    """Check if the Streamlit app is running and accessible."""
    print("🔍 Verifying Streamlit App...")
    print("=" * 40)
    
    # Check if model files exist
    model_path = Path("models/heart_disease_pipeline.pkl")
    metadata_path = Path("models/model_metadata.json")
    data_path = Path("data/heart_disease_selected.csv")
    
    print("1. Checking required files...")
    if model_path.exists():
        print("   ✅ Model file found")
    else:
        print("   ❌ Model file missing")
        return False
        
    if metadata_path.exists():
        print("   ✅ Metadata file found")
    else:
        print("   ❌ Metadata file missing")
        return False
        
    if data_path.exists():
        print("   ✅ Data file found")
    else:
        print("   ❌ Data file missing")
        return False
    
    # Check if Streamlit app is running
    print("\n2. Checking Streamlit app status...")
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("   ✅ Streamlit app is running!")
            print("   📱 App URL: http://localhost:8501")
            return True
        else:
            print(f"   ⚠️  App responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Streamlit app is not running")
        print("   💡 Start the app with: powershell -ExecutionPolicy Bypass -File start_streamlit.ps1")
        return False
    except Exception as e:
        print(f"   ❌ Error checking app: {e}")
        return False

def main():
    """Main verification function."""
    if check_streamlit_app():
        print("\n" + "=" * 40)
        print("🎉 Streamlit app verification successful!")
        print("\n📋 Next steps:")
        print("1. Open your browser and go to: http://localhost:8501")
        print("2. Test the prediction functionality")
        print("3. Explore the data insights")
        print("4. Check the about page")
        
        # Ask if user wants to open browser
        try:
            response = input("\n🌐 Open app in browser? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                webbrowser.open("http://localhost:8501")
                print("✅ Browser opened!")
        except KeyboardInterrupt:
            print("\n👋 Verification completed.")
    else:
        print("\n❌ Streamlit app verification failed!")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main() 