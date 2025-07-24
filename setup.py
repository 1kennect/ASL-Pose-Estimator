#!/usr/bin/env python3
"""
Setup script for ASL Transcription System
Helps configure the system and create necessary files
"""

import os
import sys

def create_env_file():
    """Create .env file with OpenAI API key."""
    env_content = """# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Customize the model
# OPENAI_MODEL=gpt-3.5-turbo
# OPENAI_MODEL=gpt-4

# Camera settings (optional)
# CAMERA_INDEX=0
"""
    
    if os.path.exists('.env'):
        print("⚠️  .env file already exists")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            return
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    print("📝 Please edit .env file and add your OpenAI API key")

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        ('mediapipe', 'mediapipe'),
        ('opencv-python', 'cv2'), 
        ('numpy', 'numpy'),
        ('openai', 'openai'),
        ('python-dotenv', 'dotenv')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def main():
    """Main setup function."""
    print("🎯 ASL Transcription System Setup")
    print("=" * 40)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        return
    
    # Create .env file
    print("\n2. Setting up environment...")
    create_env_file()
    
    print("\n3. Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run: python test_system.py (to test camera)")
    print("3. Run: python asl_transcription_system.py (to start the system)")
    
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main() 