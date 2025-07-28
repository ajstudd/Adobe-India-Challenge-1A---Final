#!/usr/bin/env python3
"""
Setup Script for Adobe India Hackathon Project
==============================================

This script ensures all dependencies are properly installed,
including spaCy models required for POS features.

Usage:
    python setup_project.py

Author: AI Assistant
Date: July 28, 2025
"""

import os
import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is 3.10+"""
    if sys.version_info < (3, 10):
        logger.error(f"Python 3.10+ required, but {sys.version} found")
        return False
    logger.info(f"✅ Python {sys.version} is compatible")
    return True

def install_requirements():
    """Install Python packages from requirements.txt"""
    logger.info("📦 Installing Python packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("✅ Python packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install packages: {e}")
        return False

def install_spacy_model():
    """Install spaCy English model for POS features"""
    logger.info("🔤 Installing spaCy English model...")
    try:
        # First try to load the model to check if it's already installed
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy English model already installed")
            return True
        except OSError:
            pass
        
        # Model not found, install it
        logger.info("📥 Downloading spaCy English model...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        
        # Verify installation
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ spaCy English model installed and verified")
        return True
        
    except ImportError:
        logger.error("❌ spaCy not installed. Please install requirements first.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install spaCy model: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error installing spaCy model: {e}")
        return False

def verify_installation():
    """Verify that all components are working"""
    logger.info("🔍 Verifying installation...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        import sklearn
        import fitz  # PyMuPDF
        import spacy
        
        # Test spaCy model
        nlp = spacy.load("en_core_web_sm")
        
        # Test a simple sentence
        doc = nlp("This is a test sentence.")
        pos_tags = [token.pos_ for token in doc]
        
        logger.info("✅ All components verified successfully")
        logger.info(f"✅ spaCy POS tagging working: {pos_tags}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    logger.info("📁 Creating necessary directories...")
    
    directories = [
        "input",
        "output", 
        "models",
        "labelled_data",
        "predictions",
        "reviewed"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"📁 Created/verified directory: {directory}")
    
    return True

def check_config():
    """Check if configuration files exist"""
    logger.info("⚙️ Checking configuration...")
    
    if not os.path.exists("config_main.json"):
        logger.warning("⚠️ config_main.json not found")
        return False
    
    # Check if POS features are enabled
    try:
        import json
        with open("config_main.json", "r") as f:
            config = json.load(f)
        
        pos_enabled = config.get("feature_engineering", {}).get("pos_features", False)
        if pos_enabled:
            logger.info("✅ POS features enabled in configuration")
        else:
            logger.warning("⚠️ POS features disabled in configuration")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error reading configuration: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 ADOBE INDIA HACKATHON PROJECT SETUP")
    print("=" * 50)
    print("Setting up your environment for optimal performance...")
    print()
    
    success = True
    
    # Step 1: Check Python version
    if not check_python_version():
        success = False
    
    # Step 2: Install requirements
    if success and not install_requirements():
        success = False
    
    # Step 3: Install spaCy model
    if success and not install_spacy_model():
        success = False
    
    # Step 4: Create directories
    if success and not create_directories():
        success = False
    
    # Step 5: Check configuration
    if success and not check_config():
        logger.warning("⚠️ Configuration issues detected, but continuing...")
    
    # Step 6: Verify installation
    if success and not verify_installation():
        success = False
    
    print()
    if success:
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 30)
        print("✅ All dependencies installed")
        print("✅ spaCy English model ready")
        print("✅ POS features will work correctly")
        print("✅ Project structure created")
        print()
        print("🚀 You can now run:")
        print("   python master_pipeline.py")
        print("   python generate_json_output.py")
        print("   python src/pipeline.py")
    else:
        print("❌ SETUP FAILED")
        print("=" * 15)
        print("Please check the error messages above and try again.")
        print("You may need to:")
        print("- Install Python 3.10+")
        print("- Check internet connection for downloads")
        print("- Run with administrator privileges")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
