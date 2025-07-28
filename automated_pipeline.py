#!/usr/bin/env python3
"""
Automated Pipeline for Docker Container
======================================

This script automatically processes PDFs from the input folder and generates
JSON outputs using the advanced ML pipeline. It's designed to run automatically
in the Docker container without manual intervention.

Features:
✅ Automatically detects and processes all PDFs in input folder
✅ Uses advanced ML models for heading detection
✅ Generates properly formatted JSON outputs
✅ Handles errors gracefully and logs progress
✅ Works both in Docker and local environments

Usage:
    python automated_pipeline.py

Author: AI Assistant
Date: July 29, 2025
"""

import os
import sys
import logging
import json
import glob
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def detect_environment():
    """Detect if running in Docker or local environment"""
    if os.path.exists('/app') and os.path.exists('/app/input'):
        logger.info("🐳 Running in Docker environment")
        return {
            'input_dir': '/app/input',
            'output_dir': '/app/output',
            'base_dir': '/app',
            'is_docker': True
        }
    else:
        logger.info("💻 Running in local environment")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return {
            'input_dir': os.path.join(base_dir, 'input'),
            'output_dir': os.path.join(base_dir, 'output'),
            'base_dir': base_dir,
            'is_docker': False
        }

def setup_environment(env_config):
    """Setup environment and ensure directories exist"""
    logger.info("🔧 Setting up environment...")
    
    # Ensure output directory exists
    os.makedirs(env_config['output_dir'], exist_ok=True)
    
    # Set Python path to include base directory
    if env_config['base_dir'] not in sys.path:
        sys.path.insert(0, env_config['base_dir'])
    
    # Change working directory
    os.chdir(env_config['base_dir'])
    
    logger.info(f"📁 Input directory: {env_config['input_dir']}")
    logger.info(f"📁 Output directory: {env_config['output_dir']}")
    logger.info(f"📁 Working directory: {env_config['base_dir']}")
    
    return True

def find_pdf_files(input_dir):
    """Find all PDF files in the input directory"""
    if not os.path.exists(input_dir):
        logger.error(f"❌ Input directory not found: {input_dir}")
        return []
    
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    if not pdf_files:
        logger.warning(f"⚠️ No PDF files found in {input_dir}")
        return []
    
    logger.info(f"📄 Found {len(pdf_files)} PDF files to process")
    for pdf_file in pdf_files:
        logger.info(f"   - {os.path.basename(pdf_file)}")
    
    return pdf_files

def process_with_advanced_pipeline(pdf_files, env_config):
    """Process PDFs using the advanced ML pipeline"""
    logger.info("🚀 Starting advanced ML pipeline processing...")
    
    try:
        # Import the advanced JSON generator
        from generate_json_output import JSONOutputGenerator
        
        # Initialize the generator
        generator = JSONOutputGenerator()
        
        # Override directories for Docker environment
        if env_config['is_docker']:
            generator.input_dir = env_config['input_dir']
            generator.output_dir = env_config['output_dir']
        
        # Load the model
        logger.info("🧠 Loading ML model...")
        try:
            generator.load_model("latest")
            logger.info(f"✅ Model loaded: {generator.loaded_model_version}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
        
        # Process each PDF
        success_count = 0
        total_count = len(pdf_files)
        
        for pdf_file in pdf_files:
            try:
                pdf_name = os.path.basename(pdf_file)
                logger.info(f"📄 Processing: {pdf_name}")
                
                # Extract and process PDF
                blocks_df = generator.extract_pdf_to_blocks(pdf_file)
                
                if blocks_df is None or len(blocks_df) == 0:
                    logger.warning(f"⚠️ No blocks extracted from {pdf_name}")
                    continue
                
                # Predict headings
                headings_df = generator.predict_headings(blocks_df)
                
                if headings_df is None or len(headings_df) == 0:
                    logger.warning(f"⚠️ No headings detected in {pdf_name}")
                    # Still create output with empty outline
                    json_data = {
                        "title": pdf_name.replace('.pdf', ''),
                        "outline": []
                    }
                else:
                    # Create JSON structure
                    json_data = generator.create_json_structure(pdf_name, headings_df, blocks_df)
                
                # Save JSON output
                output_filename = pdf_name.replace('.pdf', '.json')
                output_path = os.path.join(env_config['output_dir'], output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Saved: {output_filename}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_name}: {e}")
                continue
        
        logger.info(f"🎉 Processing complete: {success_count}/{total_count} files successful")
        return success_count > 0
        
    except ImportError as e:
        logger.error(f"❌ Failed to import advanced pipeline: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Advanced pipeline failed: {e}")
        return False

def process_with_basic_pipeline(pdf_files, env_config):
    """Fallback to basic pipeline if advanced pipeline fails"""
    logger.info("🔧 Using basic pipeline as fallback...")
    
    try:
        # Import basic extraction
        from src.extraction.outline_extractor import extract_outline_from_pdf
        
        success_count = 0
        
        for pdf_file in pdf_files:
            try:
                pdf_name = os.path.basename(pdf_file)
                logger.info(f"📄 Processing with basic pipeline: {pdf_name}")
                
                # Extract outline
                outline_data = extract_outline_from_pdf(pdf_file)
                
                # Save JSON output
                output_filename = pdf_name.replace('.pdf', '.json')
                output_path = os.path.join(env_config['output_dir'], output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(outline_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Saved: {output_filename}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_name}: {e}")
                continue
        
        logger.info(f"🎉 Basic processing complete: {success_count}/{len(pdf_files)} files successful")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"❌ Basic pipeline failed: {e}")
        return False

def generate_processing_summary(env_config):
    """Generate a summary of the processing results"""
    logger.info("📊 Generating processing summary...")
    
    try:
        output_files = glob.glob(os.path.join(env_config['output_dir'], "*.json"))
        
        summary = {
            "processing_timestamp": datetime.now().isoformat(),
            "environment": "docker" if env_config['is_docker'] else "local",
            "total_outputs": len(output_files),
            "output_files": [os.path.basename(f) for f in output_files],
            "success": len(output_files) > 0
        }
        
        summary_path = os.path.join(env_config['output_dir'], "_processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Summary saved to _processing_summary.json")
        logger.info(f"📄 Generated {len(output_files)} JSON files")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to generate summary: {e}")
        return False

def main():
    """Main automated pipeline function"""
    print("🤖 AUTOMATED PDF PROCESSING PIPELINE")
    print("=" * 50)
    print("🎯 Features:")
    print("   ✅ Automatic PDF detection and processing")
    print("   ✅ Advanced ML-based heading detection")
    print("   ✅ JSON output generation")
    print("   ✅ Docker and local environment support")
    print("   ✅ Error handling and logging")
    print()
    
    try:
        # Detect environment
        env_config = detect_environment()
        
        # Setup environment
        if not setup_environment(env_config):
            logger.error("❌ Environment setup failed")
            return False
        
        # Find PDF files
        pdf_files = find_pdf_files(env_config['input_dir'])
        if not pdf_files:
            logger.warning("⚠️ No PDF files to process")
            # Generate empty summary
            generate_processing_summary(env_config)
            return True
        
        # Try advanced pipeline first
        success = process_with_advanced_pipeline(pdf_files, env_config)
        
        # Fallback to basic pipeline if advanced fails
        if not success:
            logger.warning("⚠️ Advanced pipeline failed, trying basic pipeline...")
            success = process_with_basic_pipeline(pdf_files, env_config)
        
        # Generate processing summary
        generate_processing_summary(env_config)
        
        if success:
            logger.info("🎉 Automated processing completed successfully!")
            print("\n🎉 PROCESSING COMPLETED SUCCESSFULLY!")
            print(f"📁 Check the output directory: {env_config['output_dir']}")
            print(f"📄 Generated JSON files for all processed PDFs")
        else:
            logger.error("❌ Automated processing failed")
            print("\n❌ PROCESSING FAILED")
            print("Please check the logs for details")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("⏹️ Processing interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
