#!/usr/bin/env python3
"""
Test Script for Automated Pipeline
==================================

This script tests the automated pipeline to ensure it works correctly
before deploying to Docker.

Usage:
    python test_automated_pipeline.py

Author: AI Assistant
Date: July 29, 2025
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

def setup_test_environment():
    """Setup a temporary test environment"""
    print("🧪 Setting up test environment...")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix="pipeline_test_")
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy test PDFs if they exist
    current_input = "input"
    if os.path.exists(current_input):
        pdf_files = [f for f in os.listdir(current_input) if f.endswith('.pdf')]
        if pdf_files:
            print(f"📄 Found {len(pdf_files)} test PDFs")
            for pdf_file in pdf_files[:2]:  # Test with first 2 PDFs
                src = os.path.join(current_input, pdf_file)
                dst = os.path.join(input_dir, pdf_file)
                shutil.copy2(src, dst)
                print(f"   - Copied {pdf_file}")
        else:
            print("⚠️ No PDFs found in input directory")
            return None, None, None
    else:
        print("⚠️ Input directory not found")
        return None, None, None
    
    return temp_dir, input_dir, output_dir

def test_automated_pipeline(test_input_dir, test_output_dir):
    """Test the automated pipeline"""
    print("🚀 Testing automated pipeline...")
    
    # Import and setup the automated pipeline
    try:
        from automated_pipeline import (
            detect_environment, setup_environment, find_pdf_files,
            process_with_advanced_pipeline, process_with_basic_pipeline,
            generate_processing_summary
        )
        
        # Override environment for testing
        env_config = {
            'input_dir': test_input_dir,
            'output_dir': test_output_dir,
            'base_dir': os.getcwd(),
            'is_docker': False
        }
        
        # Setup environment
        setup_environment(env_config)
        
        # Find PDFs
        pdf_files = find_pdf_files(test_input_dir)
        if not pdf_files:
            print("❌ No PDF files found for testing")
            return False
        
        print(f"📄 Testing with {len(pdf_files)} PDFs")
        
        # Try advanced pipeline
        success = process_with_advanced_pipeline(pdf_files, env_config)
        
        if not success:
            print("⚠️ Advanced pipeline failed, trying basic pipeline...")
            success = process_with_basic_pipeline(pdf_files, env_config)
        
        # Generate summary
        generate_processing_summary(env_config)
        
        return success
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def verify_outputs(test_output_dir):
    """Verify the generated outputs"""
    print("🔍 Verifying outputs...")
    
    json_files = [f for f in os.listdir(test_output_dir) if f.endswith('.json') and not f.startswith('_')]
    
    if not json_files:
        print("❌ No JSON output files generated")
        return False
    
    print(f"✅ Generated {len(json_files)} JSON files:")
    
    valid_count = 0
    for json_file in json_files:
        try:
            json_path = os.path.join(test_output_dir, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Basic validation
            if 'title' in data and 'outline' in data:
                print(f"   ✅ {json_file} - Valid structure")
                print(f"      Title: {data.get('title', 'N/A')}")
                print(f"      Outline items: {len(data.get('outline', []))}")
                valid_count += 1
            else:
                print(f"   ❌ {json_file} - Invalid structure")
                
        except Exception as e:
            print(f"   ❌ {json_file} - Error: {e}")
    
    print(f"📊 Validation result: {valid_count}/{len(json_files)} files valid")
    return valid_count == len(json_files)

def cleanup_test_environment(temp_dir):
    """Clean up test environment"""
    print("🧹 Cleaning up test environment...")
    try:
        shutil.rmtree(temp_dir)
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def main():
    """Main test function"""
    print("🧪 AUTOMATED PIPELINE TEST")
    print("=" * 30)
    
    temp_dir = None
    
    try:
        # Setup test environment
        temp_dir, test_input_dir, test_output_dir = setup_test_environment()
        
        if not temp_dir:
            print("❌ Failed to setup test environment")
            return False
        
        print(f"📁 Test directory: {temp_dir}")
        
        # Test the pipeline
        pipeline_success = test_automated_pipeline(test_input_dir, test_output_dir)
        
        if not pipeline_success:
            print("❌ Pipeline test failed")
            return False
        
        # Verify outputs
        validation_success = verify_outputs(test_output_dir)
        
        if validation_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Automated pipeline is working correctly")
            print("✅ Docker deployment should work properly")
        else:
            print("\n⚠️ VALIDATION WARNINGS")
            print("⚠️ Some outputs may have issues")
        
        return pipeline_success and validation_success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
        
    finally:
        # Always cleanup
        if temp_dir:
            cleanup_test_environment(temp_dir)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
