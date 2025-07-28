#!/usr/bin/env python3
"""
Docker Test Script
==================

This script simulates the exact Docker environment and tests the automated pipeline
to ensure it works correctly when deployed.

Usage:
    python docker_test_local.py

Author: AI Assistant
Date: July 29, 2025
"""

import os
import sys
import tempfile
import shutil
import json
import subprocess
from pathlib import Path

def create_docker_simulation():
    """Create a simulation of the Docker environment"""
    print("🐳 Creating Docker environment simulation...")
    
    # Create temporary /app structure
    temp_dir = tempfile.mkdtemp(prefix="docker_sim_")
    app_dir = os.path.join(temp_dir, "app")
    input_dir = os.path.join(app_dir, "input")
    output_dir = os.path.join(app_dir, "output")
    
    os.makedirs(app_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy project files to app directory
    project_files = [
        "automated_pipeline.py",
        "generate_json_output.py",
        "intelligent_filter.py",
        "enhanced_metadata_extractor.py",
        "pos_features_handler.py",
        "config_main.json",
        "requirements.txt"
    ]
    
    for file in project_files:
        if os.path.exists(file):
            shutil.copy2(file, app_dir)
            print(f"   ✅ Copied {file}")
        else:
            print(f"   ⚠️ Missing {file}")
    
    # Copy directories
    directories = ["src", "models"]
    for dir_name in directories:
        if os.path.exists(dir_name):
            dst_dir = os.path.join(app_dir, dir_name)
            shutil.copytree(dir_name, dst_dir)
            print(f"   ✅ Copied {dir_name}/")
        else:
            print(f"   ⚠️ Missing {dir_name}/")
    
    # Copy test PDFs
    current_input = "input"
    if os.path.exists(current_input):
        pdf_files = [f for f in os.listdir(current_input) if f.endswith('.pdf')]
        if pdf_files:
            for pdf_file in pdf_files[:2]:  # Test with first 2 PDFs
                src = os.path.join(current_input, pdf_file)
                dst = os.path.join(input_dir, pdf_file)
                shutil.copy2(src, dst)
                print(f"   📄 Copied test PDF: {pdf_file}")
        else:
            print("   ⚠️ No test PDFs found")
            return None
    else:
        print("   ⚠️ Input directory not found")
        return None
    
    return temp_dir, app_dir, input_dir, output_dir

def run_docker_simulation(app_dir, input_dir, output_dir):
    """Run the automated pipeline in Docker-like environment"""
    print("🚀 Running Docker simulation...")
    
    # Change to app directory (simulate Docker working directory)
    original_cwd = os.getcwd()
    os.chdir(app_dir)
    
    # Set environment variables (simulate Docker ENV)
    env = os.environ.copy()
    env.update({
        'MODE': '1A',
        'USE_ML': 'true',
        'PYTHONPATH': app_dir
    })
    
    try:
        # Run the automated pipeline (simulate Docker CMD)
        print("🤖 Executing: python automated_pipeline.py")
        result = subprocess.run(
            [sys.executable, "automated_pipeline.py"],
            cwd=app_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("📤 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("📤 STDERR:")
            print(result.stderr)
        
        success = result.returncode == 0
        print(f"🎯 Return code: {result.returncode}")
        
        return success
        
    except subprocess.TimeoutExpired:
        print("⏰ Process timed out")
        return False
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return False
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def verify_docker_outputs(output_dir):
    """Verify outputs match expected Docker behavior"""
    print("🔍 Verifying Docker simulation outputs...")
    
    if not os.path.exists(output_dir):
        print("❌ Output directory not found")
        return False
    
    # List all files in output
    all_files = os.listdir(output_dir)
    json_files = [f for f in all_files if f.endswith('.json') and not f.startswith('_')]
    summary_files = [f for f in all_files if f.startswith('_processing_summary.json')]
    
    print(f"📁 Output directory contents:")
    for file in all_files:
        print(f"   - {file}")
    
    # Verify JSON outputs
    if not json_files:
        print("❌ No JSON output files generated")
        return False
    
    print(f"✅ Generated {len(json_files)} JSON outputs")
    
    # Verify summary
    if summary_files:
        try:
            summary_path = os.path.join(output_dir, summary_files[0])
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            print(f"📊 Processing summary:")
            print(f"   - Environment: {summary.get('environment', 'unknown')}")
            print(f"   - Total outputs: {summary.get('total_outputs', 0)}")
            print(f"   - Success: {summary.get('success', False)}")
            print(f"   - Timestamp: {summary.get('processing_timestamp', 'unknown')}")
            
        except Exception as e:
            print(f"⚠️ Could not read summary: {e}")
    
    # Validate JSON structure
    valid_count = 0
    for json_file in json_files:
        try:
            json_path = os.path.join(output_dir, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required structure
            if 'title' in data and 'outline' in data:
                outline = data.get('outline', [])
                print(f"   ✅ {json_file}: Valid (title: '{data['title']}', outline: {len(outline)} items)")
                valid_count += 1
            else:
                print(f"   ❌ {json_file}: Invalid structure")
                
        except Exception as e:
            print(f"   ❌ {json_file}: Error - {e}")
    
    success_rate = valid_count / len(json_files) if json_files else 0
    print(f"📈 Validation success rate: {success_rate*100:.1f}% ({valid_count}/{len(json_files)})")
    
    return success_rate >= 0.8  # At least 80% success rate

def cleanup_simulation(temp_dir):
    """Clean up simulation environment"""
    print("🧹 Cleaning up simulation environment...")
    try:
        shutil.rmtree(temp_dir)
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def main():
    """Main Docker test function"""
    print("🐳 DOCKER SIMULATION TEST")
    print("=" * 30)
    print("Testing the exact Docker run scenario:")
    print("docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none <image>")
    print()
    
    temp_dir = None
    
    try:
        # Create Docker simulation
        result = create_docker_simulation()
        if not result:
            print("❌ Failed to create Docker simulation")
            return False
        
        temp_dir, app_dir, input_dir, output_dir = result
        print(f"📁 Simulation directory: {temp_dir}")
        
        # Run simulation
        execution_success = run_docker_simulation(app_dir, input_dir, output_dir)
        
        if not execution_success:
            print("❌ Docker simulation execution failed")
            return False
        
        # Verify outputs
        verification_success = verify_docker_outputs(output_dir)
        
        overall_success = execution_success and verification_success
        
        print("\n" + "="*50)
        if overall_success:
            print("🎉 DOCKER SIMULATION SUCCESSFUL!")
            print("✅ Your automated pipeline is ready for Docker deployment")
            print("✅ The container will automatically process PDFs and generate JSON outputs")
            print("\n🐳 To deploy, run:")
            print("docker build --platform linux/amd64 -t mysolution.adobe .")
            print("docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none mysolution.adobe")
        else:
            print("❌ DOCKER SIMULATION FAILED")
            print("❌ Please check the issues above before Docker deployment")
            
        return overall_success
        
    except Exception as e:
        print(f"❌ Simulation failed with error: {e}")
        return False
        
    finally:
        if temp_dir:
            cleanup_simulation(temp_dir)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
