import os
import sys
import subprocess

# Add parent directory to path to import generate_json_output
sys.path.insert(0, '/app')

def run_advanced_json_generation():
    """Run the advanced JSON generation script"""
    try:
        # Set automated mode environment variable
        os.environ['AUTOMATED_MODE'] = 'true'
        
        # Import and run the advanced JSON generator
        from generate_json_output import main as generate_json_main
        print("[INFO] Running advanced JSON generation...")
        generate_json_main()
        print("[INFO] Advanced JSON generation completed successfully")
        
    except Exception as e:
        print(f"[ERROR] Advanced JSON generation failed: {e}")
        # Fallback to basic outline extraction
        print("[INFO] Falling back to basic outline extraction...")
        from src.extraction.outline_extractor import run_outline_extraction
        run_outline_extraction()

def main():
    mode = os.getenv("MODE", "1A")  # for now we have set the default mode to "1A"

    if mode == "1A":
        print("[INFO] Running in Mode 1A: Advanced PDF Processing with ML")
        run_advanced_json_generation()
    else:
        print(f"[ERROR] Unsupported MODE: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
