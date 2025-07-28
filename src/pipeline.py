import os
import sys
from src.extraction.outline_extractor import run_outline_extraction
# we wanted to keep 1A and 1B together and switch between them using an environment variable
# so that we can run the pipeline in different modes without changing the code
def main():
    mode = os.getenv("MODE", "1A")  # for now we have set the default mode to "1A"

    if mode == "1A":
        print("[INFO] Running in Mode 1A: Outline Extraction")
        run_outline_extraction()

    else:
        print(f"[ERROR] Unsupported MODE: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
