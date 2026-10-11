"""
Create smaller model files that can run in llama.cpp. This step combines the original
model and its learned adapter when a merged copy is missing, then creates the selected
Q8, Q6, or Q4 files. Each smaller version comes from the full-precision trained model,
never from a previously compressed version. It verifies existing files before reusing
them.
"""
# Previously: 05_merge_for_gguf.py plus quantize.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from quantization import main

if __name__ == "__main__":
    main()
