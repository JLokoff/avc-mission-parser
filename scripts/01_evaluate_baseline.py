"""
Measure how the original downloaded model performs before training. This sends the
evaluation commands to the model, compares its answers with the expected fields, and
saves accuracy and response times. Keep this result as the starting point for measuring
what training improves.
"""
# Previously: 00_zeroshot_baseline.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from model_workflow import main

if __name__ == "__main__":
    main(['eval', '--base'] + sys.argv[1:])
