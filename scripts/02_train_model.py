"""
Teach one selected model to turn mission commands into the six Schema V3 fields. It uses
the frozen training data and saves the learned changes as a small adapter. Use --smoke
for a short installation check; a smoke run is not a trained model for the results
table.
"""
# Previously: 02_finetune_lora.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from model_workflow import main

if __name__ == "__main__":
    main(['train'] + sys.argv[1:])
