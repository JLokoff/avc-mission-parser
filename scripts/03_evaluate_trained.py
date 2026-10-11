"""
Measure the model after training using the same evaluation commands as the baseline.
This loads the downloaded model together with its trained adapter and saves accuracy,
response times, and individual answers. It does not change the model or train on the
evaluation data.
"""
# Previously: 03_evaluate.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from model_workflow import main

if __name__ == "__main__":
    main(['eval'] + sys.argv[1:])
