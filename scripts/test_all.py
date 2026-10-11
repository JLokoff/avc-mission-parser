"""Run offline regression tests without downloading models or importing PyTorch."""
import sys
import faulthandler
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
faulthandler.dump_traceback_later(60, exit=True)
try:
    result = unittest.TextTestRunner(verbosity=2).run(suite)
finally:
    faulthandler.cancel_dump_traceback_later()
raise SystemExit(0 if result.wasSuccessful() else 1)
