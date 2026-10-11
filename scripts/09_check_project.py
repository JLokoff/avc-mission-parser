"""
Check that the repository still works after a code change. These quick tests inspect the
data, scoring, file preparation, command routing, and local HTTP interface without
downloading or training a model. They do not prove GPU compatibility or replace a real
model evaluation; the model check in step 00 tests your installed device.
"""
# Previously: test_all.py
import sys
import faulthandler
import unittest
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'core'))
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
faulthandler.dump_traceback_later(60, exit=True)
try:
    result = unittest.TextTestRunner(verbosity=2).run(suite)
finally:
    faulthandler.cancel_dump_traceback_later()
if not result.wasSuccessful():
    raise SystemExit(1)
subprocess.run([sys.executable, str(ROOT / 'tests/test_schema_v3.py')], check=True)
