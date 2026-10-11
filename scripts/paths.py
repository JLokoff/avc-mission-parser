"""Portable tool locations; inference needs only Python's standard library."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_DIR = Path(os.environ.get('AVC_LLAMA_DIR', ROOT / 'tools/llama.cpp')).resolve()

def binary(name):
    override = os.environ.get('AVC_' + name.upper().replace('-', '_'))
    if override:
        return Path(override).resolve()
    filename = name + ('.exe' if os.name == 'nt' else '')
    candidates = [LLAMA_DIR / 'build/bin' / filename, LLAMA_DIR / 'build/bin/Release' / filename]
    return next((p for p in candidates if p.exists()), candidates[0])

CONVERTER_PYTHON = Path(os.environ.get('AVC_CONVERTER_PYTHON',
    ROOT / '.venv-convert' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')))
