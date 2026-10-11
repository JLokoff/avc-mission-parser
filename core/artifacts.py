"""
Finds the installed model tools and decides where generated models and their records are
saved. It supports Mac, Windows, and Linux executable locations and writes records
through temporary files to reduce incomplete saves. This combines the former small path
and quantization-helper files.
"""
# Previously: paths.py and quantization_common.py
import os
import json
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

QUANTS = ('Q6_K', 'Q4_K_M')


def artifact_path(model, quant):
    folder = ROOT/'gguf' if quant == 'Q8_0' else ROOT/'gguf/quantization'
    if quant == 'F32':
        folder = ROOT/'gguf/quantization/masters'
    return folder/f'{model}-{quant.lower()}.gguf'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)
