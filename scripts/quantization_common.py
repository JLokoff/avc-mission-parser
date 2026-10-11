"""Shared paths and validation primitives for the V3 lower-bit study."""
import json
import os
from pathlib import Path
from comparison_common import ROOT

QUANTS = ('Q6_K', 'Q4_K_M')
STUDY_VERSION = 1


def artifact_path(model, quant):
    folder = ROOT/'gguf' if quant == 'Q8_0' else ROOT/'gguf/quantization'
    if quant == 'F32':
        folder = ROOT/'gguf/quantization/masters'
    return folder/f'{model}-{quant.lower()}.gguf'


def result_path(model, quant, grammar, limit=None):
    label = f'{model}-v3-{quant.lower()}-metal' + ('-grammar' if grammar else '')
    folder = ROOT/'results/quantization'
    suffix = f'-partial{limit}' if limit else ''
    return folder/f'{label}{suffix}.json'


def read(path):
    with Path(path).open() as f:
        return json.load(f)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def require_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f'{label} mismatch. Preserve the existing artifact and inspect before retrying.')
