"""Create Q8 directly from merged weights; Q6/Q4 independently from one FP32 GGUF."""
import argparse
import json
import shutil
import subprocess
import sys
from comparison_common import ROOT, MODELS, sha256, verify_data
from gguf_workflow import (convert, completed_run, require_tools, prepare_model_config_for_converter,
                           prepare_tokenizer_for_converter, CONVERTER, CONVERTER_PYTHON, llama_version)
from quantization_common import artifact_path, atomic_json
from paths import binary

def verify_artifact(key, quant, require_training=False):
    output = artifact_path(key, quant)
    if not output.exists() and not output.with_suffix('.json').exists():
        return False
    meta = json.loads(output.with_suffix('.json').read_text())
    if meta['gguf_sha256'] != sha256(output) or meta['quantization'] != quant:
        raise ValueError('Artifact hash or quantization mismatch')
    if meta['source']['model_key'] != key or meta['datasets'] != verify_data():
        raise ValueError('Artifact model/data mismatch')
    if meta['prompt_sha256'] != sha256(ROOT / 'prompts' / f'{key}.json'):
        raise ValueError('Prompt bundle differs from conversion')
    if require_training:
        training = completed_run(key)
        if meta['adapter_sha256'] != training['adapter_sha256'] or meta['source'] != training['source']:
            raise ValueError('Artifact belongs to a different trained adapter')
    return True

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--quants', nargs='+', choices=['Q8_0', 'Q6_K', 'Q4_K_M'], default=['Q8_0', 'Q6_K', 'Q4_K_M'])
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    training = completed_run(a.model)
    version = require_tools()
    merged = ROOT / 'merged' / a.model
    if not merged.exists():
        subprocess.run([sys.executable, str(ROOT / 'scripts/compare.py'), 'export', '--model', a.model], check=True)
    if json.loads((merged / 'training-run.json').read_text()) != training:
        raise ValueError('Merged export belongs to another training run')
    weights = json.loads((merged / 'export-metadata.json').read_text())
    if weights['dtype'] != 'float32' or any(sha256(merged / n) != h for n, h in weights['weight_files'].items()):
        raise ValueError('Expected unchanged FP32 merged weights')
    identity = {'model_key': a.model, 'source': training['source'], 'datasets': training['datasets'],
                'adapter_sha256': training['adapter_sha256'], 'merged_weight_files': weights['weight_files'],
                'prompt_sha256': sha256(ROOT / 'prompts' / f'{a.model}.json'), 'llama_cpp': version}
    if 'Q8_0' in a.quants:
        if not verify_artifact(a.model, 'Q8_0', require_training=True):
            convert(a)
    if set(a.quants) == {'Q8_0'}:
        return
    prepare_model_config_for_converter(a.model, merged)
    prepare_tokenizer_for_converter(a.model, merged)
    master = artifact_path(a.model, 'F32')
    master.parent.mkdir(parents=True, exist_ok=True)
    def create(output, quant, command, extra=None):
        if output.exists() or output.with_suffix('.json').exists():
            if not verify_artifact(a.model, quant, require_training=True):
                raise ValueError('Incomplete output')
            meta = json.loads(output.with_suffix('.json').read_text())
            for k, v in {**identity, **(extra or {})}.items():
                if meta.get(k) != v:
                    raise ValueError(f'Existing {quant} differs: {k}')
            print('Verified existing', output)
            return
        if shutil.disk_usage(ROOT).free < master_size + 2_000_000_000:
            raise RuntimeError('Insufficient disk: need FP32 model size plus a 2 GB reserve')
        partial = output.with_suffix('.gguf.partial')
        if partial.exists():
            raise RuntimeError(f'Inspect previous incomplete conversion: {partial}')
        subprocess.run([str(x) for x in command(partial)], check=True)
        subprocess.run([str(CONVERTER_PYTHON), str(ROOT / 'scripts/inspect_study_gguf.py'),
                        str(partial), '--expect', quant], check=True)
        atomic_json(output.with_suffix('.json.partial'), {**identity, 'quantization': quant,
                    'gguf_sha256': sha256(partial), 'gguf_bytes': partial.stat().st_size, **(extra or {})})
        partial.rename(output)
        output.with_suffix('.json.partial').rename(output.with_suffix('.json'))
    master_size = sum((merged / n).stat().st_size for n in weights['weight_files'])
    create(master, 'F32', lambda tmp: [CONVERTER_PYTHON, CONVERTER, merged, '--outfile', tmp, '--outtype', 'f32'])
    for quant in a.quants:
        if quant == 'Q8_0':
            continue
        create(artifact_path(a.model, quant), quant,
               lambda tmp: [binary('llama-quantize'), master, tmp, quant, a.threads],
               {'input_fp32_sha256': sha256(master), 'quantizer_sha256': sha256(binary('llama-quantize'))})

if __name__ == '__main__':
    main()
