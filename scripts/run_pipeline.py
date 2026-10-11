"""Portable sequential runner; each explicit stage stops on its first failure."""
import argparse
import subprocess
import sys
from datetime import datetime
from comparison_common import ROOT, MODELS

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['download', 'check', 'baseline', 'train', 'evaluate', 'export', 'quantize', 'benchmark'])
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--gpu-layers', type=int, default=99)
    p.add_argument('--smoke', action='store_true', help='Only for train')
    p.add_argument('--limit', type=int, help='Partial baseline/evaluate/benchmark, never a final score')
    p.add_argument('--quants', nargs='+', choices=['Q8_0', 'Q6_K', 'Q4_K_M'], default=['Q8_0', 'Q6_K', 'Q4_K_M'])
    p.add_argument('--cache-prefix', action='store_true', help='For benchmark only')
    a = p.parse_args()
    if a.smoke and a.stage != 'train':
        p.error('--smoke only applies to train')
    if a.cache_prefix and a.stage != 'benchmark':
        p.error('--cache-prefix only applies to benchmark')
    if a.limit and a.stage not in ['baseline', 'evaluate', 'benchmark']:
        p.error('--limit only applies to evaluation stages')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    for key in a.models:
        if a.stage == 'quantize':
            commands = [['quantize.py', '--model', key, '--threads', str(a.threads), '--quants', *a.quants]]
        elif a.stage == 'benchmark':
            commands = [['04_benchmark_gguf.py', '--model', key, '--quantization', quant,
                         '--label', f'{key}-{quant.lower()}-grammar-{"cached" if a.cache_prefix else "uncached"}-{stamp}',
                         '--grammar', '--threads', str(a.threads), '--gpu-layers', str(a.gpu_layers)] +
                        (['--cache-prefix'] if a.cache_prefix else []) +
                        (['--limit', str(a.limit)] if a.limit else []) for quant in a.quants]
        else:
            command = {'baseline': 'eval', 'evaluate': 'eval'}.get(a.stage, a.stage)
            commands = [['compare.py', command, '--model', key, '--device', 'cpu' if command == 'export' else a.device,
                         '--threads', str(a.threads)] + (['--base'] if a.stage == 'baseline' else []) +
                        (['--smoke'] if a.smoke else []) + (['--limit', str(a.limit)] if a.limit else [])]
        for cmd in commands:
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / cmd[0]), *cmd[1:]], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
