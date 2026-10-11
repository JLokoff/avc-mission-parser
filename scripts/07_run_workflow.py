"""
Run one workflow stage for all three models in succession, or only the models you
select. For example, benchmark can test Q8, Q6, and Q4 for each model without typing
nine separate commands. It stops if a command fails, so a later step cannot silently
hide an earlier problem.
"""
# Previously: run_pipeline.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
import argparse
import subprocess
from datetime import datetime
from model_common import ROOT, MODELS

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['download', 'check', 'baseline', 'train', 'evaluate', 'quantize', 'benchmark'])
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
    if a.limit is not None and a.stage not in ['baseline', 'evaluate', 'benchmark']:
        p.error('--limit only applies to evaluation stages')
    if a.threads < 1 or a.gpu_layers < 0 or (a.limit is not None and a.limit < 1):
        p.error('Threads and limit must be positive; GPU layers must be nonnegative')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    for key in a.models:
        if a.stage == 'quantize':
            commands = [['04_prepare_gguf.py', '--model', key, '--threads', str(a.threads), '--quants', *a.quants]]
        elif a.stage == 'benchmark':
            commands = [['05_benchmark_models.py', '--model', key, '--quantization', quant,
                         '--label', f'{key}-{quant.lower()}-grammar-{"cached" if a.cache_prefix else "uncached"}-{stamp}',
                         '--grammar', '--threads', str(a.threads), '--gpu-layers', str(a.gpu_layers)] +
                        (['--cache-prefix'] if a.cache_prefix else []) +
                        (['--limit', str(a.limit)] if a.limit else []) for quant in a.quants]
        else:
            entry = {'download': '00_prepare_models.py', 'check': '00_prepare_models.py',
                     'baseline': '01_evaluate_baseline.py', 'train': '02_train_model.py',
                     'evaluate': '03_evaluate_trained.py'}[a.stage]
            task = [a.stage] if a.stage in ('download', 'check') else []
            commands = [[entry, *task, '--model', key, '--device', a.device,
                         '--threads', str(a.threads)] + (['--smoke'] if a.smoke else []) +
                        (['--limit', str(a.limit)] if a.limit is not None else [])]
        for cmd in commands:
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / cmd[0]), *cmd[1:]], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
