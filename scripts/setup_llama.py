"""Fetch the pinned llama.cpp revision, apply the converter patch, build locally."""
import argparse
import subprocess
from pathlib import Path
from paths import ROOT, LLAMA_DIR

COMMIT = 'b49650adb31f2e49a0d76113aeb1792134fd8413'

def run(*args):
    subprocess.run([str(x) for x in args], check=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cpu', 'metal', 'cuda'], required=True)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--cuda-arch', help='For example 87 on Jetson Orin; omit on other GPUs')
    a = p.parse_args()
    if a.jobs < 1:
        p.error('--jobs must be positive')
    if not LLAMA_DIR.exists():
        run('git', 'clone', '--depth', '1', '--branch', 'b11026',
            'https://github.com/ggml-org/llama.cpp.git', LLAMA_DIR)
    head = subprocess.check_output(['git', '-C', str(LLAMA_DIR), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        raise SystemExit('Existing llama.cpp checkout is not the pinned commit; preserve it and use AVC_LLAMA_DIR for a fresh checkout.')
    patch = ROOT / 'patches/llama-b11026-granite-all-attention.patch'
    if 'AVC_GRANITE_ALL_ATTENTION_WRITER_FIX' not in (LLAMA_DIR / 'conversion/granite.py').read_text():
        run('git', '-C', LLAMA_DIR, 'apply', '--check', patch)
        run('git', '-C', LLAMA_DIR, 'apply', patch)
    flags = ['-DGGML_METAL=' + ('ON' if a.backend == 'metal' else 'OFF'),
             '-DGGML_CUDA=' + ('ON' if a.backend == 'cuda' else 'OFF'), '-DLLAMA_CURL=OFF']
    if a.cuda_arch:
        flags.append('-DCMAKE_CUDA_ARCHITECTURES=' + a.cuda_arch)
    run('cmake', '-S', LLAMA_DIR, '-B', LLAMA_DIR / 'build', '-DCMAKE_BUILD_TYPE=Release', *flags)
    run('cmake', '--build', LLAMA_DIR / 'build', '--config', 'Release',
        '--target', 'llama-server', 'llama-quantize', '--parallel', a.jobs)

if __name__ == '__main__':
    main()
