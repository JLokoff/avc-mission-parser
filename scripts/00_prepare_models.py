"""
Prepare the models and tools before you train or run them. Choose download to save a
model on this computer, check to inspect the data and test the selected device without
training, or build to install the local llama.cpp tools. Run download and check for each
model; build is needed once per computer.
"""
# Previously: setup_llama.py; download/check commands previously lived in compare.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
import argparse
import subprocess
from artifacts import ROOT, LLAMA_DIR

COMMIT = 'b49650adb31f2e49a0d76113aeb1792134fd8413'

def run(*args):
    subprocess.run([str(x) for x in args], check=True)

def build_tools(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cpu', 'metal', 'cuda'], required=True)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--cuda-arch', help='For example 87 on Jetson Orin; omit on other GPUs')
    a = p.parse_args(argv)
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

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['download', 'check', 'build'], help='Download weights, check a model, or build llama.cpp')
    a, rest = p.parse_known_args()
    if a.task == 'build':
        build_tools(rest)
    else:
        from model_workflow import main as model_main
        model_main([a.task, *rest])

if __name__ == '__main__':
    main()
