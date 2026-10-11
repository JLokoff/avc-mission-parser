"""Benchmark a V3 GGUF. Starts/stops its own server by default; --external-server uses yours.

Only standard-library dependencies; prompt token IDs were saved at export. No
training, data regeneration, or semantic correction occurs here.
"""
import argparse
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from comparison_common import ROOT, MODELS, EVAL_FILES, sha256, verify_data, eval_rows, scored_row, summarize, write_json
from gguf_workflow import SERVER, llama_version
from gguf_runtime import NUMERIC_CHECK, check_generation, default_ubatch
from quantization_common import QUANTS, artifact_path, atomic_json


def request(port, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/{route}', data=data,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=2 if route == 'health' else 300) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if route == 'health':
            raise
        detail = exc.read().decode('utf-8', errors='replace')[:4000]
        raise RuntimeError(f'llama-server /{route} returned HTTP {exc.code}: {detail}') from exc


def benchmark(args):
    lock = verify_data()
    bundle_path = ROOT / 'prompts' / f'{args.model}.json'
    bundle = json.loads(bundle_path.read_text())
    quant = getattr(args, 'quantization', 'Q8_0')
    if quant != 'Q8_0':
        from quantize import verify_artifact
        if not verify_artifact(args.model, quant):
            raise ValueError('Missing verified quantization; run scripts/quantize.py first')
    manifest_path = artifact_path(args.model, quant).with_suffix('.json')
    manifest = json.loads(manifest_path.read_text())
    gguf = artifact_path(args.model, quant)
    if bundle['datasets'] != lock or manifest['datasets'] != lock:
        raise ValueError('Prompt/model/data lock mismatch')
    if bundle['source'] != manifest['source'] or bundle['model_key'] != args.model:
        raise ValueError('Wrong model in prompt bundle or GGUF manifest')
    if sha256(gguf) != manifest['gguf_sha256'] or sha256(bundle_path) != manifest['prompt_sha256']:
        raise ValueError('GGUF or prompt bundle has changed since conversion')
    # Compare saved prompt labels and ordering against every frozen evaluation row.
    expected = eval_rows()
    clean = lambda r: {k:v for k,v in r.items() if k not in ('prompt', 'ids')}
    if [clean(r) for r in bundle['rows']] != expected:
        raise ValueError('Saved evaluation rows differ from the frozen data')
    rows = [r for r in bundle['rows'] if args.eval_set == 'all' or r['dataset'] == args.eval_set]
    if args.limit:
        rows = rows[:args.limit]
    props = request(args.port, 'props')
    if not props.get('model_path') or Path(props['model_path']).resolve() != gguf.resolve():
        raise ValueError('Server /props does not confirm the requested model path')
    if props.get('default_generation_settings', {}).get('params', {}).get('grammar'):
        raise ValueError('Start the server without a global grammar; this benchmark supplies it per request')
    for row in rows:
        tokens = request(args.port, 'tokenize', {'content': row['prompt'], 'add_special': False, 'parse_special': True})['tokens']
        ids = [t['id'] if isinstance(t, dict) else t for t in tokens]
        if ids != row['ids']:
            raise ValueError('HF/GGUF prompt token IDs differ; stop before scoring')
    grammar_path = ROOT / 'gguf' / f'{args.model}-v3.gbnf'
    def generate(row, numeric_probe=False, diagnostics=False):
        payload = {'prompt': row['ids'], 'n_predict': bundle['max_new_tokens'],
                   'temperature': 0.0, 'top_k': 1, 'repeat_penalty': 1.0,
                   'presence_penalty': 0.0, 'frequency_penalty': 0.0,
                   'cache_prompt': getattr(args, 'cache_prefix', False), 'stop': bundle['stop']}
        if diagnostics or numeric_probe:
            payload.update(n_probs=1, post_sampling_probs=False)
        if numeric_probe:
            payload['n_predict'] = 1
        if args.grammar and not numeric_probe:
            payload['grammar'] = grammar_path.read_text()
        result = request(args.port, 'completion', payload)
        if diagnostics or numeric_probe:
            check_generation(result)
        return result
    # Test raw numeric health before grammar can fail on corrupted logits.
    generate(rows[0], numeric_probe=True)
    generate(rows[0], diagnostics=True)
    preparation = None
    if getattr(args, 'cache_prefix', False):
        template = json.loads((ROOT / 'runtime' / f'{args.model}.json').read_text())
        if any(r['prompt'] != template['prefix'] + r['text'] + template['suffix'] for r in rows):
            raise ValueError('Runtime prefix differs from saved evaluation prompts')
        tokens = request(args.port, 'tokenize', {'content': template['prefix'], 'add_special': False, 'parse_special': True})['tokens']
        prefix = [t['id'] if isinstance(t, dict) else t for t in tokens]
        if any(r['ids'][:len(prefix)] != prefix for r in rows):
            raise ValueError('Prefix token boundary mismatch')
        start = time.perf_counter()
        # Pinned server samples one disposable token; no user command is included.
        response = request(args.port, 'completion', {'prompt': prefix, 'n_predict': 1,
            'temperature': 0, 'top_k': 1, 'cache_prompt': False})
        preparation = {'seconds': time.perf_counter() - start, 'prefix_tokens': len(prefix),
                       'timings': response.get('timings', {})}
    examples = []
    diagnostic_rechecks = 0
    for i, row in enumerate(rows):
        start = time.perf_counter()
        res = generate(row)
        duration = time.perf_counter() - start
        examples.append(scored_row(clean(row), res.get('content', ''), duration, args.model,
                        timings=res.get('timings', {}), stop_type=res.get('stop_type'),
                        hit_token_limit=res.get('stop_type') == 'limit' or res.get('tokens_predicted', 0) >= bundle['max_new_tokens'],
                        stopped_limit=res.get('stopped_limit'),
                        truncated=res.get('truncated')))
        if not examples[-1]['schema_valid']:
            # Recheck malformed output outside the timed request. Finite but
            # incorrect output remains an ordinary scored error. Numerical
            # corruption stops the benchmark instead of becoming a score.
            generate(row, diagnostics=True)
            diagnostic_rechecks += 1
        print(f'{i+1}/{len(rows)} correct={examples[-1]["exact_match"]} seconds={duration:.3f}', flush=True)
    summary = {**summarize(examples), 'schema_version': 3, 'model_key': args.model,
               'label': args.label, 'base': False, 'backend': 'llama.cpp', 'quantization': quant,
               'decoding': 'grammar-greedy' if args.grammar else 'unconstrained-greedy',
               'grammar_sha256': sha256(grammar_path) if args.grammar else None,
               'datasets': lock, 'source': manifest['source'], 'batch_size': 1,
               'eval_set': args.eval_set, 'partial_test': bool(args.limit),
               'cache_prompt': getattr(args, 'cache_prefix', False), 'prefix_preparation': preparation,
               'max_new_tokens': bundle['max_new_tokens'], 'gguf_bytes': gguf.stat().st_size,
               'gguf_sha256': manifest['gguf_sha256'], 'server_build': props.get('build_info'),
               'server_props': props, 'server_port': args.port, 'requested_gpu_layers': None if args.external_server else args.gpu_layers,
               'requested_threads': None if args.external_server else args.threads,
               'requested_ubatch_size': None if args.external_server else args.ubatch_size,
               'numeric_check': NUMERIC_CHECK, 'diagnostic_rechecks': diagnostic_rechecks,
               'probability_diagnostics': {'timed_requests': False, 'diagnostic_n_probs': 1},
               'latency_scope': 'local HTTP round trip including generation; prompt tokenization excluded; one warmup; numeric probe and invalid-output diagnostic rechecks excluded'}
    if quant != 'Q8_0':
        PROFILE = {'context': 4096, 'cache_type_k': 'f16', 'cache_type_v': 'f16',
                   'cache_prompt': getattr(args, 'cache_prefix', False), 'batch_size': 1}
        summary.update(study_profile={**PROFILE,
                           'gpu_layers': None if args.external_server else args.gpu_layers,
                           'threads': None if args.external_server else args.threads},
                       server_binary_sha256=None if args.external_server else sha256(SERVER),
                       input_fp32_sha256=manifest['input_fp32_sha256'],
                       tensor_inventory=manifest.get('tensor_inventory'))
    atomic_json(args.result_path, {'summary': summary, 'examples': examples})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('server_props','datasets')}, indent=2))
    print('Saved', args.result_path)


def select_server_port(requested=None):
    """Use an OS-selected port by default; reject live listeners on explicit ports.

    On Unix, SO_REUSEADDR permits closed connections in TIME_WAIT and listen()
    rejects live listeners. Windows requires SO_EXCLUSIVEADDRUSE instead: reuse
    can otherwise bind over another server. A recently closed explicit Windows
    port may remain unavailable; automatic ports avoid that wait.
    The reservation ends before llama-server starts;
    startup failure and model-path checks remain necessary for that small race.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        option = getattr(socket, 'SO_EXCLUSIVEADDRUSE', socket.SO_REUSEADDR)
        sock.setsockopt(socket.SOL_SOCKET, option, 1)
        try:
            sock.bind(('127.0.0.1', requested if requested is not None else 0))
            sock.listen(1)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot reserve local port {requested}: {exc}. "
                "Omit --port to choose an available port automatically; no existing server was stopped."
            ) from exc
        return sock.getsockname()[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=MODELS)
    p.add_argument('--label', required=True)
    p.add_argument('--quantization', type=str.upper, choices=['Q8_0', *QUANTS], default='Q8_0')
    p.add_argument('--eval-set', default='all', choices=['all'] + EVAL_FILES)
    p.add_argument('--grammar', action='store_true')
    p.add_argument('--cache-prefix', action='store_true', help='Prepare and reuse fixed instructions; measured benefit verified for Granite')
    p.add_argument('--limit', type=int)
    p.add_argument('--port', type=int, default=None,
                   help='Automatic mode: choose an available port if omitted. External mode: defaults to 8081.')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--gpu-layers', type=int, default=99)
    p.add_argument('--ubatch-size', type=int, default=None,
                   help='Prompt microbatch size; defaults to 8 for Granite on Mac GPU, 512 otherwise.')
    p.add_argument('--external-server', action='store_true')
    args = p.parse_args()
    if args.ubatch_size is None:
        args.ubatch_size = default_ubatch(args.model, args.gpu_layers)
    if not 1 <= args.ubatch_size <= 2048:
        p.error('--ubatch-size must be between 1 and 2048')
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.label):
        p.error('Use letters, digits, hyphens and underscores for --label')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    if args.threads < 1 or (args.port is not None and not 1 <= args.port <= 65535) or args.gpu_layers < 0:
        p.error('Invalid threads, port, or GPU layers')
    suffix = f'-partial{args.limit}' if args.limit else ''
    result_folder = ROOT / 'results' if args.quantization == 'Q8_0' else ROOT / 'results/quantization'
    args.result_path = result_folder / f'{args.label}{suffix}.json'
    if args.result_path.exists():
        raise SystemExit(f'Result exists: {args.result_path}. Choose a new label.')
    if args.external_server:
        args.port = 8081 if args.port is None else args.port
        benchmark(args)
        return
    llama_version()
    args.port = select_server_port(args.port)
    print(f'Starting private benchmark server on 127.0.0.1:{args.port}; prompt microbatch={args.ubatch_size}', flush=True)
    logpath = result_folder / f'{args.label}{suffix}-server.log'
    logpath.parent.mkdir(parents=True, exist_ok=True)
    model = artifact_path(args.model, args.quantization)
    if not model.exists():
        raise SystemExit(f'Missing {model}; train, merge and convert first.')
    command = [str(SERVER), '-m', str(model), '--host', '127.0.0.1', '--port', str(args.port),
               '-c', '4096', '-np', '1', '-ngl', str(args.gpu_layers), '-t', str(args.threads), '--special',
               '-b', str(max(2048, args.ubatch_size)), '-ub', str(args.ubatch_size),
               '-ctk', 'f16', '-ctv', 'f16']
    with logpath.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited; see {logpath}')
                try:
                    request(args.port, 'health')
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'Server not ready; see {logpath}')
                    time.sleep(.5)
            benchmark(args)
        except Exception as exc:
            raise RuntimeError(f'Benchmark failed: {exc}\nServer log: {logpath}') from exc
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == '__main__':
    main()
