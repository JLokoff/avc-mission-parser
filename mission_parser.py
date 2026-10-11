"""Small, local, command-to-intent Python interface. No PyTorch dependency.

Use one long-lived MissionParser context; it owns and closes its llama-server.
It parses commands only. It never invokes flight controls or mission actions.
"""
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'core'))
from model_common import MODELS, ENUMS, FIELDS, parse, sha256
from output_grammar import grammar
from runtime_checks import check_generation, default_ubatch
from artifacts import binary
from artifacts import artifact_path


def validate_intent(intent):
    if not isinstance(intent, dict) or set(intent) != set(FIELDS):
        raise ValueError('Expected exactly the six V3 fields')
    for field in FIELDS:
        if not isinstance(intent[field], str) or intent[field] not in ENUMS[field]:
            raise ValueError(f'Invalid or missing {field}')
    if intent['until_region'] != 'none' and intent['stay_region'] == 'none':
        raise ValueError('until_region requires stay_region')
    return intent


def validate_command(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 2048:
        raise ValueError('text must be a nonempty string of at most 2048 characters')
    if '<' in text or '>' in text or '\x00' in text:
        raise ValueError('Command contains reserved template delimiters')
    return text


class MissionParser:
    def __init__(self, model='granite350', quantization='Q8_0', gguf=None,
                 gpu_layers=99, threads=8, cache_prefix=None, server_binary=None):
        if model not in MODELS or quantization not in ('Q8_0', 'Q6_K', 'Q4_K_M'):
            raise ValueError('Unsupported model or quantization')
        if threads < 1 or gpu_layers < 0:
            raise ValueError('threads must be positive; gpu_layers nonnegative')
        self.model, self.quantization = model, quantization
        self.gguf = Path(gguf).resolve() if gguf else artifact_path(model, quantization)
        self.binary = Path(server_binary) if server_binary else binary('llama-server')
        self.threads, self.gpu_layers = threads, gpu_layers
        self.cache_prefix = (model == 'granite350') if cache_prefix is None else bool(cache_prefix)
        self.template = json.loads((ROOT / 'runtime' / f'{model}.json').read_text())
        self.lock = threading.RLock()
        self.process = None
        self.log = None
        self.port = None
        self.ready = False

    def request(self, route, payload=None):
        req = urllib.request.Request(f'http://127.0.0.1:{self.port}/{route}',
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=2 if route == 'health' else 120) as response:
            return json.load(response)

    def tokenize(self, text):
        result = self.request('tokenize', {'content': text, 'add_special': False, 'parse_special': True})
        return [t['id'] if isinstance(t, dict) else t for t in result['tokens']]

    def start(self):
        with self.lock:
            if self.ready:
                return self
            if not self.binary.is_file():
                raise FileNotFoundError('Build llama.cpp first: python scripts/00_prepare_models.py build --backend cpu|metal|cuda')
            manifest = json.loads(self.gguf.with_suffix('.json').read_text())
            if (manifest['source'] != self.template['source'] or manifest['quantization'] != self.quantization
                    or manifest['gguf_sha256'] != sha256(self.gguf)):
                raise ValueError('Model identity, quantization, or GGUF hash mismatch')
            if sha256(ROOT / 'core/schema.py') != self.template['schema_sha256']:
                raise ValueError('Runtime prompt belongs to a different schema')
            with socket.socket() as s:
                s.bind(('127.0.0.1', 0))
                self.port = s.getsockname()[1]
            logs = ROOT / 'logs'
            logs.mkdir(exist_ok=True)
            self.log_path = logs / f'parser-{self.model}-{time.time_ns()}.log'
            self.log = self.log_path.open('w', encoding='utf-8')
            cmd = [str(self.binary), '-m', str(self.gguf), '--host', '127.0.0.1', '--port', str(self.port),
                   '-c', '4096', '-np', '1', '-ngl', str(self.gpu_layers), '-t', str(self.threads), '--special',
                   '-b', '2048', '-ub', str(default_ubatch(self.model, self.gpu_layers)), '-ctk', 'f16', '-ctv', 'f16']
            try:
                self.process = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 180
                while True:
                    if self.process.poll() is not None:
                        raise RuntimeError(f'llama-server exited; inspect {self.log_path}')
                    try:
                        self.request('health')
                        break
                    except (OSError, TimeoutError):
                        if time.monotonic() > deadline:
                            raise RuntimeError('llama-server startup timed out')
                        time.sleep(.2)
                props = self.request('props')
                if Path(props.get('model_path', '')).resolve() != self.gguf.resolve():
                    raise RuntimeError('Unexpected server model')
                self.prefix_ids = self.tokenize(self.template['prefix'])
                probe = self.payload(self.prefix_ids, False)
                probe.pop('grammar')
                probe.update(n_predict=1, n_probs=1, post_sampling_probs=False)
                check_generation(self.request('completion', probe))
                self.ready = True
                self.preparation = self.prepare_prefix() if self.cache_prefix else None
                return self
            except BaseException:
                self.close()
                raise

    def payload(self, ids, cache):
        return {'prompt': ids, 'id_slot': 0, 'n_predict': self.template['max_new_tokens'],
                'temperature': 0, 'top_k': 1, 'repeat_penalty': 1., 'presence_penalty': 0.,
                'frequency_penalty': 0., 'cache_prompt': cache, 'stop': self.template['stop'],
                'grammar': grammar(self.model)}

    def prepare_prefix(self):
        """Recompute the fixed prefix after startup or on explicit request."""
        with self.lock:
            if not self.ready:
                raise RuntimeError('Parser has not started')
            p = self.payload(self.prefix_ids, False)
            p.pop('grammar')
            # This pinned server samples one token even with n_predict=0.
            # Discard it. Every later request supplies the complete new prompt.
            p['n_predict'] = 1
            start = time.perf_counter()
            response = self.request('completion', p)
            return {'seconds': time.perf_counter() - start, 'prefix_tokens': len(self.prefix_ids),
                    'timings': response.get('timings', {})}

    def parse(self, text):
        validate_command(text)
        with self.lock:
            if not self.ready or self.process.poll() is not None:
                raise RuntimeError('Parser is not running')
            started = time.perf_counter()
            rendered = self.template['prefix'] + text + self.template['suffix']
            ids = self.tokenize(rendered)
            if len(ids) + self.template['max_new_tokens'] > 4096:
                raise ValueError('Command exceeds the model context budget')
            if ids[:len(self.prefix_ids)] != self.prefix_ids:
                raise ValueError('Unexpected prefix token boundary')
            result = self.request('completion', self.payload(ids, self.cache_prefix))
            raw = result.get('content', '')
            if (result.get('truncated') or result.get('stopped_limit') or result.get('stop_type') == 'limit'
                    or result.get('tokens_predicted', 0) >= self.template['max_new_tokens']):
                return {'ok': False, 'intent': None, 'error': 'incomplete_output', 'raw_output': raw}
            try:
                intent = validate_intent(parse(self.model, raw))
            except ValueError as exc:
                return {'ok': False, 'intent': None, 'error': 'invalid_output', 'detail': str(exc), 'raw_output': raw}
            return {'ok': True, 'schema_version': 3, 'model': self.model, 'intent': intent,
                    'latency_s': time.perf_counter() - started, 'timings': result.get('timings', {}),
                    'cache_prefix_requested': self.cache_prefix}

    def close(self):
        with self.lock:
            self.ready = False
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            if self.log is not None:
                self.log.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
