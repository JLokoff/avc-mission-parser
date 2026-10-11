"""Exercise the scoring path with simulated server responses; no model inference."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

spec = importlib.util.spec_from_file_location('benchmark_numeric', Path(__file__).resolve().parents[1] / 'scripts/04_benchmark_gguf.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class BenchmarkNumericTests(unittest.TestCase):
    def run_case(self, recheck_probability, expected_error):
        row = b.eval_rows()[0]
        bundle = {'rows': [{**row, 'prompt': 'fixture', 'ids': [1, 2]}], 'datasets': {},
                  'source': {}, 'model_key': 'granite350', 'max_new_tokens': 256, 'stop': []}
        calls = []

        def request(port, route, payload=None):
            if route == 'props':
                return {'model_path': str(b.ROOT/'gguf/granite350-q8_0.gguf')}
            if route == 'tokenize':
                return {'tokens': bundle['rows'][0]['ids']}
            calls.append(payload.copy())
            if 'n_probs' not in payload:
                return {'content': '!!!', 'tokens_predicted': 3}
            value = recheck_probability if len(calls) == 4 else -0.1
            return {'content': '!!!', 'completion_probabilities': [{'top_logprobs': [{'logprob': value}]}]}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'result.json'
            root = Path(tmp)
            (root / 'prompts').mkdir()
            (root / 'gguf').mkdir()
            bundle_path = root / 'prompts/granite350.json'
            bundle_path.write_text(json.dumps(bundle))
            model = root / 'gguf/granite350-q8_0.gguf'
            model.write_bytes(b'GGUF-test-fixture')
            model.with_suffix('.json').write_text(json.dumps({'datasets': {}, 'source': {},
                'gguf_sha256': b.sha256(model), 'prompt_sha256': b.sha256(bundle_path)}))
            args = SimpleNamespace(model='granite350', eval_set='all', limit=1, port=1,
                grammar=False, label='test', external_server=False, gpu_layers=99,
                threads=8, ubatch_size=8, result_path=path)
            with patch.object(b, 'request', request), patch('builtins.print'), \
                 patch.object(b, 'ROOT', root), patch.object(b, 'artifact_path', return_value=model), \
                 patch.object(b, 'verify_data', return_value={}), patch.object(b, 'eval_rows', return_value=[row]):
                if expected_error:
                    with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
                        b.benchmark(args)
                    self.assertFalse(path.exists())
                else:
                    b.benchmark(args)
                    result = json.loads(path.read_text())
                    self.assertEqual(result['summary']['n_correct'], 0)
                    self.assertEqual(result['summary']['diagnostic_rechecks'], 1)
            self.assertEqual(len(calls), 4)
            self.assertNotIn('n_probs', calls[2])
            self.assertEqual(calls[3]['n_probs'], 1)

    def test_numerically_broken_invalid_output_never_saves_score(self):
        self.run_case(None, expected_error=True)

    def test_finite_invalid_output_is_scored_as_model_error(self):
        self.run_case(-0.5, expected_error=False)

    def test_http_failure_includes_server_error_body(self):
        error = urllib.error.HTTPError('http://localhost/completion', 500, 'error', {}, io.BytesIO(b'grammar stack failed'))
        with patch.object(b.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 500: grammar stack failed'):
                b.request(1, 'completion', {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
