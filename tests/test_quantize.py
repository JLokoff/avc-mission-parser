"""Check source precision and command construction without running conversion."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import quantize as q

class QuantizeTests(unittest.TestCase):
    def test_lower_bits_use_same_fp32_master_never_q8(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'merged/granite350'
            folder.mkdir(parents=True)
            weights = folder / 'model.safetensors'
            weights.write_bytes(b'fp32-fixture')
            training = {'source': {'model_key': 'granite350'}, 'datasets': {}, 'adapter_sha256': 'fixture'}
            (folder / 'training-run.json').write_text(json.dumps(training))
            (folder / 'export-metadata.json').write_text(json.dumps({'dtype': 'float32', 'weight_files': {'model.safetensors': q.sha256(weights)}}))
            (root / 'prompts').mkdir()
            (root / 'prompts/granite350.json').write_text('{}')
            def artifact(key, quant):
                return root / 'gguf' / ('masters' if quant == 'F32' else 'quantization') / f'{key}-{quant.lower()}.gguf'
            (root / 'gguf/quantization').mkdir(parents=True)
            calls = []
            def run(cmd, **kwargs):
                cmd = list(map(str, cmd)); calls.append(cmd)
                if '--outfile' in cmd:
                    Path(cmd[cmd.index('--outfile') + 1]).write_bytes(b'GGUF-fp32-fixture')
                elif cmd[0].endswith('llama-quantize'):
                    Path(cmd[2]).write_bytes(b'GGUF-quantized-fixture')
                return SimpleNamespace(returncode=0)
            with patch.object(q, 'ROOT', root), patch.object(q, 'completed_run', return_value=training), \
                 patch.object(q, 'require_tools', return_value={}), patch.object(q, 'llama_version', return_value={}), \
                 patch.object(q, 'prepare_model_config_for_converter'), patch.object(q, 'prepare_tokenizer_for_converter'), \
                 patch.object(q, 'artifact_path', side_effect=artifact), patch.object(q, 'binary', return_value=Path('llama-quantize')), \
                 patch.object(q, 'sha256', side_effect=lambda p: 'binary' if str(p)=='llama-quantize' else original_sha(p)), \
                 patch.object(q.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12)), \
                 patch.object(q.subprocess, 'run', side_effect=run), \
                 patch.object(sys, 'argv', ['quantize.py', '--model', 'granite350', '--quants', 'Q6_K', 'Q4_K_M']):
                q.main()
            jobs = [c for c in calls if c[0] == 'llama-quantize']
            self.assertEqual([c[3] for c in jobs], ['Q6_K', 'Q4_K_M'])
            self.assertTrue(all(c[1] == str(artifact('granite350', 'F32')) for c in jobs))
            self.assertFalse(any('--allow-requantize' in c for c in calls))
            self.assertTrue(artifact('granite350', 'Q4_K_M').with_suffix('.json').exists())

original_sha = q.sha256
