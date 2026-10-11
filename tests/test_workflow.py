"""
Check that every numbered command is usable and that the batch runner sends each model
to the correct step. These tests also check format selection and stop-on-error behavior
without downloading, training, or converting a real model.
"""
import ast
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('workflow', ROOT / 'scripts/07_run_workflow.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

class WorkflowTests(unittest.TestCase):
    def test_all_python_files_parse(self):
        for folder in ('core', 'scripts', 'tests', 'examples'):
            for path in (ROOT / folder).glob('*.py'):
                with self.subTest(path=path.name):
                    ast.parse(path.read_text(encoding='utf-8'))

    def test_numbered_scripts_offer_help_without_models(self):
        scripts = sorted((ROOT / 'scripts').glob('*.py'))
        self.assertEqual([p.name[:2] for p in scripts], [f'{n:02d}' for n in range(10)])
        for path in scripts[:9]:
            with self.subTest(script=path.name):
                result = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_every_stage_routes_all_models(self):
        expected = {'download': '00_', 'check': '00_', 'baseline': '01_',
                    'train': '02_', 'evaluate': '03_', 'quantize': '04_', 'benchmark': '05_'}
        for stage, prefix in expected.items():
            with self.subTest(stage=stage), patch.object(sys, 'argv', ['workflow', stage]), patch.object(w.subprocess, 'run') as run:
                w.main()
                jobs = [c.args[0] for c in run.call_args_list]
                self.assertEqual(len(jobs), 9 if stage == 'benchmark' else 3)
                self.assertEqual({j[j.index('--model')+1] for j in jobs}, set(w.MODELS))
                self.assertTrue(all(Path(j[2]).is_file() and Path(j[2]).name.startswith(prefix) for j in jobs))
                if stage == 'baseline':
                    self.assertTrue(all(j[j.index('--device')+1] == 'cpu' for j in jobs))
                if stage == 'quantize':
                    self.assertTrue(all(j[j.index('--quants')+1:] == ['Q8_0', 'Q6_K', 'Q4_K_M'] for j in jobs))
                if stage == 'benchmark':
                    self.assertEqual({j[j.index('--quantization')+1] for j in jobs}, {'Q8_0', 'Q6_K', 'Q4_K_M'})

    def test_selected_model_prefix_and_limit_propagate(self):
        for key in w.MODELS:
            with self.subTest(model=key), patch.object(sys, 'argv', ['workflow', 'benchmark', '--models', key, '--quants', 'Q6_K', '--cache-prefix', '--limit', '2']), patch.object(w.subprocess, 'run') as run:
                w.main()
                run.assert_called_once()
                cmd = run.call_args.args[0]
                self.assertIn('--cache-prefix', cmd)
                self.assertEqual(cmd[cmd.index('--model')+1], key)
                self.assertEqual(cmd[cmd.index('--limit')+1], '2')

    def test_failure_stops_remaining_jobs(self):
        with patch.object(sys, 'argv', ['workflow', 'benchmark']), patch.object(w.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'fixture')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                w.main()
            run.assert_called_once()
