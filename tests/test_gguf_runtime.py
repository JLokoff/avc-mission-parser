"""Regressions for corrupted Metal output and the Granite compatibility profile."""
import unittest
from unittest.mock import patch
from gguf_runtime import check_generation, default_ubatch


def response(value, selected=None):
    return {'completion_probabilities': [{'logprob': selected, 'top_logprobs': [{'logprob': value}]}]}


class RuntimeTests(unittest.TestCase):
    def test_nonfinite_logits_are_rejected(self):
        for value in (None, float('nan'), float('inf'), float('-inf'), 'NaN', True):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
                check_generation(response(value))

    def test_forced_token_underflow_does_not_reject_valid_distribution(self):
        check_generation(response(-0.2, selected=None))

    def test_bad_semantic_answer_is_not_a_backend_failure(self):
        check_generation({**response(-0.5), 'content': 'incorrect answer'})

    def test_missing_diagnostics_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'diagnostics'):
            check_generation({'content': '!!!'})

    def test_error_payload_is_not_scored(self):
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            check_generation({'error': 'grammar failed'})

    def test_only_mac_granite_gpu_changes_default(self):
        with patch('gguf_runtime.sys.platform', 'darwin'):
            self.assertEqual(default_ubatch('granite350', 99), 8)
            self.assertEqual(default_ubatch('granite350', 0), 512)
            self.assertEqual(default_ubatch('functiongemma', 99), 512)
            self.assertEqual(default_ubatch('lfm350', 99), 512)
        with patch('gguf_runtime.sys.platform', 'linux'):
            self.assertEqual(default_ubatch('granite350', 99), 512)


if __name__ == '__main__':
    unittest.main(verbosity=2)
