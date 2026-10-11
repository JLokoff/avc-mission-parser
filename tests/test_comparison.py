"""Regression checks for benchmark integrity. No model downloads or training."""
import json
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import comparison_common as c
from build_grammars import grammar


class ComparisonTests(unittest.TestCase):
    def test_frozen_data_and_oversampling(self):
        lock = c.verify_data()
        self.assertEqual(lock['train_set.jsonl']['count'], 2932)
        self.assertEqual(len(c.eval_rows()), 792)
        self.assertEqual(len({' '.join(r['text'].lower().split()) for r in c.read_rows('train_set.jsonl')}), 1551)

    def test_native_roundtrip_all_labels(self):
        for name in ['train_set.jsonl'] + c.EVAL_FILES:
            for row in c.read_rows(name):
                for key in c.MODELS:
                    self.assertEqual(c.parse(key, c.completion(key, row['expected'])), row['expected'])

    def test_wrong_function_and_multiple_calls_rejected(self):
        row = c.eval_rows()[0]
        for key in c.MODELS:
            text = c.completion(key, row['expected'])
            self.assertTrue(all(v is None for v in c.parse(key, text.replace('set_leader_intent', 'other_tool')).values()))
            self.assertTrue(all(v is None for v in c.parse(key, text + text).values()))
            self.assertTrue(all(v is None for v in c.parse(key, 'Explanation: ' + text).values()))

    def test_duplicate_fields_rejected(self):
        examples = {
            'functiongemma': '<start_function_call>call:set_leader_intent{action:<escape>abort<escape>,action:<escape>pause<escape>}<end_function_call>',
            'granite350': '<tool_call>{"name":"set_leader_intent","arguments":{"action":"abort","action":"pause"}}</tool_call>',
            'lfm350': "<|tool_call_start|>[set_leader_intent(action='abort', action='pause')]<|tool_call_end|>",
        }
        for key,text in examples.items():
            self.assertTrue(all(v is None for v in c.parse(key,text).values()))

    def test_missing_invalid_and_cross_field_values_count_as_errors(self):
        row = c.eval_rows()[0]
        for key in c.MODELS:
            bad = {**row['expected'], 'action': 'fly_somewhere'}
            result = c.scored_row(row, c.completion(key,bad), 1, key)
            self.assertFalse(result['schema_valid'])
            self.assertFalse(result['exact_match'])
            bad = {**row['expected'], 'stay_region': 'none', 'until_region': 'E'}
            self.assertFalse(c.scored_row(row,c.completion(key,bad),1,key)['cross_field_valid'])
            self.assertFalse(c.scored_row(row,'',1,key)['schema_valid'])

    def test_metrics_and_dataset_slices(self):
        row = c.eval_rows()[0]
        samples = [c.scored_row(row,c.completion('granite350',row['expected']),1,'granite350'), c.scored_row(row,'invalid',3,'granite350')]
        result = c.summarize(samples)
        self.assertEqual(result['exact_match_accuracy'],.5)
        self.assertEqual(result['latency_s']['median'],2)
        self.assertEqual(result['latency_s']['p95'],3)
        self.assertEqual(result['invalid_output_count'],1)
        self.assertEqual(result['by_dataset']['eval_set.jsonl']['n'],2)

    def test_grammar_files_have_not_drifted(self):
        for key in c.MODELS:
            self.assertEqual((c.ROOT/'gguf'/f'{key}-v3.gbnf').read_text(),grammar(key))

    def test_data_edits_refused(self):
        with patch.object(c, 'sha256', return_value='changed'):
            with self.assertRaisesRegex(ValueError,'Dataset changed'):
                c.verify_data()

    def test_training_not_accepted_as_evaluation(self):
        with self.assertRaises(ValueError):
            c.eval_rows('train_set.jsonl')


if __name__ == '__main__':
    unittest.main(verbosity=2)
