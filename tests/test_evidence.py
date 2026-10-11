import gzip
import json
import unittest
from pathlib import Path
from comparison_common import summarize

class EvidenceTests(unittest.TestCase):
    def test_published_summaries_match_archived_predictions(self):
        root = Path(__file__).resolve().parents[1]
        with gzip.open(root / 'evidence/runs.json.gz', 'rt', encoding='utf-8') as f:
            archive = json.load(f)
        summaries = {s['evidence_id']: s for s in json.loads((root / 'evidence/summary.json').read_text())}
        self.assertEqual(len(archive['runs']), len(summaries))
        for run in archive['runs']:
            s = summaries[run['source']]
            self.assertEqual(len(run['examples']), 792)
            calculated = summarize(run['examples'])
            for field in ['n_correct', 'exact_match_accuracy', 'per_field_accuracy', 'latency_s']:
                self.assertEqual(s[field], calculated[field], (run['source'], field))
