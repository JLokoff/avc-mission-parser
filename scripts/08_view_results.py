"""
Read the published reference results without unpacking a folder full of logs. By default
this lists each saved run and its score and speed. Select a run to see its full details,
or add --wrong-only to inspect the commands that model answered incorrectly.
"""
# Previously: read_evidence.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
import argparse
import gzip
import json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', help='Exact evidence ID shown in the listing')
    p.add_argument('--wrong-only', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    with gzip.open(root / 'evidence/runs.json.gz', 'rt', encoding='utf-8') as f:
        runs = json.load(f)['runs']
    if a.run:
        selected = [r for r in runs if r['source'] == a.run]
        if not selected:
            p.error('Unknown run ID; omit --run to list them')
        r = selected[0]
        if a.wrong_only:
            print(json.dumps([x for x in r['examples'] if not x['exact_match']], indent=2))
        else:
            print(json.dumps(r, indent=2))
    else:
        for r in runs:
            s = r['summary']
            print(f'{r["source"]}: {s["n_correct"]}/{s["n_examples"]}, median={s["latency_s"]["median"]:.3f}s')

if __name__ == '__main__':
    main()
