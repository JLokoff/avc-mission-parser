"""Standing check for schema v3. Run after any builder or schema change:

    python tests/test_schema_v3.py

Guards the three things that have silently broken before:
  1. a data file carrying fields the schema no longer has (the v2 `constraints` key),
  2. the GBNF grammar drifting out of sync with schema.py (field order AND every enum),
  3. until_region set on a row with no stay_region to release -- the one cross-field rule
     the grammar provably cannot express, so nothing but a check can catch it.

ponytail: one script, asserts, no framework. It fails loudly and names the row.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
from schema import FIELDS, FUNCTION_SCHEMA  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENUMS = {f: FUNCTION_SCHEMA["function"]["parameters"]["properties"][f]["enum"]
         for f in FIELDS}


def check_data():
    paths = sorted(glob.glob(os.path.join(ROOT, "data", "*_set.jsonl")))
    assert paths, "no data/*_set.jsonl found"
    total = 0
    for path in paths:
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                exp = json.loads(line)["expected"]
                where = f"{name}:{lineno}"
                assert tuple(exp) == FIELDS, f"{where}: fields {tuple(exp)} != {FIELDS}"
                for field, value in exp.items():
                    assert value in ENUMS[field], f"{where}: {field}={value!r} not in enum"
                if exp["until_region"] != "none":
                    assert exp["stay_region"] != "none", \
                        f"{where}: until_region with no stay_region to release"
                total += 1
    print(f"data: {total} rows across {len(paths)} sets, all fields and enums valid")


def check_grammar():
    from model_common import MODELS
    from output_grammar import grammar
    for key in MODELS:
        path = os.path.join(ROOT, 'gguf', key + '-v3.gbnf')
        assert open(path, encoding='utf-8').read() == grammar(key), 'Grammar drift: ' + key
    print('grammar: all three native grammars match schema.py')


def check_expand():
    from schema import expand_v2
    # the v2 overloading really is undone: the region leaves target_location
    row = expand_v2("unspecified", "avoid_regions", "SW", "collect_target")
    assert row["avoid_region"] == "SW" and row["target_location"] == "unspecified", row
    # and the impossible combinations are refused rather than written out
    for bad, why in [
        (("blue", "none", "NE", "collect_target", "none", "E"), "until without stay"),
        (("blue", "none", "ZZ", "collect_target"), "value outside the enum"),
        (("blue", "avoid_regions", "SW", "collect_target", "none", "none", "NE"),
         "region given twice"),
    ]:
        try:
            expand_v2(*bad)
        except ValueError:
            continue
        raise AssertionError(f"expand_v2 accepted {why}: {bad}")
    print("expand_v2: un-overloads target_location, rejects the 3 impossible shapes")


if __name__ == "__main__":
    check_data()
    check_grammar()
    check_expand()
    print("OK")
