"""
Builds the allowed answer-format rules for each model. These rules keep generated
answers within the field names and values in Schema V3; they do not choose the correct
answer for the model. The conversion and inference code use this automatically.
"""
# Previously: build_grammars.py
import json
from model_common import ROOT, MODELS, FIELDS, ENUMS, completion, stop_tokens


def grammar(key):
    markers = {f: f"AVCSLOT{i}" for i, f in enumerate(FIELDS)}
    template = completion(key, markers).strip()
    for end in stop_tokens(key):
        if template.endswith(end):
            template = template[:-len(end)].rstrip()
    pieces = []
    for f in FIELDS:
        before, template = template.split(markers[f], 1)
        pieces.extend([json.dumps(before), f.replace('_', '-')])
    pieces.append(json.dumps(template))
    rules = ['# Generated from schema.py; native wrapper only differs between models.',
             '# Vocabulary restriction only. Cross-field consistency is scored separately.',
             'root ::= ' + ' '.join(pieces)]
    rules += [f.replace('_', '-') + ' ::= ' + ' | '.join(json.dumps(v) for v in ENUMS[f]) for f in FIELDS]
    return '\n'.join(rules) + '\n'


def main():
    (ROOT / 'gguf').mkdir(exist_ok=True)
    for key in MODELS:
        path = ROOT / 'gguf' / f'{key}-v3.gbnf'
        path.write_text(grammar(key))
        print(path)


if __name__ == '__main__':
    main()
