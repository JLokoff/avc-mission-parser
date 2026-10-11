"""Run with .venv-convert Python. Verify recipe metadata and report actual tensor precision."""
import argparse
import collections
import json
import numpy as np
from gguf import GGUFReader, LlamaFileType

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('path')
p.add_argument('--expect', required=True, choices=['F32','Q6_K','Q5_K_M','Q4_K_M','Q3_K_M'])
a=p.parse_args()
r=GGUFReader(a.path)
types=collections.Counter(t.tensor_type.name for t in r.tensors)
if not types:
    raise SystemExit('GGUF contains no tensors')
field=r.get_field('general.file_type')
actual_filetype=field.contents() if field else None
wanted_filetype=getattr(LlamaFileType,'ALL_F32' if a.expect=='F32' else 'MOSTLY_'+a.expect).value
if actual_filetype != wanted_filetype:
    raise SystemExit(f'GGUF recipe metadata mismatch: expected {wanted_filetype}, found {actual_filetype}')
notes=[]
if a.expect=='F32':
    if set(types)!={'F32'}:
        raise SystemExit(f'Refusing a quantized or rounded source: expected all F32 tensors, found {dict(types)}')
    for t in r.tensors:
        if not np.isfinite(t.data).all():
            raise SystemExit('Nonfinite FP32 source tensor: '+t.name)
    target_present=True
else:
    wanted={'Q6_K':'Q6_K','Q5_K_M':'Q5_K','Q4_K_M':'Q4_K','Q3_K_M':'Q3_K'}[a.expect]
    target_present=bool(types[wanted])
    if not any(name.startswith(('Q','IQ')) for name in types):
        raise SystemExit('Requested quantization produced no quantized tensors')
    if not target_present:
        notes.append(f'{a.expect} recipe contains no {wanted} tensors. Standard llama.cpp shape fallbacks/precision overrides apply; do not describe this as a true {wanted} model.')
params=collections.Counter();total_bytes=0
for t in r.tensors:
    params[t.tensor_type.name]+=int(t.n_elements)
    total_bytes+=int(t.n_bytes)
print(json.dumps({'tensor_count':len(r.tensors),'tensor_types':dict(types),
    'parameters_by_type':dict(params),'tensor_data_bytes':total_bytes,
    'effective_weight_bits':8*total_bytes/sum(params.values()),
    'target_tensor_type_present':target_present,'warnings':notes,
    'requested_format':a.expect,'gguf_file_type':actual_filetype}))
