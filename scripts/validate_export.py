"""Developer smoke-only integration check: adapter merge, conversion, native grammar.

Requires a completed --smoke run. Temporary model files are removed on success or
failure; only validation logs/metadata persist. No comparison score is produced.
"""
import argparse
import importlib.util
import json
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from peft import PeftModel
from comparison_common import ROOT, MODELS, eval_rows, parse, ENUMS, write_json, prompt, sha256
from compare import load_model, tokenizer_and_formatter, save_export
from gguf_workflow import require_tools, prepare_model_config_for_converter, prepare_tokenizer_for_converter, CONVERTER_PYTHON, CONVERTER, SERVER
from gguf_runtime import default_ubatch, check_generation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--gpu-layers', type=int, default=99)
    args=p.parse_args()
    report_dir=ROOT/'validation'; report_dir.mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('gguf_benchmark', ROOT/'scripts/04_benchmark_gguf.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    run=json.loads((ROOT/'checkpoints'/f'{args.model}-smoke'/'run.json').read_text())
    if not run.get('completed') or not run.get('smoke_only'):
        raise ValueError('This check accepts only completed smoke adapters')
    version=require_tools()
    base=ROOT/'models'/args.model
    tokenizer,formatter=tokenizer_and_formatter(args.model,base)
    model=load_model(base,SimpleNamespace(device='cpu',threads=8))
    model=PeftModel.from_pretrained(model,ROOT/'checkpoints'/f'{args.model}-smoke').eval()
    inputs=tokenizer(prompt(args.model,formatter,eval_rows()[0]['text']),add_special_tokens=False,return_tensors='pt')
    with torch.inference_mode():
        before=model(**inputs,logits_to_keep=1,use_cache=False).logits
        model=model.merge_and_unload(safe_merge=True).eval()
        after=model(**inputs,logits_to_keep=1,use_cache=False).logits
    maxdiff=(before-after).abs().max().item()
    if not torch.allclose(before,after,atol=.003,rtol=.001):
        raise AssertionError(f'Adapter merge changed logits: max_abs_diff={maxdiff}')
    with tempfile.TemporaryDirectory(prefix='avc-v3-export-') as scratch:
        scratch=Path(scratch)
        merged=scratch/'merged'; bundle_path=scratch/'prompts.json'; gguf=scratch/'smoke-q8_0.gguf'
        save_export(args.model,model,tokenizer,formatter,merged,bundle_path,run)
        del model,before,after
        prepare_model_config_for_converter(args.model,merged)
        prepare_tokenizer_for_converter(args.model,merged)
        logpath=report_dir/f'{args.model}-conversion.log'
        with logpath.open('w') as log:
            subprocess.run([str(CONVERTER_PYTHON),str(CONVERTER),str(merged),'--outfile',str(gguf),'--outtype','q8_0'],stdout=log,stderr=subprocess.STDOUT,check=True)
        bundle=json.loads(bundle_path.read_text())
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        with (report_dir/f'{args.model}-server.log').open('w') as log:
            process=subprocess.Popen([str(SERVER),'-m',str(gguf),'--host','127.0.0.1','--port',str(port),'-c','4096','-np','1','-ngl',str(args.gpu_layers),'-t','8','--special',
                '-b','2048','-ub',str(default_ubatch(args.model,args.gpu_layers))],stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+90
                while True:
                    if process.poll() is not None: raise RuntimeError('Smoke GGUF server exited')
                    try: mod.request(port,'health'); break
                    except Exception:
                        if time.monotonic()>deadline: raise
                        time.sleep(.5)
                props=mod.request(port,'props')
                if Path(props.get('model_path','')).resolve()!=gguf.resolve():
                    raise AssertionError('Server does not expose expected model path')
                for row in bundle['rows']:
                    tokens=mod.request(port,'tokenize',{'content':row['prompt'],'add_special':False,'parse_special':True})['tokens']
                    ids=[x['id'] if isinstance(x,dict) else x for x in tokens]
                    if ids!=row['ids']: raise AssertionError('GGUF token IDs differ')
                # Ordinary generation must run, but a two-step adapter need not be accurate.
                payload={'prompt':bundle['rows'][0]['ids'],'n_predict':bundle['max_new_tokens'],'temperature':0,'top_k':1,'repeat_penalty':1,'cache_prompt':False,'stop':bundle['stop']}
                check_generation(mod.request(port,'completion',{**payload,'n_predict':1,'n_probs':1,'post_sampling_probs':False}))
                plain=mod.request(port,'completion',payload)
                payload['grammar']=(ROOT/'gguf'/f'{args.model}-v3.gbnf').read_text()
                constrained=mod.request(port,'completion',payload)
                parsed=parse(args.model,constrained['content'])
                if not all(parsed[f] in ENUMS[f] for f in ENUMS):
                    raise AssertionError(f'Grammar response did not parse: {constrained}')
                write_json(report_dir/f'{args.model}-export-check.json',{'smoke_only':True,'not_an_accuracy_benchmark':True,'model':args.model,'merge_max_abs_logit_difference':maxdiff,'gguf_bytes':gguf.stat().st_size,'all_792_prompt_token_ids_match':True,'grammar_response':constrained['content'],'plain_response':plain['content'],'grammar_schema_valid':True,'llama_cpp':version})
            finally:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill();process.wait(timeout=10)
    print('Smoke adapter merge / Q8 conversion / all prompt IDs / native grammar passed:',args.model)


if __name__=='__main__': main()
