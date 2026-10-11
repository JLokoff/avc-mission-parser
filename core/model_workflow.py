"""
Provides the shared download, model checks, training, evaluation, and merging used by
the numbered steps. Keeping these operations together means all three models receive the
same experiment settings. Normally run a numbered script instead of running this support
file directly.
"""
# Previously: compare.py, adapted from upstream 02_finetune_lora.py and 03_evaluate.py
import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

from model_common import (ROOT, MODELS, FIELDS, verify_data, read_rows, prompt,
                               completion, masked_example, stop_tokens, call_tokens, summarize,
                               scored_row, write_json, sha256, eval_rows, EVAL_FILES, MAX_NEW_TOKENS)


def code_fingerprint():
    return {name: sha256(ROOT / "core" / name) for name in ["schema.py", "model_common.py", "model_workflow.py"]}


def result_path(args):
    host = "mac" if platform.system() == "Darwin" else "pc"
    label = args.label or f"{args.model}-{'base' if args.base else 'finetuned'}-{host}-{args.device}-{args.eval_set.replace('.jsonl', '')}"
    if args.limit:
        label += f"-partial{args.limit}"
    return ROOT / "results" / f"{label}.json"


def metadata():
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "machine": platform.machine(),
              "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")}
    for name in ["torch", "transformers", "peft", "accelerate", "huggingface_hub"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    dest = ROOT / "models" / args.model
    if (dest / "source.json").exists():
        source_path(args.model)
        print("Already prepared:", dest)
        return
    revision = json.loads((ROOT / "model-lock.json").read_text())[args.model]["revision"]
    snapshot_download(MODELS[args.model], revision=revision, local_dir=str(dest),
                      allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.model", "*.txt", "LICENSE", "README.md"])
    write_json(dest / "source.json", {"model_key": args.model, "repo_id": MODELS[args.model], "revision": revision})
    print("Prepared original trainable checkpoint:", dest)


def source_path(key):
    path = ROOT / "models" / key
    if not (path / "source.json").exists():
        raise SystemExit(f"First run: python scripts/00_prepare_models.py download --model {key}")
    wanted = json.loads((ROOT / "model-lock.json").read_text())[key]
    if json.loads((path / "source.json").read_text()) != wanted:
        raise ValueError("Base model revision differs from model-lock.json")
    if not list(path.glob("*.safetensors")):
        raise ValueError(f"Missing base model weights: {path}")
    return path


def tokenizer_and_formatter(key, path):
    from transformers import AutoProcessor, AutoTokenizer
    if key == "functiongemma":
        formatter = AutoProcessor.from_pretrained(path, local_files_only=True)
        tokenizer = formatter.tokenizer if hasattr(formatter, "tokenizer") else formatter
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        formatter = tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, formatter


def runtime(args):
    import torch
    from transformers import set_seed
    torch.set_num_threads(args.threads)
    set_seed(42)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Use --device cpu or follow the optional GPU setup in the guide.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("Apple MPS is unavailable. Check native arm64 Python, macOS, and the Mac PyTorch wheel; see README.md.")
    return torch


def synchronize(torch, device):
    # GPU work is asynchronous; include completed work in every timed interval.
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_model(path, args):
    torch = runtime(args)
    from transformers import AutoModelForCausalLM
    # Keep the same FP32 profile on CPU, CUDA and Apple MPS.
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
               attn_implementation="eager", local_files_only=True).to(args.device)


def preflight(args):
    lock = verify_data()
    print(json.dumps({"datasets": lock, "environment": metadata()}, indent=2))
    torch = runtime(args)
    # A real forward/backward operation checks the chosen backend before downloads/training.
    x = torch.randn(32, 32, device=args.device, requires_grad=True)
    loss = (x @ x).square().mean()
    loss.backward()
    synchronize(torch, args.device)
    assert torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()
    print("Forward/backward device check passed:", args.device)
    if args.model:
        path = source_path(args.model)
        tokenizer, formatter = tokenizer_and_formatter(args.model, path)
        rows = read_rows("train_set.jsonl")
        lengths = []
        for row in rows:
            item = masked_example(tokenizer, prompt(args.model, formatter, row["text"]), completion(args.model, row["expected"]))
            lengths.append(len(item["input_ids"]))
        for token in stop_tokens(args.model):
            assert token in tokenizer.get_vocab(), f"Missing stop token: {token}"
        for token in call_tokens(args.model):
            assert token in tokenizer.get_vocab(), f"Missing call token: {token}"
        for row in eval_rows():
            item = masked_example(tokenizer, prompt(args.model, formatter, row["text"]), completion(args.model, row["expected"]))
            target_length = sum(x != -100 for x in item["labels"])
            if target_length >= MAX_NEW_TOKENS:
                raise ValueError(f"Output budget too small: {target_length}")
        print("All training and evaluation examples tokenize with correct loss masks and sufficient output budget.")
        print("Sequence lengths:", min(lengths), "to", max(lengths))
        print("Example training prompt:\n", prompt(args.model, formatter, rows[0]["text"]))
        print("Example completion:\n", completion(args.model, rows[0]["expected"]))


def train(args):
    lock = verify_data()
    import shutil
    if not args.smoke and shutil.disk_usage(ROOT).free < 2_000_000_000:
        raise SystemExit("Less than 2 GB disk space is free. Free roughly 10 GB before full training and exports; see README.md.")
    path = source_path(args.model)
    out = ROOT / "checkpoints" / (args.model + ("-smoke" if args.smoke else "-lora"))
    if out.exists() and not args.resume:
        raise SystemExit(f"Output already exists: {out}. Use --resume <checkpoint-directory> for a real interrupted run, or choose a fresh package copy.")
    torch = runtime(args)
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    tokenizer, formatter = tokenizer_and_formatter(args.model, path)
    model = load_model(path, args)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                           target_modules="all-linear", task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    rows = read_rows("train_set.jsonl")
    if args.smoke:
        rows = rows[:8]
    examples = [masked_example(tokenizer, prompt(args.model, formatter, row["text"]),
                               completion(args.model, row["expected"])) for row in rows]

    # Same right-padded training collator and completion-only labels as the original.
    def collate(batch):
        width = max(len(x["input_ids"]) for x in batch)
        ids, masks, labels = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
            masks.append([1] * len(item["input_ids"]) + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in
                [("input_ids", ids), ("attention_mask", masks), ("labels", labels)]}

    training_args = TrainingArguments(
        output_dir=str(out), max_steps=2 if args.smoke else 504,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=1 if args.smoke else 5, save_strategy="no" if args.smoke else "steps", save_steps=126,
        save_total_limit=2, report_to=[], bf16=False, fp16=False,
        use_cpu=args.device == "cpu", optim="adamw_torch", lr_scheduler_type="linear",
        warmup_steps=0, weight_decay=0.0, max_grad_norm=1.0,
        seed=42, data_seed=42, dataloader_num_workers=0,
        dataloader_pin_memory=args.device == "cuda", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if training_args.device.type != args.device:
        raise SystemExit(f"Trainer selected {training_args.device}; requested {args.device}. Check the installed environment.")
    manifest = {"model_key": args.model, "source": json.loads((path / "source.json").read_text()),
                "datasets": lock, "schema_version": 3, "code": code_fingerprint(), "environment": metadata(), "device": args.device,
                "threads": args.threads, "dtype": "float32", "smoke_only": args.smoke,
                "settings": training_args.to_dict(),
                "lora": {"r": 16, "alpha": 32, "dropout": .05, "target_modules": "all-linear"},
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        if checkpoint.parent != out.resolve() or not (checkpoint / "trainer_state.json").exists():
            raise ValueError("Resume must point to a checkpoint inside this model's output directory")
        previous = json.loads((out / "run.json").read_text())
        for field in ["source", "datasets", "lora", "dtype", "model_key", "smoke_only", "code", "settings"]:
            assert previous[field] == manifest[field], f"Resume mismatch: {field}"
    manifest["completed"] = False
    write_json(out / "run.json", manifest)
    class FiniteLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = result[0] if return_outputs else result
            if not torch.isfinite(loss.detach()).all():
                raise RuntimeError("Non-finite training loss; stop and inspect precision/memory/runtime settings.")
            return result

    trainer = FiniteLossTrainer(model=model, args=training_args, train_dataset=examples, data_collator=collate)
    synchronize(torch, args.device)
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=args.resume or None)
    synchronize(torch, args.device)
    training_seconds = time.perf_counter() - t0
    model.save_pretrained(out)
    formatter.save_pretrained(out)
    manifest["training_seconds_this_session"] = training_seconds
    manifest["training_seconds"] = training_seconds + (previous.get("training_seconds", 0) if args.resume else 0)
    manifest["resumed"] = bool(args.resume)
    manifest["timing_note"] = "Completed-session wall time; interrupted work before an unrecorded shutdown is excluded."
    manifest["adapter_sha256"] = sha256(out / "adapter_model.safetensors")
    manifest["completed"] = True
    write_json(out / "run.json", manifest)
    print("Smoke test only; not for evaluation." if args.smoke else "Training complete.", out)


def adapter_model(args, path):
    model = load_model(path, args)
    if not args.base:
        from peft import PeftModel
        adapter = ROOT / "checkpoints" / f"{args.model}-lora"
        run = json.loads((adapter / "run.json").read_text())
        assert run.get("completed") and not run["smoke_only"], "Training is not complete"
        assert run["datasets"] == verify_data()
        assert run["adapter_sha256"] == sha256(adapter / "adapter_model.safetensors"), "Adapter changed"
        assert run["source"] == json.loads((path / "source.json").read_text()), "Base checkpoint differs from training"
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    model.eval()
    return model


def evaluate(args):
    lock = verify_data()
    path = source_path(args.model)
    torch = runtime(args)
    tokenizer, formatter = tokenizer_and_formatter(args.model, path)
    model = adapter_model(args, path)
    eos = model.generation_config.eos_token_id
    eos_ids = list(eos) if isinstance(eos, list) else ([eos] if eos is not None else [])
    eos_ids += [tokenizer.convert_tokens_to_ids(t) for t in stop_tokens(args.model)]
    eos_ids = list(dict.fromkeys(eos_ids))
    def generate(text):
        inputs = tokenizer(prompt(args.model, formatter, text), add_special_tokens=False, return_tensors="pt").to(args.device)
        synchronize(torch, args.device)
        started = time.perf_counter()
        with torch.inference_mode():
            seq = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                                 repetition_penalty=1.0, no_repeat_ngram_size=0,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids)
        synchronize(torch, args.device)
        elapsed = time.perf_counter() - started
        generated = seq[0, inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=False), elapsed, len(generated)
    if args.text:
        raw, seconds, count = generate(args.text)
        from model_common import parse
        print(json.dumps({"raw_output": raw, "parsed": parse(args.model, raw), "seconds": seconds, "tokens": count}, indent=2))
        return
    rows = eval_rows(args.eval_set)
    generate(rows[0]["text"])  # Same single warmup as original GGUF benchmark.
    if args.limit:
        rows = rows[:args.limit]
    out = result_path(args)
    if out.exists():
        raise SystemExit(f"Result already exists: {out}. Choose --label for a new run.")
    examples = []
    for i, row in enumerate(rows):
        raw, seconds, count = generate(row["text"])
        examples.append(scored_row(row, raw, seconds, args.model, generated_tokens=count,
                                   hit_token_limit=count >= MAX_NEW_TOKENS))
        print(f"{i + 1}/{len(rows)} correct={examples[-1]['exact_match']} seconds={seconds:.2f}", flush=True)
    summary = {**summarize(examples), "model_key": args.model, "base": args.base,
               "batch_size": 1, "dtype": "float32", "backend": "pytorch-eager", "device": args.device,
               "threads": args.threads, "max_new_tokens": MAX_NEW_TOKENS, "cache_prompt": False,
               "environment": metadata(), "datasets": lock,
               "source": json.loads((path / "source.json").read_text())}
    if not args.base:
        summary["training_run"] = json.loads((ROOT / "checkpoints" / f"{args.model}-lora" / "run.json").read_text())
    summary.update({"schema_version": 3, "code": code_fingerprint(), "eval_set": args.eval_set,
                    "partial_test": bool(args.limit), "decoding": "unconstrained-greedy",
                    "latency_scope": "generation only; tokenization excluded; batch 1; one warmup"})
    write_json(out, {"summary": summary, "examples": examples})
    print(json.dumps(summary, indent=2)); print("Saved", out)


def export(args):
    verify_data()
    path = source_path(args.model)
    tokenizer, formatter = tokenizer_and_formatter(args.model, path)
    model = adapter_model(args, path)
    out = ROOT / "merged" / args.model
    if out.exists():
        raise SystemExit(f"Export already exists: {out}; preserve it or move it before exporting again.")
    training_run = json.loads((ROOT / "checkpoints" / f"{args.model}-lora" / "run.json").read_text())
    save_export(args.model, model, tokenizer, formatter, out, ROOT / "prompts" / f"{args.model}.json", training_run)


def save_export(key, model, tokenizer, formatter, out, prompt_path, training_run):
    """Shared export implementation; tests supply an isolated smoke-only output path."""
    if out.exists():
        raise ValueError(f"Export path already exists: {out}")
    needed = sum(p.numel() * 4 for p in model.parameters()) + 300_000_000
    import shutil
    if shutil.disk_usage(ROOT).free < needed:
        raise RuntimeError(f"Need at least {needed / 1e9:.2f} GB free for this merged export")
    padding = 0
    rows = model.get_input_embeddings().weight.shape[0]
    required = max(tokenizer.get_vocab().values()) + 1
    if required > rows:
        extra = {token: idx for token, idx in tokenizer.get_vocab().items() if idx >= rows}
        if key != "functiongemma" or set(extra) != {"<image_soft_token>", "<end_of_image>"}:
            raise SystemExit(f"Unexpected tokenizer/embedding mismatch: {extra}")
        # The same conversion fix documented in the supplied PROGRESS.md.
        import torch
        model.resize_token_embeddings(required, mean_resizing=False)
        with torch.no_grad():
            model.get_input_embeddings().weight[rows:].zero_()
            model.get_output_embeddings().weight[rows:].zero_()
        padding = required - rows
        print("Added", padding, "zero embedding rows for unused multimodal tokens. Re-evaluate the converted artifact.")
    # Preserve FP32 merging; pinned converter quantizes directly to Q8_0.
    model.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
    formatter.save_pretrained(out)
    write_json(out / "export-metadata.json", {"dtype": "float32", "vocab_padding_rows": padding,
               "weight_files": {f.name: sha256(f) for f in out.glob("*.safetensors")}})
    write_json(out / "training-run.json", training_run)
    rows = []
    for row in eval_rows():
        text = prompt(key, formatter, row["text"])
        rows.append({**row, "prompt": text, "ids": tokenizer(text, add_special_tokens=False)["input_ids"]})
    write_json(prompt_path, {"model_key": key,
               "datasets": verify_data(), "source": json.loads((source_path(key) / "source.json").read_text()),
               "stop": stop_tokens(key), "max_new_tokens": MAX_NEW_TOKENS, "rows": rows})
    print("Saved merged FP32 checkpoint:", out)
    print("Saved model-specific prompt IDs for offline GGUF evaluation.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "download", "train", "eval", "export"])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--eval-set", choices=["all"] + EVAL_FILES, default="all")
    parser.add_argument("--limit", type=int, help="Partial evaluation for debugging; never a final score")
    parser.add_argument("--label", help="Unique result name")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--text")
    args = parser.parse_args(argv)
    if args.command != "check" and not args.model:
        parser.error("--model is required")
    if args.command == "export" and args.base:
        parser.error("Export expects a completed adapter, not --base")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.label and (not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.label)):
        parser.error("Use letters, digits, hyphens or underscores in --label")
    if args.command == "export" and args.device != "cpu":
        parser.error("Export on CPU to keep merges consistent")
    if args.threads < 1:
        parser.error("--threads must be positive")
    {"check": preflight, "download": download, "train": train, "eval": evaluate, "export": export}[args.command](args)


if __name__ == "__main__":
    main()
