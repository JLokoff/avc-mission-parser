# Results and methodology

Reference hardware: M4 MacBook Pro, 16 GB unified memory. All scores below use 792 frozen Schema V3 examples. MB means decimal megabytes. No Jetson latency has been measured.

## Baseline to trained PyTorch (FP32 CPU)

| Model | Base correct | Trained correct | Base median | Trained median | Training time on MPS |
|---|---:|---:|---:|---:|---:|
| FunctionGemma 270M | 0/792 (0.00%) | 733/792 (92.55%) | 1.362 s | 1.420 s | 57.1 min |
| LFM2.5 350M | 2/792 (0.25%) | 761/792 (96.09%) | 3.559 s | 3.398 s | 49.3 min |
| Granite 4.0 350M | 34/792 (4.29%) | 757/792 (95.58%) | 2.625 s | 2.559 s | 74.7 min |

## GGUF + grammar, uncached (Metal)

| Model | Format | Size | Correct | Median | p95 |
|---|---|---:|---:|---:|---:|
| functiongemma | Q8_0 | 299.8 MB | 734/792 (92.68%) | 0.314 s | 0.335 s |
| functiongemma | Q6_K | 291.2 MB | 735/792 (92.80%) | 0.356 s | 0.751 s |
| functiongemma | Q4_K_M | 261.3 MB | 737/792 (93.06%) | 0.323 s | 0.348 s |
| lfm350 | Q8_0 | 379.2 MB | 760/792 (95.96%) | 0.332 s | 0.368 s |
| lfm350 | Q6_K | 293.4 MB | 761/792 (96.09%) | 0.318 s | 0.339 s |
| lfm350 | Q4_K_M | 229.3 MB | 757/792 (95.58%) | 0.284 s | 0.310 s |
| granite350 | Q8_0 | 378.1 MB | 758/792 (95.71%) | 1.094 s | 1.145 s |
| granite350 | Q6_K | 292.8 MB | 757/792 (95.58%) | 1.501 s | 1.611 s |
| granite350 | Q4_K_M | 237.0 MB | 736/792 (92.93%) | 1.160 s | 1.201 s |

## Granite + grammar, prepared prefix (Metal)

| Format | Correct | Median | p95 |
|---|---:|---:|---:|
| Q8_0 | 758/792 (95.71%) | 0.350 s | 0.425 s |
| Q8_0 shuffled | 758/792 (95.71%) | 0.345 s | 0.388 s |
| Q6_K | 757/792 (95.58%) | 0.301 s | 0.340 s |
| Q4_K_M | 736/792 (92.93%) | 0.270 s | 0.303 s |
| Q3_K_M | 628/792 (79.29%) | 0.315 s | 0.341 s |

Cached Q8/Q6/Q4/Q3 outputs matched their corresponding uncached outputs exactly on this set. Q8 was additionally shuffled with no changes. The fresh 40-command Q8 timing sample measured 1.088 s uncached versus 0.350 s cached; the median per-command speedup was 3.114×. Prefix preparation itself took 0.739–0.758 s, separately from model loading. Q6/Q4/Q3 cached tests did not repeat the shuffled pass.

## Per-field accuracy: Q8 + grammar

| Model | Color | Avoid | Stay | Until | Target location | Action |
|---|---:|---:|---:|---:|---:|---:|
| functiongemma | 99.87% | 97.85% | 96.09% | 99.75% | 98.99% | 97.35% |
| lfm350 | 99.87% | 98.48% | 97.98% | 99.87% | 99.75% | 98.36% |
| granite350 | 99.87% | 98.74% | 97.73% | 99.62% | 99.62% | 98.61% |

## What the statistics mean

- **Exact-match accuracy:** all six extracted values must match the reference. Correct syntax alone is insufficient.
- **Per-field accuracy:** fraction correct for one field, even if another field in that command was wrong.
- **Median:** middle observed response time; **mean:** arithmetic average; **p95:** 95% of measured requests finish at or below this time.
- **Size:** file bytes, not peak process memory. Runtime also needs cache, buffers, and application memory.
- **Invalid output:** native wrapper, missing/duplicate fields, or enum violations. **Cross-field violation:** until_region used without stay_region.
- **Prompt time:** evaluating input tokens. **Generation time:** sequentially producing the answer. Smaller weights do not guarantee a proportional decrease in either.

## Interpretation and limits

The CPU-to-Metal comparison changes execution framework, device, numeric precision, and possibly grammar. Do not attribute the full speedup to quantization. Granite's slow uncached prompt processing used microbatch 8 to avoid a demonstrated nonfinite Metal path in the pinned build. Prefix reuse removes much of that repeated computation. Smaller quantizations mainly expose their decoding benefit after this overhead is reduced.

FunctionGemma's dominant embedding/output tensors retain higher precision under the default K-quant recipes; file-size and latency reductions are consequently smaller than the format names suggest. LFM and FunctionGemma showed little cache reuse in earlier small probes; no full cached accuracy claim is made for them.

Grammar ensures allowed output structure, not semantic truth. Small one- or two-example differences between quantizations are not evidence that quantization generally improves intelligence. The study uses one training seed, one checkpoint per model, a small domain-specific evaluation, no repeated hardware timing trials, and a Mac that may vary thermally. Treat novel paraphrases and actual drone integration as separate tests.

`stay_region` is the weakest Q8 field across all three models; route containment and negation deserve particular attention. Inspect individual failed commands before adding examples: some apparent errors expose schema/label limits. Do not train on this frozen evaluation set. Create a separate development set for changes and a new untouched test set for final validation.

The compressed archive preserves the historical Q5/Q3 experiments and plain-generation comparisons without adding them to the primary execution workflow. No failed numerical run is counted as a model score. See [evidence layout](../evidence/README.md).
