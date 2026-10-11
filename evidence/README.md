# Reference results

Only two data files are needed to keep the findings auditable without dozens of visible logs:

- `summary.json`: metrics and provenance for 28 successful full evaluations.
- `runs.json.gz`: the same summaries plus every command, expected label, model output, parsed fields, and latency.

Included: three models' CPU baselines and trained CPU evaluations; Q8/Q6/Q4 plain and grammar evaluations; Granite cached Q8 ordered/shuffled and cached Q6/Q4. Only the supported quantization formats are included.

Excluded: failed/nonfinite runs, partial smoke tests, duplicate server logs, weights, private paths. Source absolute paths/usernames were replaced with `$PROJECT`/`$USER_HOME`; repetitive `server_props` were omitted. The original source file SHA-256 is retained separately from the sanitized payload.

```bash
python scripts/08_view_results.py
python scripts/08_view_results.py --run granite350-v3-q8-metal-grammar.json --wrong-only
```

Each full run contains 792 entries. Summaries can be recomputed from those entries using `model_common.summarize`. Subset labels are diagnostic tags; they are not necessarily independent partitions. New local runs go in ignored `results/`, not this published reference archive.
