# Reference results

Only two data files are needed to keep the findings auditable without dozens of visible logs:

- `summary.json`: metrics and provenance for 41 successful full evaluations.
- `runs.json.gz`: the same summaries plus every command, expected label, model output, parsed fields, and latency.

Included: three models' CPU baselines and trained CPU evaluations; Q8 plain/grammar; historical Q6/Q5/Q4/Q3 plain/grammar; Granite cached Q8 ordered/shuffled and cached Q6/Q4/Q3. Historical Q5/Q3 are retained as evidence, while supported conversion commands in this repository intentionally focus on Q8/Q6/Q4.

Excluded: failed/nonfinite runs, partial smoke tests, duplicate server logs, weights, private paths. Source absolute paths/usernames were replaced with `$PROJECT`/`$USER_HOME`; repetitive `server_props` were omitted. The original source file SHA-256 is retained separately from the sanitized payload.

```bash
python scripts/read_evidence.py
python scripts/read_evidence.py --run granite350-v3-q8-metal-grammar.json --wrong-only
```

Each full run contains 792 entries. Summaries can be recomputed from those entries using `comparison_common.summarize`. Subset labels are diagnostic tags; they are not necessarily independent partitions. New local runs go in ignored `results/`, not this published reference archive.
