# What each file is for

You normally use only the ten numbered scripts and the README. The remaining files keep the training data, model behavior, and saved findings reproducible. Do not edit generated files or frozen labels just to make a check pass.

## Scripts you run

### `scripts/00_prepare_models.py`

Prepare the models and tools before you train or run them. Choose download to save a model on this computer, check to inspect the data and test the selected device without training, or build to install the local llama.cpp tools. Run download and check for each model; build is needed once per computer. Previously: `setup_llama.py; download/check commands previously lived in compare.py`.

### `scripts/01_evaluate_baseline.py`

Measure how the original downloaded model performs before training. This sends the evaluation commands to the model, compares its answers with the expected fields, and saves accuracy and response times. Keep this result as the starting point for measuring what training improves. Previously: `00_zeroshot_baseline.py`.

### `scripts/02_train_model.py`

Teach one selected model to turn mission commands into the six Schema V3 fields. It uses the frozen training data and saves the learned changes as a small adapter. Use --smoke for a short installation check; a smoke run is not a trained model for the results table. Previously: `02_finetune_lora.py`.

### `scripts/03_evaluate_trained.py`

Measure the model after training using the same evaluation commands as the baseline. This loads the downloaded model together with its trained adapter and saves accuracy, response times, and individual answers. It does not change the model or train on the evaluation data. Previously: `03_evaluate.py`.

### `scripts/04_prepare_gguf.py`

Create smaller model files that can run in llama.cpp. This step combines the original model and its learned adapter when a merged copy is missing, then creates the selected Q8, Q6, or Q4 files. Each smaller version comes from the full-precision trained model, never from a previously compressed version. It verifies existing files before reusing them. Previously: `05_merge_for_gguf.py plus quantize.py`.

### `scripts/05_benchmark_models.py`

Test a finished GGUF model on the evaluation commands and save its correctness and response times. You can require the expected answer format with --grammar and prepare the shared instructions in advance with --cache-prefix. This step starts and stops its own local model server, so it can be repeated without training or converting again. Previously: `04_benchmark_gguf.py`.

### `scripts/06_serve_api.py`

Keep one trained model running on this computer so another program can send it mission commands. Send a text command to the local /parse address and receive the six mission fields or an error. The model stays loaded between requests; the server interprets commands but never flies the drone or changes the mission itself. Previously: `serve.py`.

### `scripts/07_run_workflow.py`

Run one workflow stage for all three models in succession, or only the models you select. For example, benchmark can test Q8, Q6, and Q4 for each model without typing nine separate commands. It stops if a command fails, so a later step cannot silently hide an earlier problem. Previously: `run_pipeline.py`.

### `scripts/08_view_results.py`

Read the published reference results without unpacking a folder full of logs. By default this lists each saved run and its score and speed. Select a run to see its full details, or add --wrong-only to inspect the commands that model answered incorrectly. Previously: `read_evidence.py`.

### `scripts/09_check_project.py`

Check that the repository still works after a code change. These quick tests inspect the data, scoring, file preparation, command routing, and local HTTP interface without downloading or training a model. They do not prove GPU compatibility or replace a real model evaluation; the model check in step 00 tests your installed device. Previously: `test_all.py`.

## Shared support code

**`core/model_workflow.py`** — Provides the shared download, model checks, training, evaluation, and merging used by the numbered steps. Keeping these operations together means all three models receive the same experiment settings. Normally run a numbered script instead of running this support file directly. Previously: `compare.py, adapted from upstream 02_finetune_lora.py and 03_evaluate.py`.

**`core/model_common.py`** — Keeps the three models on the same comparison rules. It prepares their differently formatted prompts, reads their answers, verifies the frozen datasets, and calculates the scores. The public workflow uses these helpers automatically. Previously: `comparison_common.py`.

**`core/gguf_conversion.py`** — Handles the model-specific details needed to turn a trained model into a usable GGUF file. It checks the training record and converter version, prepares compatible model settings, and creates the Q8 file with a record of its source. Step 04 calls this automatically. Previously: `gguf_workflow.py`.

**`core/quantization.py`** — Coordinates the combined merge and quantization step. It checks that the trained model, saved prompts, and existing files belong together, then creates Q8 or a full-precision master for independent Q6 and Q4 conversion. An incomplete or mismatched file stops the process rather than being silently reused. Previously: `quantize.py and the former merge entry point`.

**`core/artifacts.py`** — Finds the installed model tools and decides where generated models and their records are saved. It supports Mac, Windows, and Linux executable locations and writes records through temporary files to reduce incomplete saves. This combines the former small path and quantization-helper files. Previously: `paths.py and quantization_common.py`.

**`core/runtime_checks.py`** — Checks that the model server is producing valid numerical output before results are trusted. It also selects the smaller Granite prompt-processing batch needed on the tested Mac setup. A broken server calculation is reported as a runtime failure rather than counted as a model misunderstanding. Previously: `gguf_runtime.py`.

**`core/output_grammar.py`** — Builds the allowed answer-format rules for each model. These rules keep generated answers within the field names and values in Schema V3; they do not choose the correct answer for the model. The conversion and inference code use this automatically. Previously: `build_grammars.py`.

**`core/inspect_gguf.py`** — Inspects a newly converted model file and reports the types and sizes of its stored weights. It verifies the requested conversion format and checks that the full-precision source has usable numbers. Step 04 runs it in the converter environment; most users never need to call it themselves. Previously: `inspect_study_gguf.py`.

**`core/schema.py`** — Defines the six mission fields, their allowed values, and the original FunctionGemma instructions and answer reader. Previously `scripts/schema.py`; it is deliberately kept byte-for-byte identical to the upstream schema so existing saved model templates and comparison records stay valid. This guide supplies its plain-language explanation without changing that frozen file.

**`mission_parser.py`** — The Python interface another team program imports. It loads one local model, optionally prepares the fixed instructions, accepts commands, and returns validated mission fields. It does not issue flight commands. Keep one parser open rather than reloading it for each command.

## Data and configuration

**`data/README.md`** — Explains training versus held-out evaluation, lists every dataset and its row count, and explains why JSON Lines files cannot have comment headers.

**`data/train_set.jsonl`** — The 2,932 training examples (1,551 normalized unique commands), with expected mission fields. Repetition intentionally gives some examples more training weight. This is the only file used for learning.

**`data/eval_set.jsonl`, `data/heldout_set.jsonl`, and `data/heldout2_set.jsonl` through `data/heldout8_set.jsonl`** — Nine evaluation files totaling 792 commands. Their separation records where the test groups came from; combining or regenerating them would change the locked comparison. Each line pairs a command with its expected answer.

**`dataset-lock.json`** — Records the expected size and fingerprint of each dataset, so accidental data edits are detected. **`model-lock.json`** — Records exactly which downloaded version of each base model is used. **`source-v3.json`** — Records the upstream project and frozen schema/data identity. These small records prevent unnoticed changes to the experiment.

**`runtime/functiongemma.json`, `runtime/lfm350.json`, and `runtime/granite350.json`** — Store each model's fixed instructions, command wrapper, and stopping rules. They let the lightweight inference interface build the same prompt without installing the training libraries.

**`gguf/functiongemma-v3.gbnf`, `gguf/lfm350-v3.gbnf`, and `gguf/granite350-v3.gbnf`** — The allowed output-format rules for each model. They differ because the models write function calls differently. They constrain structure, not whether the chosen action or color is correct.

**`requirements-comparison.txt`** — Python libraries used for training and PyTorch evaluation. **`requirements-converter.txt`** — Libraries for a separate conversion environment. They stay separate because the recorded training and conversion workflows need different library versions.

**`patches/README.md`** — Explains what a patch is, why this Granite fix is needed, and when it is applied.

**`patches/llama-b11026-granite-all-attention.patch`** — The small compatibility fix applied to the pinned llama.cpp converter for this Granite model. Step 00 applies it automatically; do not delete it while using that pinned converter.

## Examples, checks, and saved findings

**`examples/python_client.py`** — A short example for importing the parser in another Python program. **`examples/http_client.py`** — A short example for sending a command to the server started by step 06. Choose the integration style that suits your team.

**`tests/test_comparison.py`** — Data, prompt, parsing, and scoring checks for the three models. **`tests/test_schema_v3.py`** — Checks the schema and its original translation rules. **`tests/test_quantize.py`** — Checks full-precision sources and conversion commands. **`tests/test_benchmark_numeric.py`** and **`tests/test_gguf_runtime.py`** — Distinguish a broken numerical calculation from a normal wrong answer. **`tests/test_benchmark_ports.py`** — Checks server-port reuse and conflicts. **`tests/test_interface.py`** — Checks Python and HTTP input/output rules. **`tests/test_evidence.py`** — Recalculates published scores. **`tests/test_workflow.py`** — Checks numbered entry points, every model/format route, and stop-on-failure behavior. These have different responsibilities and remain separate for easier troubleshooting.

**`.github/workflows/checks.yml`** — Runs the offline checks on Mac, Windows, and Linux whenever code changes. It does not run GPU training or a full model evaluation. **`.gitignore`** — Keeps downloaded weights, temporary environments, and newly generated outputs out of GitHub.

**`evidence/summary.json`** — Compact metrics for 28 retained full evaluations. **`evidence/runs.json.gz`** — Compressed individual predictions behind those metrics, including expected answers and timing. **`evidence/README.md`** — Explains how to read them with step 08. Detailed results stay compressed to keep browsing uncluttered.

## Reading material

**`README.md`** — Main walkthrough and result highlights. **`docs/SETUP.md`** — Platform-specific installation details. **`docs/SCHEMA.md`** — What the six fields mean. **`docs/INTEGRATION.md`** — How another program should call the parser and handle responses. **`docs/RESULTS.md`** — Accuracy, timing, and comparison limits. **`docs/FILES.md`** — This guide. **`NOTICE.md`** — Credits and upstream/model terms.

## Files created when you run the workflow

`models/` holds downloaded weights; `checkpoints/` holds training progress and adapters; `merged/` holds full-precision trained exports; `prompts/` holds saved evaluation prompts; `gguf/` receives deployable model files and their JSON source records; `results/` receives new evaluations; `logs/` holds server messages; `tools/` holds the built inference tools. These outputs are intentionally ignored by Git. A model file's adjacent JSON record identifies its source and conversion; keep the two together when deploying.
