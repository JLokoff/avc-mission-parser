# AVC Mission Parser

Turn a human mission command into six structured fields for an autonomy stack. Train and compare **FunctionGemma 270M**, **LFM2.5 350M**, and **Granite 4.0 350M** on the same frozen Schema V3 data, then deploy locally with llama.cpp.

**Input:** “Stay to the south until you reach the eastern half, then collect the blue object in the northeast.”

```json
{
  "target_color": "blue",
  "avoid_region": "none",
  "stay_region": "S",
  "until_region": "E",
  "target_location": "NE",
  "action": "collect_target"
}
```

The parser extracts intent. GNC, Perception, the mission state machine, and flight safety logic decide whether and how to act. This repository does not send actuator commands.

## Start here

| What you want | Follow this path |
|---|---|
| Understand the interface | [Schema and mission semantics](docs/SCHEMA.md) |
| Train and reproduce the study | [Setup](docs/SETUP.md), then steps 1–5 below |
| Run an existing trained GGUF | Build llama.cpp, copy the GGUF and its JSON manifest, then [Python/HTTP inference](#python-and-local-http-inference) |
| Compare speed and accuracy | [Results and methodology](docs/RESULTS.md) |
| Integrate with GNC/Perception | [Integration contract](docs/INTEGRATION.md) |

Code, frozen datasets, prompt templates, grammars, and measured results are included. **Base weights, trained adapters, and GGUF binaries are not included.** Download and train using the commands below, or transfer an existing compatible artifact from your training computer. No API subscription is needed for local inference.

## Platforms

- **Apple Silicon Mac:** CPU/MPS training; Metal GGUF inference. This is the measured reference platform (M4 MacBook Pro, 16 GB).
- **Windows PC:** CPU or supported NVIDIA CUDA training/inference. Native Python scripts and Windows executable paths are supported; WSL2 is also an option.
- **Linux / Jetson Orin Nano or NX:** CPU/CUDA runtime build options. For deployment, training elsewhere and transferring the GGUF avoids installing PyTorch on the Jetson.

Windows and Jetson performance have **not** been measured here. GPU compatibility depends on the installed driver, CUDA/JetPack, compiler, and PyTorch wheel. An original Jetson Nano is different from an **Orin Nano**. See [platform setup](docs/SETUP.md); no universal hardware speed multiplier is claimed.

## 1. Install and download

Install Git, Python **3.11**, CMake, and your platform's C++ build tools. Clone this repo:

```bash
git clone https://github.com/JLokoff/avc-mission-parser.git
cd avc-mission-parser
python3.11 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install torch==2.6.0
python -m pip install -r requirements-comparison.txt
python -m pip check
```

On Windows use `py -3.11 -m venv .venv` and `.venv\Scripts\Activate.ps1`. The `torch==2.6.0` command above reproduces the Mac environment; use the appropriate official CUDA or Jetson wheel for those devices. Detailed alternatives are in [SETUP.md](docs/SETUP.md).

FunctionGemma requires accepting Google's terms on its [Hugging Face page](https://huggingface.co/google/functiongemma-270m-it), then authenticating locally:

```bash
hf auth login
python scripts/07_run_workflow.py download
python scripts/07_run_workflow.py check --device cpu
```

Never put access tokens in code. Downloads use revisions in `model-lock.json`. All pipeline commands accept `--models granite350`, `--models lfm350`, or `--models functiongemma` to select one model. Otherwise they run all three sequentially and stop at the first failure.

## 2. Establish a baseline and train

Run baseline evaluation on CPU to match the archived PyTorch reference:

```bash
python scripts/07_run_workflow.py baseline --device cpu
python scripts/07_run_workflow.py train --device mps --smoke
python scripts/07_run_workflow.py train --device mps
python scripts/07_run_workflow.py evaluate --device cpu
```

Replace `mps` with `cuda` for a supported NVIDIA GPU or `cpu` for CPU training. `--smoke` runs only two optimizer steps on eight rows and saves separate smoke checkpoints. It must not be used for final evaluation.

Full training uses LoRA rank 16, alpha 32, dropout 0.05, all linear layers, FP32, batch size 1, gradient accumulation 8, learning rate 0.0002, 504 optimizer steps, and seed 42. Completion tokens alone contribute to training loss. Evaluation uses deterministic greedy generation with a 256-token output cap.

The dataset contains **2,932 training rows representing 1,551 normalized unique commands**; repeated rows intentionally weight selected examples. Evaluation contains **792 rows across nine frozen files**. Do not regenerate, deduplicate, or tune on these evaluation labels while claiming a direct comparison. Dataset hashes and overlap checks run automatically.

Resume an interrupted full training run with an existing checkpoint:

```bash
python scripts/02_train_model.py --model granite350 --device mps --threads 8 --resume checkpoints/granite350-lora/checkpoint-252
```

`252` is an example. Resume requires matching data, settings, source weights, and training code. Completed output directories are protected from accidental overwrite. Keep several GB free for each merged export and allow additional space for FP32 GGUF masters; 15 GB of working space is a reasonable starting allowance for all three beyond downloaded dependencies.

## 3. Build the inference/conversion tools

```bash
python scripts/00_prepare_models.py build --backend metal
python -m venv .venv-convert
.venv-convert/bin/python -m pip install -r requirements-converter.txt
```

Choose `--backend cpu` or `--backend cuda` on other hardware. On Windows the converter executable is `.venv-convert\Scripts\python.exe`. For Orin, use `--backend cuda --cuda-arch 87`; see [SETUP.md](docs/SETUP.md).

The setup script pins llama.cpp to `b11026` / `b49650adb31f2e49a0d76113aeb1792134fd8413` and applies the included Granite converter patch. Use a separate converter environment: the recorded workflow uses Transformers 5 for training and Transformers 4.57.6 for conversion. Keep the training environment active when running repository commands.

## 4. Merge and quantize

```bash
python scripts/07_run_workflow.py quantize --quants Q8_0 Q6_K Q4_K_M
```

Merging combines the original model weights with the learned LoRA update into a standalone FP32 model. It is required before GGUF conversion; it does not combine different models or datasets.

- **Q8_0:** convert the merged FP32 checkpoint directly into a Q8 GGUF.
- **Q6_K / Q4_K_M:** create an FP32 GGUF master from that same merged checkpoint and independently quantize it to each format. **Q8 is never used as the input for lower-bit conversion.**
- K-quant formats use blocks and mixed tensor precisions. “Q4” does not mean every tensor is exactly four bits. `core/inspect_gguf.py` checks format metadata and reports actual tensor types.

Existing artifacts are hash-checked and preserved. Interrupted `.partial` files require inspection before retrying. No training occurs during quantization. Single-model example:

```bash
python scripts/04_prepare_gguf.py --model granite350 --quants Q8_0 Q6_K Q4_K_M
```

## 5. Evaluate GGUF and prefix reuse

```bash
python scripts/07_run_workflow.py benchmark --quants Q8_0 Q6_K Q4_K_M
python scripts/07_run_workflow.py benchmark --models granite350 --quants Q8_0 Q6_K Q4_K_M --cache-prefix
```

Use `--gpu-layers 0` for CPU-only execution. Add `--limit 5` for a partial installation check; omit it for a 792-command score. Results go to ignored `results/` folders with unique labels. The published reference results live separately in `evidence/`.

The pipeline benchmarks use grammar-constrained generation. To measure plain generation, run the underlying script without `--grammar`:

```bash
python scripts/05_benchmark_models.py --model granite350 --quantization Q8_0 --label granite-q8-plain
```

**Grammar** restricts output structure and enum values; it does not guarantee the intended meaning. Prefix reuse saves the model's computed fixed instructions. Each request still contains the full prompt and only the new command, with no preceding commands appended as chat history.

Granite's 552-token prefix was validated on all 792 examples, including a shuffled Q8 pass: no output changes were observed. Preparation costs about 0.75 s once, excluding model loading. Keep the server running to retain the benefit. Caching support and benefit vary by architecture/runtime; LFM and FunctionGemma default to uncached inference. Explicit caching on those models is experimental and must be evaluated separately.

## Python and local HTTP inference

After training/conversion, start a persistent service:

```bash
python scripts/06_serve_api.py --model granite350 --quantization Q8_0
```

In a second terminal:

```bash
python examples/http_client.py
```

The API listens only on `127.0.0.1:8080`. `GET /health` reports readiness. `POST /parse` accepts `{"text":"Collect the blue box in the northeast corner."}` and returns `{"ok":true,"schema_version":3,"model":"granite350","intent":{...},...}`. Invalid or incomplete model output returns `ok: false` and no usable intent. See [HTTP status codes and state semantics](docs/INTEGRATION.md).

For direct Python integration, run from the repository root or add the repository root to your application's module path:

```python
from mission_parser import MissionParser

with MissionParser(model="granite350", quantization="Q8_0") as parser:
    result = parser.parse("Collect the blue box in the northeast corner.")
    if result["ok"]:
        intent = result["intent"]
        # Validate against mission state, then hand off to your GNC interface.
```

Create the parser **once**, outside your incoming-command loop. Granite preloads automatically; use `cache_prefix=False` in Python or `--no-cache-prefix` on the server to disable it. A transferred model can be selected with `--gguf /path/to/model.gguf`; the matching `.json` conversion manifest must accompany it. Only the GGUF, manifest, repository runtime code/templates, and locally built llama-server are needed for inference. No model hub access is needed once these are present.

## Measured results

All accuracies below use the same 792 examples. PyTorch is FP32 CPU; Q8 is Metal + grammar on the M4 Mac. These are different execution configurations, so the speed difference is not quantization alone.

| Model | Base accuracy | Trained PyTorch accuracy | Q8 + grammar accuracy | Q8 size | Q8 uncached median |
|---|---:|---:|---:|---:|---:|
| FunctionGemma 270M | 0/792 (0.00%) | 733/792 (92.55%) | 734/792 (92.68%) | 299.8 MB | 0.314 s |
| LFM2.5 350M | 2/792 (0.25%) | 761/792 (96.09%) | 760/792 (95.96%) | 379.2 MB | 0.332 s |
| Granite 4.0 350M | 34/792 (4.29%) | 757/792 (95.58%) | 758/792 (95.71%) | 378.1 MB | 1.094 s |

Granite after prefix preparation:

| Quantization | Size | Accuracy | Cached median |
|---|---:|---:|---:|
| Q8_0 | 378.1 MB | 758/792 (95.71%) | 0.350 s |
| Q6_K | 292.8 MB | 757/792 (95.58%) | 0.301 s |
| Q4_K_M | 237.0 MB | 736/792 (92.93%) | 0.270 s |

Q6 is a useful Granite size/speed tradeoff in this dataset: one fewer correct example than Q8. Q4 loses 22 correct examples versus Q8. This is one trained checkpoint per model, not a multi-seed statistical ranking. Baseline scores measure exact native function-call compliance as well as semantic extraction; zero exact matches is not zero language understanding.

The [results guide](docs/RESULTS.md) contains all three Q8/Q6/Q4 comparisons, field-level errors, timing definitions, and caveats. Detailed successful baseline, trained, Q8/Q6/Q4, and plain-generation runsare compressed in one archive, keeping the repository browsable. Smoke tests, failed numeric runs, duplicate logs, and model binaries are excluded.

## Numbered workflow

The scripts below are the files you normally run. Steps 00–05 prepare and measure a model; 06 deploys it. Steps 07–09 are optional shortcuts and checks, not additional training stages. Each script starts with a plain-language explanation and its previous filename.

| Step | What it does |
|---|---|
| `00_prepare_models.py` | Prepare the models and tools before you train or run them. |
| `01_evaluate_baseline.py` | Measure how the original downloaded model performs before training. |
| `02_train_model.py` | Teach one selected model to turn mission commands into the six Schema V3 fields. |
| `03_evaluate_trained.py` | Measure the model after training using the same evaluation commands as the baseline. |
| `04_prepare_gguf.py` | Create smaller model files that can run in llama.cpp. |
| `05_benchmark_models.py` | Test a finished GGUF model on the evaluation commands and save its correctness and response times. |
| `06_serve_api.py` | Keep one trained model running on this computer so another program can send it mission commands. |
| `07_run_workflow.py` | Run one workflow stage for all three models in succession, or only the models you select. |
| `08_view_results.py` | Read the published reference results without unpacking a folder full of logs. |
| `09_check_project.py` | Check that the repository still works after a code change. |

Shared code lives in `core/` so the same logic is not copied into every script. The [complete file guide](docs/FILES.md) explains every remaining source, data, configuration, and evidence file. Conversion now includes merging; there is no separate merge command to remember. Benchmarking stays separate so you can measure an existing model without recreating it.

**Existing users:** the schema, data, model settings, and existing GGUF files have not changed. Use the new script names below. Old unfinished training runs should be resumed with the code that created them; the training-code check intentionally rejects resuming under this reorganized version. Completed adapters and compatible GGUF files can still be reused.

## Checks and troubleshooting

```bash
python scripts/09_check_project.py
```

These offline tests cover schema/data integrity, output parsing, numerical checks, port handling, and the HTTP contract. GitHub Actions runs them on macOS, Ubuntu, and Windows; that is not a GPU hardware certification.

- **Missing pip:** `python -m ensurepip --upgrade` inside the environment.
- **FunctionGemma 403:** accept model terms and authenticate the same Hugging Face account.
- **LFM convolution fallback on Mac:** expected; its CUDA-only optimized kernel is not needed for this workflow.
- **Granite Metal nonfinite output:** retain the automatic microbatch of 8 on Mac GPU; larger values produced broken logits in the pinned build. The benchmark stops on numeric failures.
- **MPS evaluation abort:** use CPU for the PyTorch reference and Metal llama.cpp for deployment evaluation.
- **Address in use:** private inference/benchmark servers select an available port. Change `06_serve_api.py --port` if the public-facing localhost API port is occupied.
- **Model import/version errors:** use the separate pinned training and converter environments. Do not install converter requirements into the training environment.

## Attribution and model terms

Schema V3, frozen data, and the original FunctionGemma workflow come from [abhihari010/FunctionGemma, schema-v3-regions](https://github.com/abhihari010/FunctionGemma/tree/schema-v3-regions), commit recorded in `source-v3.json`. This repository adds the three-model comparison, quantization workflow, prefix study, portability work, and integration interfaces.

Model terms differ: [Granite](https://huggingface.co/ibm-granite/granite-4.0-350m) lists Apache 2.0, [LFM](https://huggingface.co/LiquidAI/LFM2.5-350M) lists LFM 1.0, and [FunctionGemma](https://huggingface.co/google/functiongemma-270m-it) uses Gemma terms. Weights are obtained from their authors. See [NOTICE.md](NOTICE.md); no blanket license is asserted over upstream code, datasets, or model weights.
