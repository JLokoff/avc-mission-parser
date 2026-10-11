# Platform setup

Use Python 3.11 for the recorded training/converter environments. Runtime inference and offline tests use only Python's standard library plus a locally compiled llama-server; they do not import torch, Transformers, FastAPI, or Flask.

## Mac

Install Xcode Command Line Tools, Git, CMake, and native arm64 Python. If using Homebrew, `brew install python@3.11 cmake` supplies the latter two; `xcode-select --install` starts Apple's compiler installation.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0
python -m pip install -r requirements-comparison.txt
python scripts/setup_llama.py --backend metal
python -m venv .venv-convert
.venv-convert/bin/python -m pip install -r requirements-converter.txt
```

Use `--device mps` for training and `--device cpu` for the reference PyTorch evaluation. Optional `caffeinate -i` before a long-running command prevents idle sleep; it does not make a closed laptop continue running.

## Windows PC

Install Python 3.11, Git, CMake, and Visual Studio Build Tools with **Desktop development with C++**. Run CMake from a Developer PowerShell if the compiler is not found.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-comparison.txt
python scripts/setup_llama.py --backend cpu
python -m venv .venv-convert
.venv-convert\Scripts\python.exe -m pip install -r requirements-converter.txt
```

If activation is restricted, call `.venv\Scripts\python.exe` explicitly rather than changing system execution policy. The remaining README commands use `python` and work once the environment is active. Inference on CPU requires `--gpu-layers 0`; training uses `--device cpu`.

For supported NVIDIA hardware, install the driver/CUDA toolkit and the wheel selected by the [official PyTorch installer](https://pytorch.org/get-started/locally/), then build with `--backend cuda` and train with `--device cuda`. An old Quadro M2000 is not assumed compatible with current CUDA wheels. CPU is the fallback. This project deliberately does not silently install a CUDA wheel or replace the GPU driver.

WSL2 uses the Linux commands and paths. Keep the whole environment within WSL; do not mix Windows and Linux Python environments or binaries.

## Jetson Orin Nano / Orin NX

Use a JetPack image with a working CUDA toolkit and C++ compiler. Install Git, CMake, Python, and build tools from the platform package manager. To deploy a model already trained elsewhere:

```bash
python3 scripts/setup_llama.py --backend cuda --cuda-arch 87 --jobs 2
python3 scripts/serve.py --model granite350 --quantization Q6_K --gguf /path/to/granite350-q6_k.gguf
```

Copy the matching `granite350-q6_k.json` conversion manifest beside the GGUF. Clone the repository for its Python code, runtime templates, and grammars. No training environment is required for this path. `--jobs 2` limits build memory pressure. Actual speed depends on power mode, cooling, background perception workloads, and the Jetson SKU.

Training or conversion **on the Jetson** additionally requires a compatible Python/PyTorch stack. Follow [NVIDIA's Jetson PyTorch installation matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) for the installed JetPack. Do not blindly install the Mac-pinned torch wheel on aarch64. The historical converter uses torch 2.11.0 / Transformers 4.57.6; if those cannot coexist with your JetPack environment, perform conversion on a Mac/PC and transfer the artifact. On-device training/conversion has not been validated by this study.

## Tool overrides

For an existing pinned toolchain, set `AVC_LLAMA_DIR` to its checkout directory. `AVC_LLAMA_SERVER`, `AVC_LLAMA_QUANTIZE`, and `AVC_CONVERTER_PYTHON` can override executable locations. The build helper handles both `build/bin/llama-server` and Windows `build/bin/Release/llama-server.exe`. Conversion verifies the pinned llama.cpp tag and commit; runtime manifests bind model files to hashes and model identity.

Use the same software build, device, power settings, decoding mode, thread count, and caching policy when comparing latency. Keep platform-specific results separately labelled.
