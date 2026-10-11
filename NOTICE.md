# Provenance and terms

This repository packages the user's AVC comparison work. Original Schema V3 definitions, frozen datasets, and the FunctionGemma training/evaluation starting point came from:

- https://github.com/abhihari010/FunctionGemma/tree/schema-v3-regions
- Source commit: `b73783a46e9b1b0b418a8157ec634b61c2703aa3`
- Dataset and schema hashes: `source-v3.json` and `dataset-lock.json`

The supplied source snapshot did not contain a LICENSE file. This publication preserves attribution and does not assert a new blanket license over that material. Public visibility is not itself an open-source license. Obtain the relevant authors' permission before redistribution or reuse that requires it; a repository-wide license should only be added by the rights holders.

The current package preserves `core/schema.py` and the datasets unchanged. Comparison, conversion, benchmarking, prefix-reuse, runtime integration, and documentation adaptations are identified in the README and Git history. Historical result provenance is retained in the compressed evidence archive; local usernames/absolute paths are sanitized for publication.

Model weights and trained adapters are excluded. Their model-specific terms continue to apply:

- FunctionGemma: https://huggingface.co/google/functiongemma-270m-it (Gemma terms; gated access).
- LFM2.5 350M: https://huggingface.co/LiquidAI/LFM2.5-350M (LFM 1.0 terms).
- Granite 4.0 350M: https://huggingface.co/ibm-granite/granite-4.0-350m (Apache 2.0).

llama.cpp is downloaded separately from https://github.com/ggml-org/llama.cpp and retains its own license. The included patch records the local Granite conversion fix. The public repository contains no competition rules PDFs, private correspondence, access tokens, or local environment folders.
