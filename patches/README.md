# Why this patch is included

A patch is a small, recorded edit to someone else’s code. It says which lines to add or change, so the same fix can be applied automatically on another computer. The `+` lines in this file are additions; nearby unchanged lines identify where they belong. It is not another AI model or a training dataset.

## Our Granite converter fix

`llama-b11026-granite-all-attention.patch` fixes how the pinned llama.cpp converter identifies Granite 4.0 350M. This model uses a configuration wrapper that can describe hybrid models, but this particular model has only attention layers. Without the fix, the converter can label its GGUF as a hybrid model, making the runtime expect extra model components that are not present.

The patch checks the actual layer types and writes the standard Granite architecture label before the GGUF writer is created. It changes conversion bookkeeping, not the training examples or learned weights. FunctionGemma and LFM do not need this Granite-specific change.

## Do we need to keep it?

**Yes, for rebuilding Granite GGUF files with the converter version this repository currently pins.** Step 00 applies it automatically when you run `python scripts/00_prepare_models.py build --backend metal` (or `cpu`/`cuda`). You do not need to apply it by hand. The build checks whether the fix is already present before applying it.

Simply running an already correctly converted GGUF does not apply this patch. Keep it in the repository so another team member can reproduce the conversion. If we later move to a converter version with an equivalent built-in fix, we can remove it after testing that version. Deleting it today would break the documented build process.
