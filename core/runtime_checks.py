"""
Checks that the model server is producing valid numerical output before results are
trusted. It also selects the smaller Granite prompt-processing batch needed on the
tested Mac setup. A broken server calculation is reported as a runtime failure rather
than counted as a model misunderstanding.
"""
# Previously: gguf_runtime.py
import math
import sys

NUMERIC_CHECK = 'preflight-and-invalid-output-top-logprob-v1'


def default_ubatch(model, gpu_layers):
    # Pinned b49650a on this Mac: Granite generates nonfinite logits with
    # Metal's matrix-matrix prompt path (>=16 tokens). <=8 uses the working
    # matrix-vector path. This changes execution, not weights or decoding.
    return 8 if sys.platform == 'darwin' and model == 'granite350' and gpu_layers else 512


def check_generation(result):
    """Reject backend numerical failures, never ordinary incorrect predictions.

    Inspect the highest PRE-sampling probability, not the selected token:
    grammar may force a low-probability token whose probability underflows.
    A valid distribution must have a finite highest log-probability.
    """
    if result.get('error'):
        raise RuntimeError(f"llama-server generation failed: {result['error']}")
    probabilities = result.get('completion_probabilities')
    if not probabilities:
        raise RuntimeError('No token probability diagnostics returned; cannot verify numerical health.')
    for index, token in enumerate(probabilities):
        top = token.get('top_logprobs') or []
        value = top[0].get('logprob') if top else None
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise RuntimeError(
                f'Nonfinite or missing model probabilities at output token {index + 1}. '
                'This is an inference-backend failure, not a 0% accuracy score. '
                'No completed benchmark result will be saved. On Granite/Metal use --ubatch-size 8.'
            )
