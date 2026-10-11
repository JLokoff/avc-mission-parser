# Python and HTTP integration

The stable boundary is **human text → parsed intent**. Both interfaces use the same `MissionParser`; neither applies an intent to the drone.

## Deployment bundle

Keep the repository's `mission_parser.py`, `scripts/`, `core/`, and `runtime/` together, plus your GGUF and matching conversion JSON manifest. Build llama-server on the target device. The default artifact locations are `gguf/<model>-q8_0.gguf` and `gguf/quantization/<model>-q6_k.gguf` or `-q4_k_m.gguf`.

No trained weights are bundled on GitHub. Retrain from the frozen data or obtain your team's compatible artifacts. The service checks model identity, quantization, schema hash, and GGUF hash before loading. Templates were extracted from the exact original prompt bundles and checked against every saved evaluation prompt.

## Local HTTP contract

Start `python scripts/06_serve_api.py --model granite350 --quantization Q8_0`. It binds only to localhost. This basic HTTP server handles one request at a time, with a single model slot. It has no public-network authentication or TLS layer; put any remote transport in a separate, authenticated application.

```http
POST /parse
Content-Type: application/json

{"text":"Collect the blue box in the northeast corner."}
```

```json
{
  "ok": true,
  "schema_version": 3,
  "model": "granite350",
  "intent": {
    "target_color": "blue",
    "avoid_region": "none",
    "stay_region": "none",
    "until_region": "none",
    "target_location": "NE",
    "action": "collect_target"
  },
  "latency_s": 0.35,
  "timings": {},
  "cache_prefix_requested": true
}
```

The timing number above is illustrative; actual responses contain measured latency and server timing fields.

| Status | Meaning |
|---|---|
| 200 | Successful extraction, or a healthy `GET /health` |
| 400 | Invalid request shape/text; only `text` is accepted |
| 404 | Unknown endpoint |
| 413 | Missing/oversized body; maximum 16 KB |
| 415 | Expected application/json |
| 422 | Model output invalid or incomplete; do not apply it |
| 503 | Inference unavailable; do not apply an intent |

Text is limited to 2,048 characters, with a separate context-token check. Reserved `<`/`>` template delimiters and NUL bytes are rejected. These checks preserve message boundaries; they are not a general natural-language prompt-injection detector.

The Python API raises `ValueError` for invalid caller input and runtime exceptions for unavailable inference. Invalid model output returns `ok: false, intent: null`. The HTTP wrapper translates these into the status codes above.

## Mission-state handoff

1. Accept a command through your chosen transport. The transport, not the LLM, handles authentication, acknowledgements, and duplicate command IDs.
2. Call the persistent parser. Do not create a new server for each message.
3. Check `ok`, `schema_version`, and the returned fields.
4. Validate against mission state, known targets, map regions, and permissible transitions. Resolve what unspecified fields retain and what happens if a constraint conflicts with the requested route.
5. Apply the approved change through GNC/Perception. Keep emergency stop and collision avoidance independent of LLM availability.

The repository deliberately does not implement a guessed state merge policy: V3 cannot distinguish “clear this constraint” from “no constraint mentioned.” The team must agree on this contract before connecting the parser to flight control.

## Prefix lifecycle

`MissionParser.start()` loads the model, checks numeric health, and prepares Granite's fixed instructions. Each parse call sends the complete template plus the new command with cache reuse enabled. An internal lock serializes access, so simultaneous Python calls cannot mix server-slot state. No previous user command is appended to the next prompt.

`parser.prepare_prefix()` explicitly prepares the fixed prefix again. Restart the service when changing templates or models. Cache loss is a performance issue; requests still include all instructions. The pinned server's warmup emits one disposable token, which is discarded rather than treated as an intent.

Cache reuse is enabled by default only for Granite. You can request it on the other models using `--cache-prefix`, but their saved state behavior differs and the Granite speedup cannot be assumed. Inspect returned `timings.cache_n` and run a full paired accuracy test before adopting a new policy.

Benchmark timings exclude request tokenization and startup preparation. The Python API's `latency_s` includes tokenization and generation after startup, so it is not exactly the same timing scope as the archived benchmark.
