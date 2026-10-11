# Frozen Schema V3

`core/schema.py` is preserved byte-for-byte and checked against `source-v3.json`. Do not confuse this implemented schema with earlier proposed property trees.

| Field | Allowed values | Meaning |
|---|---|---|
| `target_color` | blue, red, yellow, black, unspecified | Target explicitly mentioned |
| `target_location` | NW, NE, SW, SE, unspecified | Target quadrant; not a route or prohibited region |
| `avoid_region` | NW, NE, SW, SE, N, S, E, W, none | One explicitly named keep-out region |
| `stay_region` | NW, NE, SW, SE, N, S, E, W, pathway, none | One containment region or safe pathway |
| `until_region` | NW, NE, SW, SE, N, S, E, W, none | Where the `stay_region` restriction ends |
| `action` | collect_target, read_chip, abort, return_to_start, report_status, pause, resume, retry_read, retry_send | Requested mission operation |

All six fields are required. `until_region` must be `none` when `stay_region` is `none`. The runtime rejects violations rather than silently repairing them. Grammar constrains individual enum fields; runtime validation enforces this cross-field rule.

## Important distinctions

- **collect_target:** obtain the physical target. Returning it, reading its tag, and reporting are mission-state-machine policies, not extra fields in this schema.
- **read_chip:** request the message/tag read without expressing physical collection.
- **abort:** cancel the current mission. **return_to_start:** request travel back. The flight controller must define safe payload handling and return/landing behavior; they are not interchangeable parser labels.
- **pause / resume:** suspend progress versus continue. The parser does not implement hover, braking, or emergency landing.
- **retry_read / retry_send:** repeat a read or transmission; the mission controller checks that the relevant state exists.
- **report_status:** request status; actual telemetry comes from the autonomy stack.

V3 intentionally has no `update_only`: target-position corrections and constraint-only examples use `collect_target` in the frozen labels. `unspecified` and `none` mean the command did not state that field. They do **not** inherently authorize clearing an existing target or region restriction. Merging an extraction into active mission state is an integration policy, not a capability encoded by these six fields.

## Examples

| Command | Color | Avoid | Stay | Until | Target location | Action |
|---|---|---|---|---|---|---|
| Collect the blue box in the northeast corner. | blue | none | none | none | NE | collect_target |
| Avoid the southwest quarter of the field. | unspecified | SW | none | none | unspecified | collect_target |
| The blue object has moved to the northeast quarter. | blue | none | none | none | NE | collect_target |
| Stay south until you reach the eastern half, then collect the blue object in the northeast. | blue | none | S | E | NE | collect_target |
| Pause the mission. | unspecified | none | none | none | unspecified | pause |
| Try reading the chip again. | unspecified | none | none | none | unspecified | retry_read |

## Known scope limits

This version cannot faithfully represent arbitrary coordinates, multiple regions of the same type, unnamed prohibited zones, green/orange targets, multi-action sequences, or an explicit clear-constraint operation. It also has no `message_unclear` action or calibrated confidence estimate. A grammar-valid result is not proof that a command is unambiguous or supported. Route these cases to operator review or a separately designed input policy; do not invent silent mappings.

Mandatory collision avoidance and field boundaries belong to deterministic autonomy safeguards. Region words must be resolved against a shared field coordinate frame by GNC/Perception. Schema changes require a new version, updated datasets/prompts/grammar, and new evaluations rather than relabelling the historical results.
