# Chip data files

One YAML file per chip. Every spec field has five keys:

| Key | Meaning |
| --- | --- |
| `value` | A number (not a string) |
| `unit` | Plain text, e.g. `TB/s`, `TFLOPS`, `GB`, `W` |
| `source_url` | Where the number comes from. Required for `published` and `estimated`, may be `null` for `assumed` |
| `as_of` | ISO date (`YYYY-MM-DD`) the source was read |
| `confidence` | `published` (stated by the vendor or a paper), `estimated` (derived from published numbers), `assumed` (a guess we flag openly) |

Rules:

- Never invent a spec. If you cannot cite it, tag it `assumed`.
- `example_chip.yaml` is a placeholder and is all `assumed`.

The rules are enforced by `src/chiplens/schema.py` and `tests/test_schema.py`.
