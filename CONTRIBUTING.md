# Contributing

## Add a chip

1. Copy `data/chips/example_chip.yaml` to `data/chips/<your_chip>.yaml`.
2. Fill in each spec with `value`, `unit`, `source_url`, `as_of` and `confidence`.
   - `published`: the source states the number.
   - `estimated`: you derived it from published numbers. Say how in `note`.
   - `assumed`: a guess. `source_url` may be `null`. Never present a guess as published.
3. Use these spec names so the scenario layer finds them:
   `peak_compute_<precision>` (TFLOPS), `board_power` (W), and either
   `memory_capacity` (GB) + `memory_bandwidth` (TB/s) or `hbm_*` / `ddr_*` tiers.
4. Run `pytest`. The catalog test loads every file and rejects schema problems.

## Add a model

Copy `data/models/llama3_70b.yaml`. Required: `params_billion`. For KV cache
modeling also add `layers`, `hidden_size`, `attention_heads`, `kv_heads`.

## Add a validation case

Add a file in `validation/data/` with the published numbers and their source,
then regenerate the report with `chiplens validate --write`. Show errors honestly,
including when the model is badly wrong.

## Rules for code

- Models are pure functions in `src/chiplens/`, with type hints and docstrings
  that state units.
- Write the test first, with a hand-worked expected value in the docstring.
  Never change an expected value just to make a test pass. Fix the code, or fix
  your arithmetic and say so in the commit message.
- `ruff check .`, `ruff format --check .` and `pytest` must pass.
- Conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`).
- Record real design choices in `docs/decisions.md`.
