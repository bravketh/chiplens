# chiplens

[![CI](https://github.com/OWNER/chiplens/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/chiplens/actions/workflows/ci.yml)

<!-- TODO: replace OWNER with your GitHub username once the repo is public. -->

An open-source tradeoff and cost model for AI inference chips.

## The problem

People who choose, design or sell AI inference chips keep asking the same
question: **for this model, what limits this chip, and what does a token cost?**
The answer depends on memory bandwidth, compute, batch size, power and die
cost, and public specs are scattered and often unsourced.

chiplens puts those into one transparent model:

- Compares real chips using **public, cited specs**. Every number carries a unit,
  a source URL, a date and a confidence tag (`published`, `estimated`, `assumed`).
- Lets anyone add their own chip or model as a YAML file.
- Checks its own predictions against numbers vendors publish, and shows the
  errors honestly.

## Status

| Milestone | What | State |
| --- | --- | --- |
| M0 | Scaffold, CI, chip data schema | done |
| M1 | Roofline and batching model | next |
| M2 | Memory tiers | planned |
| M3 | Chip and model data, CLI, first web page (v0.1) | planned |

Nothing is modeled yet. The only data file is a placeholder chip whose
numbers are all `assumed`.

## Develop

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

## Layout

```
src/chiplens/   pure-Python models, no UI code
data/chips/     one YAML file per chip (see data/chips/README.md)
data/models/    LLM definitions (from M3)
validation/     predicted vs published numbers (from M8)
docs/           methodology and decisions
tests/          hand-worked expected values
```
