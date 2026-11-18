# Decisions

Each entry: what we chose, why, and what we gave up.

## D1: Models are pure functions in `src/chiplens/` (M0)

**Chose:** functions that take numbers and return numbers, no UI or file access inside.
**Why:** easy to test by hand, reusable from a CLI, a web app or a notebook.
**Gave up:** a little convenience, since callers must load data and pass it in.

## D2: Chip data is YAML with five keys per field (M0)

**Chose:** `value`, `unit`, `source_url`, `as_of`, `confidence` on every spec.
**Why:** makes it impossible to add a number without saying where it came from
and how much to trust it.
**Gave up:** files are more verbose than a plain table.

## D3: `source_url` may be null only for `assumed` (M0)

**Chose:** `published` and `estimated` fields must have an http(s) URL.
**Why:** an assumed number has no source by definition. Forcing a fake URL would
be worse than an honest blank.
**Gave up:** the validator cannot tell whether a URL really supports the number.

## D4: Validation is structural only, with `pyyaml` and hand-written checks (M0)

**Chose:** a small validator in `schema.py` with readable error messages.
**Why:** one dependency, and learners can read the whole thing.
**Gave up:** the features of a schema library (pydantic, JSON Schema). We can
switch later if the schema grows.

## D5: Python 3.11+, CI on 3.11 and 3.12 (M0)

**Why:** modern type syntax (`int | float`) and good error messages.

## D6: Roofline functions take plain numbers in practical units (M1)

**Chose:** GB, TB/s, TFLOPS, billions of params as arguments, not a chip object.
**Why:** hand-checkable against a calculator, and independent of the YAML loader.
**Gave up:** callers must pull numbers out of chip files themselves until M3.

## D7: Optimistic model, limits stated up front (M1)

**Chose:** ideal overlap of memory and compute, no KV cache.
**Why:** it gives the clean "balanced batch" idea first. The gaps are listed in
docs/methodology.md and are the backlog for later milestones and validation.
**Gave up:** realism. Results are upper bounds, not predictions.

## D8: Optional per-field `note`, not validated (M3)

**Chose:** a free-text `note` on any spec to record table numbers, conflicts and derivations.
**Why:** real sources disagree with themselves (SN40L HBM bandwidth is 1.8 TB/s in a
table and about 2 TB/s in the text). The note keeps the conflict visible next to the number.
**Gave up:** the note is not machine checked.

## D9: Dense compute for H100 is `estimated`, not `published` (M3)

**Chose:** half of NVIDIA's sparse TFLOPS figure, tagged estimated.
**Why:** the page lists numbers "with sparsity" and does not state the dense value.
**Gave up:** certainty. Replace with a dense figure from a datasheet when one is read.

## D10: Parameter count derived from architecture, tagged estimated (M3)

**Chose:** 70.55B for Llama 3 70B, from the published dimensions.
**Why:** no source read states a total. A test checks the stored value equals the formula.

## D11: SRAM is not a weight tier (M3)

**Chose:** weight tiers are HBM and DDR only.
**Why:** the SN40L paper gives no exact SRAM bandwidth ("hundreds of TB/s"). Using it would
mean inventing a number. 520 MB is negligible against tens of GB of weights anyway.

## D12: Ideal multi-chip scaling (M3)

**Chose:** sockets multiply capacity, bandwidth, compute and power with no network cost.
**Why:** simplest defensible default. It is the main reason multi-chip results are optimistic.

## D13: Streamlit for the web app, thin over `scenario.py` (M3)

**Chose:** Streamlit, no modeling code in the app, a headless smoke test in CI.
**Why:** deploys free, and keeps the logic testable without a browser.

## D14: Every cost input is `assumed` (M7)

**Chose:** a cost file where nothing is sourced, and output that says so.
**Why:** no public, citable wafer price, defect density or margin was found. Showing a
guess with a source would break the project's one rule.

## D15: Validation reports failure as the result (M8)

**Chose:** a committed report that says chiplens does not reproduce the SN40L numbers,
with a test that fails if the report goes stale.
**Why:** the credible claim is knowing where and why the model is wrong, not hiding it.

## D16: One arithmetic slip corrected in a test (M8)

The audit test first expected 2.2217 for 457 / (28800/140). The exact value is
63980 / 28800 = 2.22153. The expected value was my arithmetic error, not a code
issue, and was fixed with this note rather than by changing the code.
