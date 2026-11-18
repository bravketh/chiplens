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
