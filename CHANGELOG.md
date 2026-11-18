# Changelog

## [Unreleased]

## [1.2.0] - 2026-11-18

Product-manager workflow.

- Requirements check (`requirements.py`): a customer target in tokens/s per user passes
  or fails, with the smallest bandwidth, compute or capacity multiplier that closes the
  gap, and "will not help" for levers that cannot. Compares all chips against the target.
- Speculative decoding (`tokens_per_read`) and calibration (`efficiency`,
  `audit.calibration_factor`) in `scenario.analyze`.
- Claim triage in the app: published-claims library, implied tokens per weight read,
  one-click apply, and calibration from a measured result.
- One-page Markdown brief (`brief.py`) with a download button.
- Data: NVIDIA H200 SXM, Google TPU v5p and v5e (Google Cloud docs), Intel Gaudi 3
  (Intel white paper and product brief PDFs). TPU power is an assumed placeholder.
  Not added because no citable specs could be read: AMD MI300X, NVIDIA B200, Groq,
  Cerebras WSE-3.
- Includes the 1.1.0 changes below (released together).

## [1.1.0] - 2026-11-18 (shipped in the v1.2.0 tag)

Guided web app and what-if analysis.

- App: opens with a plain-English verdict (memory-, compute- or capacity-bound),
  a roofline with your operating point, step-time and memory-fit pictures, a batching
  explorer, an "Upgrades" view comparing 2x bandwidth, compute and capacity, a chip
  comparison, a claim checker with a SambaNova demo, cost, validation and a Learn view.
  Preset stories (chat, API server, long documents). Persistent navigation.
- `insight.diagnose`: tested verdict engine. `scenario.with_multipliers`: what-if chips
  (scaled specs are relabelled assumed). More fields on `scenario.Result`.
- Docs: `docs/LEARNING_GUIDE.md`.

## [1.0.0] - 2026-11-18

First complete release.

- M8: claim auditor (bandwidth and compute bounds), validation against SN40L
  Table IV with a generated report, `chiplens audit-user`, `audit-total`, `validate`.
  Finding: the model does not reproduce the published numbers (errors +73%, -55%, -72%).
- M7: economics (COGS, TCO, cost per million tokens, payback, Monte Carlo),
  `chiplens economics`, economics and validation sections in the web app.
- Docs: methodology with formulas, decisions log, write-up, contributing guide, MIT license.

## [0.2.0] - 2026-11-18

Physical and manufacturing models.

- M6: operator-fusion model (activation traffic and kernel-launch overhead).
- M5: die cost, wafer geometry, Poisson and Murphy yield, chiplet comparison.
- M4: power and efficiency (energy per token, tokens per joule, rack capacity,
  electricity cost).

## [0.1.0] - 2026-11-18

First live-demo release.

- M3: chip and model YAML data with sources (SN40L, H100 SXM, Llama 3 70B, 405B, 8B),
  catalog loader with unit conversion, scenario layer that reports input confidence,
  `chiplens` command line tool, Streamlit web app.
- M2: memory tier model (SRAM/HBM/DDR placement), KV cache sizing, parameter counting.
- M1: roofline and batching model for LLM decoding, demo chart.
- M0: package skeleton, CI, chip data schema with validator.
