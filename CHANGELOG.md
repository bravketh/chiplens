# Changelog

## [Unreleased]

## [0.1.0] - 2026-11-18

First live-demo release.

- M3: chip and model YAML data with sources (SN40L, H100 SXM, Llama 3 70B, 405B, 8B),
  catalog loader with unit conversion, scenario layer that reports input confidence,
  `chiplens` command line tool, Streamlit web app.
- M2: memory tier model (SRAM/HBM/DDR placement), KV cache sizing, parameter counting.
- M1: roofline and batching model for LLM decoding, demo chart.
- M0: package skeleton, CI, chip data schema with validator.
