# Methodology

For each model: what it computes, what it assumes, and what it ignores.

## M1: roofline and batching for LLM decoding

**Code:** `src/chiplens/roofline.py`. **Demo:** `examples/m1_roofline_chart.py`.

Units: parameters in billions, weights in GB, bandwidth in TB/s, compute in
TFLOPS, speeds in tokens/s.

| Quantity | Formula |
| --- | --- |
| Weights (GB) | W = P * b |
| Memory limit (tokens/s, one user) | T_mem = 1000 * BW / W |
| Compute limit (tokens/s, all users) | T_cmp = 1000 * F / (2 * P) |
| Balanced batch | B* = T_cmp / T_mem |
| Per-user speed | t(B) = min(T_mem, T_cmp / B) |
| System throughput | B * t(B) |
| Compute utilization | min(1, B / B*) |
| Arithmetic intensity (ops/byte) | 2 * B / b |
| Ridge point (ops/byte) | F / BW |

**Assumes**

- Each token reads every weight exactly once, and does 2 ops per weight.
- A batch of B users shares one read of the weights.
- The chip reaches its peak bandwidth and peak compute.
- Memory and compute overlap perfectly, so the slower one alone sets the speed.

**Ignores (and how this makes the model optimistic)**

- KV cache: attention reads a per-user cache that grows with context length, so
  large batches are limited by memory far sooner than B* suggests.
- Attention arithmetic, activations and norms.
- Prefill (processing the prompt), which behaves very differently from decode.
- Multi-chip scaling, networking, and weights split across devices.
- Real utilization: kernels reach well under peak bandwidth and compute.
- Mixture-of-experts models, where only some weights are read per token.

**Validation:** none yet. No claim of agreement with any vendor result until
`validation/` shows it.
