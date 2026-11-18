# Methodology

For each model: what it computes, what it assumes, and what it ignores.
Everything is a pure function in `src/chiplens/`. Units: parameters in billions,
sizes in GB, bandwidth in TB/s, compute in TFLOPS, speeds in tokens/s.

The headline caveat: **every result is an optimistic upper bound**, not a
prediction. See [validation/results.md](../validation/results.md) for how far off
that can be.

## Decode, in one paragraph

Generating one token means streaming every weight from memory once and doing 2
operations per weight (a multiply and an add). A batch of $B$ users shares one
read of the weights, so arithmetic per byte grows with $B$. A chip is limited by
whichever is slower, memory or compute.

## M1: roofline and batching (`roofline.py`)

| Quantity | Formula |
| --- | --- |
| Weights (GB) | $W = P \cdot b$ |
| Memory limit, one user (tokens/s) | $T_{mem} = \dfrac{1000\,\mathrm{BW}}{W}$ |
| Compute limit, all users (tokens/s) | $T_{cmp} = \dfrac{1000\,F}{2P}$ |
| Balanced batch | $B^{*} = \dfrac{T_{cmp}}{T_{mem}}$ |
| Per-user speed | $t(B) = \min\!\left(T_{mem},\ \dfrac{T_{cmp}}{B}\right)$ |
| System throughput | $B \cdot t(B)$ |
| Compute utilization | $\min(1,\ B/B^{*})$ |
| Arithmetic intensity (ops/byte) | $I = \dfrac{2B}{b}$ |
| Ridge point (ops/byte) | $\dfrac{F}{\mathrm{BW}}$ |

$P$ is parameters (billions), $b$ bytes per weight, $F$ peak TFLOPS.

**Assumes:** each token reads every weight once and does 2 ops per weight; peak
bandwidth and peak compute are reached; memory and compute overlap perfectly.

**Ignores:** attention arithmetic, prefill, real utilization, overheads.

## M2: memory tiers, KV cache, model size (`memory_tiers.py`, `llm.py`)

Weights fill the fastest tier first. Time per token is the sum over tiers:

$$t = \sum_i \frac{\mathrm{GB}_i}{1000\,\mathrm{BW}_i}, \qquad \text{tokens/s} = \frac{1}{t}$$

KV cache per user (K and V for every layer and past token):

$$\text{bytes per token} = 2 \cdot L \cdot n_{kv} \cdot d_{head} \cdot b, \qquad \text{KV} = \text{bytes per token} \cdot \text{context}$$

With a KV cache each step reads $W + B \cdot \mathrm{KV}$:

$$t(B) = \min\!\left(\frac{1000\,\mathrm{BW}}{W + B\cdot \mathrm{KV}},\ \frac{T_{cmp}}{B}\right)$$

Parameter count of a Llama-style model:

$$P = L\left(2d^2 + 2d\,n_{kv}d_{head} + 3\,d\,d_{ffn}\right) + k\,V d$$

with $k=2$ for separate input and output embeddings and $k=1$ if tied.

**Assumes:** sequential reads across tiers (so a spilled model is pessimistic if
a real chip overlaps tiers); KV cache read from the fastest tier at the weights'
precision; SRAM not used as a weight tier.

**Ignores:** prefetching and caching hot weights, KV compression, norm weights
(about 0.002% of 70B), mixture-of-experts, attention FLOPs.

## M3: scenario layer (`scenario.py`, `catalog.py`)

Wiring, not new physics. Picks numbers from chip and model files, converts units,
calls the functions above, and reports the **confidence tag of every input used**.

**Assumes:** $n$ sockets scale perfectly: capacity, bandwidth, compute and power
all multiply by $n$ (ideal tensor parallelism, zero network cost). This is the
biggest simplification for multi-chip results.

## M4: power and efficiency (`power.py`)

$$E_{token} = \frac{P_{chip}}{T}\ \text{(J)}, \qquad \text{tokens per joule} = \frac{T}{P_{chip}}$$

$$n_{rack} = \left\lfloor \frac{1000\,P_{rack}}{P_{chip}\cdot o} \right\rfloor$$

where $o\ge 1$ covers host, network and cooling share. Electricity cost per
million tokens is $\dfrac{P\cdot \mathrm{PUE}}{T}\cdot\dfrac{10^6}{3.6\times10^6}\cdot c_{kWh}$.

**Assumes:** steady draw at board power. **Ignores:** idle and part-load power,
the real split between chip and system power.

## M5: die cost, yield, chiplets (`diecost.py`)

$$N_{dies} = \left\lfloor \frac{\pi (d/2)^2}{A} - \frac{\pi d}{\sqrt{2A}} \right\rfloor$$

$$Y_{Poisson} = e^{-AD}, \qquad Y_{Murphy} = \left(\frac{1 - e^{-AD}}{AD}\right)^{2}$$

with $A$ in cm$^2$ and $D$ in defects per cm$^2$. Cost per good die is
$\dfrac{C_{wafer}}{N_{dies}\,Y}$. A chip built from $n$ equal chiplets uses
$A_{each} = \dfrac{A}{n}(1+\delta)$, where $\delta$ is die-to-die area overhead
(only when $n>1$), and costs $n$ times one good chiplet (known-good-die).

**Ignores:** reticle limit, scribe lanes, test cost, packaging and assembly
yield, defect clustering beyond Murphy, speed binning.

## M6: operator fusion (`fusion.py`)

$$\text{traffic} = 2\,m\,L\,(B\,d\,b_a)\,(1-f), \qquad t_{step} = \max\!\left(t_{cmp},\ \frac{W + \text{traffic}}{1000\,\mathrm{BW}}\right) + n_{launch}\,t_{launch}$$

with $m$ intermediate tensors per layer, $f$ the fused fraction.

**What it shows:** at batch 1 the extra traffic is about 0.03 GB against 140 GB
of weights, so fusion barely helps decode memory traffic. It matters through
launch overhead and at large batch. The SN40L paper reports far larger fusion
gains on prefill and training; this model does not cover those.

**Ignores:** the wider MLP intermediates (understating traffic), attention score
matrices, overlap of launches with execution. $m$ and the launch time are inputs
you choose, not measured values.

## M7: economics (`economics.py`, `report.py`)

$$\text{price} = \text{COGS}\cdot\mu, \qquad \text{capex per hour} = \frac{\text{capex}}{8760\,Y_{life}}$$

$$\text{opex per hour} = \frac{f_{opex}\,\text{capex}}{8760} + \frac{P}{1000}\cdot\mathrm{PUE}\cdot c_{kWh}$$

$$\text{cost per million tokens} = \frac{\text{hourly cost}}{T\cdot 3600\cdot u / 10^6}, \qquad \text{payback (months)} = \frac{\text{capex}}{730\,(\text{revenue} - \text{opex})}$$

$u$ is utilization. Monte Carlo draws utilization (base $\pm0.3$), selling price
($0.5\times$ to $2\times$) and defect density ($0.5\times$ to $2\times$) from
triangular distributions and reports p10, p50 and p90.

**Every cost input is `assumed`** (`data/assumptions/cost_example.yaml`). Because
throughput is an upper bound, cost per token is a floor and payback is
optimistic. **Ignores:** financing and discounting, resale value, price decline,
network and host cost, demand limits.

## M8: claim auditor and validation (`audit.py`, `validation.py`)

Two hard bounds: per-user speed $\le \dfrac{1000\,\mathrm{BW}}{W}$ and total
throughput $\le \dfrac{1000\,F}{2P}$. A claim above a bound is not automatically
false; the auditor lists what could explain it (speculative decoding, smaller
effective weights, mixture-of-experts, a different batch or precision, an error).

Validation compares the batch-1 bandwidth bound with Table IV of the SN40L paper
and reports signed errors. Current result: errors of +73%, -55% and -72%. The
model does not reproduce the published numbers. See
[validation/results.md](../validation/results.md).

## Speculative decoding and calibration (`scenario.py`, `audit.py`)

$$t(B) = \eta \cdot \min\!\left(\frac{k}{t_{mem}},\ \frac{1}{t_{cmp}}\right)$$

with $k \ge 1$ tokens per weight read and $\eta \le 1$ the efficiency. $t_{mem}$ is the
memory time per step and $t_{cmp} = B / T_{cmp}$ the compute time per step.
Calibration is $\eta = \dfrac{\text{measured}}{\text{predicted}}$.

**Assumes:** speculative decoding is ideal: memory-limited speed scales by $k$, compute
does not, and the compute spent on rejected draft tokens is ignored. $\eta$ is a single
constant across batch sizes, which real systems do not obey.

**Ignores:** draft-model cost, acceptance rate varying with content, and efficiency that
changes with batch or context.

## Requirements check (`requirements.py`)

A target passes when speed per user is at least $T$ **and** the KV caches fit. For each
lever (bandwidth, compute, capacity) on its own, bisection finds the smallest multiplier
(up to 16x) that passes, or reports that no multiplier helps. Scaled specs are
hypotheticals, relabelled `assumed`.

**Ignores:** the cost of a lever (a 2x bandwidth part may cost far more than 2x compute),
levers moving together, and any limit other than the roofline.
