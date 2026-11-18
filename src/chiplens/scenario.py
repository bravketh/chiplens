"""Scenario layer: run a chip and a model through the roofline models.

Takes loaded chip and model data, picks the right numbers, calls the pure
functions in ``roofline``, ``memory_tiers``, ``llm`` and reports every input's
confidence tag so no guess is shown as a fact.

Modeling choices (see docs/methodology.md):
- ``sockets`` chips are assumed to scale perfectly: capacity, bandwidth,
  compute and power all multiply by the socket count (ideal tensor parallelism,
  no network cost).
- Weights go in HBM first, then DDR. SRAM is not used as a weight tier.
- The KV cache is read from the fastest tier at the weights' precision, and
  its capacity is only checked against total memory.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from chiplens.catalog import has_spec, spec_confidence, spec_value
from chiplens.llm import head_dim, kv_gb_per_user, max_batch_by_capacity
from chiplens.memory_tiers import Tier, place_weights, step_time_s
from chiplens.roofline import compute_limit_tokens_per_s, weights_gb

PRECISION_BYTES = {"fp8": 1.0, "int8": 1.0, "bf16": 2.0, "fp16": 2.0}
_RANK = {"published": 0, "estimated": 1, "assumed": 2}


@dataclass(frozen=True)
class Result:
    """Everything one scenario reports. Units in the field names or docs."""

    weights_gb: float
    placement: list[tuple[str, float]]  # (tier, GB)
    per_user_tokens_per_s: float
    system_tokens_per_s: float
    balanced_batch: float  # without KV cache
    compute_utilization: float  # 0 to 1
    kv_gb_per_user: float
    kv_fits: bool
    max_batch_by_capacity: int | None  # None when there is no KV cache
    power_w: float
    tokens_per_joule: float
    used_inputs: dict[str, str]  # spec name -> confidence tag
    assumed_inputs: list[str]
    estimated_inputs: list[str]
    worst_confidence: str
    # Fields below feed the UI: the bottleneck, roofline and capacity pictures.
    batch: int
    capacity_gb: float  # total memory across tiers and sockets
    kv_total_gb: float  # batch * KV per user
    peak_tflops: float  # across all sockets
    effective_bandwidth_tb_s: float  # weights / step time, falls when weights spill
    fastest_bandwidth_tb_s: float  # bandwidth of the fastest tier, all sockets
    achieved_tflops: float  # peak * utilization
    memory_step_s: float  # seconds per step if memory were the only limit
    compute_step_s: float  # seconds per step if compute were the only limit
    bound: str  # "memory" or "compute"
    batch1_bound_tokens_per_s: float  # 1 / (time to stream the weights once)
    tokens_per_read: float  # speculative decoding: tokens per weight read
    efficiency: float  # fraction of the ideal speed actually reached


def chip_tiers(chip: dict[str, Any], sockets: int, used: dict[str, str]) -> list[Tier]:
    """Build weight tiers from a chip's specs, scaled by socket count."""

    def get(key: str, unit: str) -> float:
        used[key] = spec_confidence(chip, key)
        return spec_value(chip, key, unit)

    tiers: list[Tier] = []
    if has_spec(chip, "hbm_capacity"):
        tiers.append(
            Tier(
                "hbm",
                get("hbm_capacity", "GB") * sockets,
                get("hbm_bandwidth", "TB/s") * sockets,
            )
        )
        if has_spec(chip, "ddr_capacity"):
            tiers.append(
                Tier(
                    "ddr",
                    get("ddr_capacity", "GB") * sockets,
                    get("ddr_bandwidth", "TB/s") * sockets,
                )
            )
    else:
        tiers.append(
            Tier(
                "memory",
                get("memory_capacity", "GB") * sockets,
                get("memory_bandwidth", "TB/s") * sockets,
            )
        )
    return tiers


def analyze(
    chip: dict[str, Any],
    model: dict[str, Any],
    precision: str,
    sockets: int = 1,
    batch: int = 1,
    context_tokens: int = 0,
    tokens_per_read: float = 1.0,
    efficiency: float = 1.0,
) -> Result:
    """Run one scenario.

    Args:
        chip: Loaded chip data.
        model: Loaded model data.
        precision: One of fp8, int8, bf16, fp16. Sets bytes per weight and which
            ``peak_compute_<precision>`` spec is used.
        sockets: Chips working together (ideal scaling).
        batch: Concurrent users.
        context_tokens: Tokens of context per user (0 ignores the KV cache).
        tokens_per_read: Tokens produced per read of the weights. 1 is plain
            decoding. Above 1 models speculative decoding in its ideal form: the
            memory-limited speed scales up, the compute limit does not, and the
            extra compute spent on rejected draft tokens is ignored.
        efficiency: Fraction of the ideal speed a real system reaches (0 to 1).
            Use ``audit.calibration_factor`` to derive it from a measurement.

    Returns:
        A Result with speeds, limits, power and the confidence of every input.

    Raises:
        ValueError: On unknown precision, a missing spec, weights that do not
            fit, or a KV request for a model with no architecture data.
    """
    if precision not in PRECISION_BYTES:
        raise ValueError(f"precision must be one of {sorted(PRECISION_BYTES)}")
    if sockets < 1 or batch < 1 or context_tokens < 0:
        raise ValueError("sockets and batch must be >= 1 and context_tokens >= 0")
    if tokens_per_read < 1.0:
        raise ValueError(f"tokens_per_read must be >= 1, got {tokens_per_read}")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")

    used: dict[str, str] = {}
    bytes_per = PRECISION_BYTES[precision]

    params = spec_value(model, "params_billion")
    used["params_billion"] = spec_confidence(model, "params_billion")
    w = weights_gb(params, bytes_per)

    compute_key = f"peak_compute_{precision}"
    if not has_spec(chip, compute_key):
        raise ValueError(f"chip '{chip['name']}' has no {compute_key} spec")
    peak = spec_value(chip, compute_key, "TFLOPS") * sockets
    used[compute_key] = spec_confidence(chip, compute_key)

    tiers = chip_tiers(chip, sockets, used)
    placement = place_weights(w, tiers)
    step = step_time_s(w, tiers)
    fastest_bw = max(t.bandwidth_tb_s for t in tiers)
    total_capacity = sum(t.capacity_gb for t in tiers)

    kv_gb = 0.0
    if context_tokens > 0:
        arch = ("layers", "kv_heads", "hidden_size", "attention_heads")
        if not all(has_spec(model, k) for k in arch):
            raise ValueError(
                f"model '{model['name']}' has no architecture data "
                f"({', '.join(arch)}), so the KV cache cannot be modeled"
            )
        for k in arch:
            used[k] = spec_confidence(model, k)
        kv_gb = kv_gb_per_user(
            int(spec_value(model, "layers")),
            int(spec_value(model, "kv_heads")),
            head_dim(
                int(spec_value(model, "hidden_size")),
                int(spec_value(model, "attention_heads")),
            ),
            context_tokens,
            bytes_per,
        )

    t_cmp = compute_limit_tokens_per_s(peak, params)
    memory_time = step + batch * kv_gb / (1000.0 * fastest_bw)
    compute_time = batch / t_cmp
    per_user = efficiency * min(tokens_per_read / memory_time, 1.0 / compute_time)
    system = batch * per_user
    utilization = min(1.0, system * 2.0 * params / (1000.0 * peak))

    power = spec_value(chip, "board_power", "W") * sockets
    used["board_power"] = spec_confidence(chip, "board_power")

    assumed = sorted(k for k, v in used.items() if v == "assumed")
    estimated = sorted(k for k, v in used.items() if v == "estimated")
    worst = max(used.values(), key=lambda v: _RANK[v])

    return Result(
        weights_gb=w,
        placement=placement,
        per_user_tokens_per_s=per_user,
        system_tokens_per_s=system,
        balanced_batch=t_cmp * step,
        compute_utilization=utilization,
        kv_gb_per_user=kv_gb,
        kv_fits=(w + batch * kv_gb) <= total_capacity,
        max_batch_by_capacity=(
            max_batch_by_capacity(total_capacity, w, kv_gb) if kv_gb > 0 else None
        ),
        power_w=power,
        tokens_per_joule=system / power,
        used_inputs=used,
        assumed_inputs=assumed,
        estimated_inputs=estimated,
        worst_confidence=worst,
        batch=batch,
        capacity_gb=total_capacity,
        kv_total_gb=batch * kv_gb,
        peak_tflops=peak,
        effective_bandwidth_tb_s=w / (1000.0 * step),
        fastest_bandwidth_tb_s=fastest_bw,
        achieved_tflops=peak * utilization,
        memory_step_s=memory_time,
        compute_step_s=compute_time,
        bound="memory" if memory_time / tokens_per_read >= compute_time else "compute",
        batch1_bound_tokens_per_s=1.0 / step,
        tokens_per_read=tokens_per_read,
        efficiency=efficiency,
    )


def with_multipliers(
    chip: dict[str, Any],
    bandwidth: float = 1.0,
    compute: float = 1.0,
    capacity: float = 1.0,
) -> dict[str, Any]:
    """A what-if copy of a chip with bandwidth, compute and/or capacity scaled.

    Every ``*_bandwidth`` spec is multiplied by ``bandwidth``, every
    ``peak_compute_*`` spec by ``compute`` and every ``*_capacity`` spec by
    ``capacity``. A scaled spec is no longer a sourced
    number, so it is relabelled ``assumed`` and the chip's name says what-if. The
    original chip is never changed.

    Args:
        chip: Loaded chip data.
        bandwidth: Factor for all memory bandwidths (2.0 doubles them).
        compute: Factor for all peak compute figures.
        capacity: Factor for all memory capacities.

    Raises:
        ValueError: If a factor is not positive.
    """
    if bandwidth <= 0 or compute <= 0 or capacity <= 0:
        raise ValueError("multipliers must be positive")
    out = copy.deepcopy(chip)
    changed = False
    for key, field in out["specs"].items():
        factor = 1.0
        if key.endswith("_bandwidth"):
            factor = bandwidth
        elif key.startswith("peak_compute_"):
            factor = compute
        elif key.endswith("_capacity"):
            factor = capacity
        if factor != 1.0:
            field["value"] = field["value"] * factor
            field["confidence"] = "assumed"
            field["note"] = f"WHAT-IF: sourced value multiplied by {factor:g}."
            changed = True
    if changed:
        out["name"] = f"{chip['name']} (what-if)"
    return out
