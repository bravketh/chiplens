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
    per_user = min(1.0 / memory_time, t_cmp / batch)
    system = batch * per_user

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
        compute_utilization=min(1.0, system * 2.0 * params / (1000.0 * peak)),
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
    )
