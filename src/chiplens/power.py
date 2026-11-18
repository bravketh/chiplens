"""Power and efficiency (M4).

Turns speed and chip power into energy per token, tokens per joule, rack
capacity and electricity cost.

Units: power in W, speeds in tokens/s, rack budgets in kW, electricity price in
USD per kWh. One joule is one watt for one second. 1 kWh = 3.6e6 J.

What this models: steady draw at the stated power while producing tokens.
What it ignores: power varying with load (idle chips draw less than board
power, busy ones may draw more), cooling beyond a single overhead factor, and
network or host power unless folded into ``overhead``.
"""

from __future__ import annotations

import math

JOULES_PER_KWH = 3.6e6


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def energy_per_token_j(power_w: float, tokens_per_s: float) -> float:
    """Energy to make one token.

    E = P / T      (W divided by tokens/s gives J per token)
    """
    _require_positive(power_w=power_w, tokens_per_s=tokens_per_s)
    return power_w / tokens_per_s


def tokens_per_joule(power_w: float, tokens_per_s: float) -> float:
    """Efficiency: tokens produced per joule. T / P, the inverse of energy per token."""
    _require_positive(power_w=power_w, tokens_per_s=tokens_per_s)
    return tokens_per_s / power_w


def chips_per_rack(rack_power_kw: float, chip_power_w: float, overhead: float) -> int:
    """Whole chips that fit a rack power budget.

    n = floor( 1000 * rack_kw / (chip_w * overhead) )

    Args:
        rack_power_kw: Rack power budget in kW.
        chip_power_w: Chip power in W.
        overhead: Multiplier for host, network and cooling share (1.0 means none).
    """
    _require_positive(rack_power_kw=rack_power_kw, chip_power_w=chip_power_w)
    if overhead < 1.0:
        raise ValueError(f"overhead must be >= 1, got {overhead}")
    return math.floor(1000.0 * rack_power_kw / (chip_power_w * overhead))


def rack_power_used_kw(n_chips: int, chip_power_w: float, overhead: float) -> float:
    """Power drawn by ``n_chips`` including overhead, in kW."""
    return n_chips * chip_power_w * overhead / 1000.0


def rack_throughput(
    rack_power_kw: float,
    chip_power_w: float,
    overhead: float,
    tokens_per_s_per_chip: float,
) -> float:
    """Tokens/s from a power-limited rack."""
    return chips_per_rack(rack_power_kw, chip_power_w, overhead) * tokens_per_s_per_chip


def electricity_cost_per_million_tokens(
    chip_power_w: float, tokens_per_s: float, price_per_kwh: float, pue: float
) -> float:
    """Electricity cost to make a million tokens, in USD.

    cost = (P * PUE / T) * 1e6 / 3.6e6 * price

    PUE (power usage effectiveness) is the datacenter overhead multiplier on
    chip power, 1.0 or higher.
    """
    _require_positive(price_per_kwh=price_per_kwh)
    if pue < 1.0:
        raise ValueError(f"pue must be >= 1, got {pue}")
    joules = energy_per_token_j(chip_power_w * pue, tokens_per_s) * 1e6
    return joules / JOULES_PER_KWH * price_per_kwh
