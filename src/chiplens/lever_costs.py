"""Lever costs: price each way of closing a gap, so the cheapest can be picked.

``requirements.required_multipliers`` says how much of each lever is needed. This
module turns that into dollars with a simple model: raising a lever by +100% (2x)
costs a fixed amount per chip, and cost is linear in the multiplier:

    cost = (m - 1) * unit_cost        [USD per chip, m is the multiplier]

The unit costs are assumptions (see data/assumptions/lever_costs.yaml). The useful
output is the ranking, not the dollar figure.
"""

from __future__ import annotations

from typing import Any

from chiplens.requirements import LEVERS

_SPEC_KEYS = {
    "bandwidth": "bandwidth_cost_per_100pct",
    "compute": "compute_cost_per_100pct",
    "capacity": "capacity_cost_per_100pct",
}


def unit_costs_from(data: dict[str, Any]) -> dict[str, float]:
    """Read the three unit costs (USD per chip per +100%) from an assumptions file."""
    return {
        lever: float(data["specs"][key]["value"]) for lever, key in _SPEC_KEYS.items()
    }


def lever_costs(
    need: dict[str, float | None], unit_costs: dict[str, float]
) -> dict[str, float | None]:
    """Extra cost per chip of each lever's required multiplier.

    Args:
        need: Output of ``required_multipliers`` (multiplier, 1.0 or None).
        unit_costs: USD per chip for +100% of each lever.

    Returns:
        USD per chip for each lever. 0.0 if not needed, None if it cannot fix the gap.

    Raises:
        KeyError: If a lever has no unit cost.
    """
    out: dict[str, float | None] = {}
    for lever in LEVERS:
        m = need[lever]
        out[lever] = None if m is None else (m - 1.0) * unit_costs[lever]
    return out


def cheapest_lever(
    need: dict[str, float | None], unit_costs: dict[str, float]
) -> tuple[str, float] | None:
    """The lowest-cost lever that closes the gap, with its cost (USD per chip).

    Returns None if nothing is needed (all costs 0) or no lever can fix the gap.
    """
    costs = lever_costs(need, unit_costs)
    priced = {k: v for k, v in costs.items() if v is not None and v > 0}
    if not priced:
        return None
    best = min(priced, key=priced.__getitem__)
    return best, priced[best]
