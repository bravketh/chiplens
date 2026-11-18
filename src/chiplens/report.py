"""Economics report: joins a scenario result to the die cost and economics models.

This is wiring, not new modeling. It reads cost inputs from an assumptions file
(``data/assumptions/``), so every number it uses carries a confidence tag.
"""

from __future__ import annotations

from typing import Any

from chiplens.catalog import spec_confidence, spec_value
from chiplens.diecost import chiplet_silicon_cost
from chiplens.economics import (
    chip_price,
    cogs_per_chip,
    cost_per_million_tokens,
    hourly_opex_cost,
    hourly_tco,
    monte_carlo,
    payback_months,
    percentiles,
    revenue_per_hour,
)
from chiplens.scenario import Result

_RANK = {"published": 0, "estimated": 1, "assumed": 2}

_NEEDED = (
    "wafer_cost",
    "wafer_diameter",
    "defect_density",
    "memory_and_packaging_cost",
    "chip_margin_multiplier",
    "lifetime",
    "power_price",
    "pue",
    "opex_fraction",
    "utilization",
    "price_per_million_tokens",
)


def economics_report(
    result: Result,
    assumptions: dict[str, Any],
    die_area_mm2: float,
    n_dies: int = 1,
    d2d_overhead: float = 0.0,
    sockets: int = 1,
    n_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Cost per token, payback and uncertainty for one scenario.

    Args:
        result: Output of ``scenario.analyze`` (gives tokens/s and power).
        assumptions: Loaded cost assumptions file.
        die_area_mm2: Total logic area per chip, in mm^2 (an input you supply).
        n_dies: Chiplets per chip (1 is monolithic).
        d2d_overhead: Extra area fraction per chiplet for die-to-die links.
        sockets: Chips the result covers (capex scales with it).
        n_samples: Monte Carlo runs.
        seed: Random seed for the Monte Carlo.

    Monte Carlo ranges (low, mode, high): utilization base -0.3 to +0.3
    (clamped to 0.05..1), selling price 0.5x to 2x, defect density 0.5x to 2x.

    Returns:
        A dictionary of costs, the p10/p50/p90 of cost per million tokens and
        payback, and the confidence tags of the assumptions used.
    """
    a = {k: spec_value(assumptions, k) for k in _NEEDED}
    tags = {k: spec_confidence(assumptions, k) for k in _NEEDED}

    def evaluate(util: float, price_per_m: float, defect: float) -> dict[str, float]:
        silicon = chiplet_silicon_cost(
            die_area_mm2,
            n_dies,
            a["wafer_cost"],
            a["wafer_diameter"],
            defect,
            d2d_overhead,
        )
        cogs = cogs_per_chip(silicon, a["memory_and_packaging_cost"])
        price = chip_price(cogs, a["chip_margin_multiplier"])
        capex = price * sockets
        tco = hourly_tco(
            capex,
            a["lifetime"],
            a["opex_fraction"],
            result.power_w,
            a["pue"],
            a["power_price"],
        )
        opex = hourly_opex_cost(
            capex, a["opex_fraction"], result.power_w, a["pue"], a["power_price"]
        )
        rev = revenue_per_hour(result.system_tokens_per_s, util, price_per_m)
        return {
            "silicon": silicon,
            "cogs": cogs,
            "price": price,
            "capex": capex,
            "tco": tco,
            "cost_m": cost_per_million_tokens(tco, result.system_tokens_per_s, util),
            "revenue": rev,
            "payback": payback_months(capex, rev - opex),
        }

    base = evaluate(
        a["utilization"], a["price_per_million_tokens"], a["defect_density"]
    )

    u, p, d = a["utilization"], a["price_per_million_tokens"], a["defect_density"]
    dists = {
        "util": (max(0.05, u - 0.3), u, min(1.0, u + 0.3)),
        "price_per_m": (0.5 * p, p, 2.0 * p),
        "defect": (0.5 * d, d, 2.0 * d),
    }
    cost_draws = monte_carlo(
        lambda **s: evaluate(**s)["cost_m"], dists, n_samples, seed
    )
    payback_draws = monte_carlo(
        lambda **s: evaluate(**s)["payback"], dists, n_samples, seed
    )

    assumed = sorted(k for k, v in tags.items() if v == "assumed")
    return {
        "silicon_cost": base["silicon"],
        "cogs_per_chip": base["cogs"],
        "price_per_chip": base["price"],
        "capex_total": base["capex"],
        "hourly_tco": base["tco"],
        "cost_per_million_tokens": base["cost_m"],
        "revenue_per_hour": base["revenue"],
        "payback_months": base["payback"],
        "cost_per_million_tokens_p10_p50_p90": percentiles(cost_draws, [10, 50, 90]),
        "payback_months_p10_p50_p90": percentiles(payback_draws, [10, 50, 90]),
        "monte_carlo_ranges": dists,
        "assumed_inputs": assumed,
        "worst_confidence": max(tags.values(), key=lambda v: _RANK[v]),
    }
