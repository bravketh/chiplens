"""Economics (M7): COGS, total cost of ownership, cost per token, payback, risk.

Chain of ideas:
    COGS (what it costs to make a chip) -> price (COGS times a margin multiplier)
    -> capex for the buyer -> hourly cost (amortized capex + opex + power)
    -> cost per million tokens -> payback if tokens are sold at a price.

Units: USD, hours, years, tokens/s, power in W, price per kWh. A year is 8760
hours and a month is 730 hours.

What this models: straight-line amortization, opex as a share of purchase price,
steady utilization, a constant selling price per token.
What it ignores: financing cost and discounting, resale value, price decline
over time, networking and host server cost (fold into capex), ramp-up, failures,
taxes, and demand limits (it assumes every token produced is sold).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

HOURS_PER_YEAR = 8760.0
HOURS_PER_MONTH = HOURS_PER_YEAR / 12.0


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def cogs_per_chip(silicon_cost: float, memory_and_packaging_cost: float) -> float:
    """Cost of goods sold for one chip: silicon plus memory, packaging and test."""
    return silicon_cost + memory_and_packaging_cost


def chip_price(cogs: float, margin_multiplier: float) -> float:
    """Selling price = COGS * margin multiplier (2.0 means 50% gross margin)."""
    _require_positive(cogs=cogs, margin_multiplier=margin_multiplier)
    return cogs * margin_multiplier


def hourly_capex_cost(capex: float, lifetime_years: float) -> float:
    """Straight-line purchase cost per hour: capex / (years * 8760)."""
    _require_positive(capex=capex, lifetime_years=lifetime_years)
    return capex / (lifetime_years * HOURS_PER_YEAR)


def hourly_opex_cost(
    capex: float,
    opex_fraction_per_year: float,
    power_w: float,
    pue: float,
    price_per_kwh: float,
) -> float:
    """Running cost per hour: yearly upkeep share plus electricity.

    opex = f * capex / 8760  +  (P / 1000) * PUE * price
    """
    _require_positive(capex=capex, power_w=power_w, price_per_kwh=price_per_kwh)
    if opex_fraction_per_year < 0:
        raise ValueError("opex_fraction_per_year must be >= 0")
    if pue < 1.0:
        raise ValueError(f"pue must be >= 1, got {pue}")
    upkeep = opex_fraction_per_year * capex / HOURS_PER_YEAR
    electricity = power_w / 1000.0 * pue * price_per_kwh
    return upkeep + electricity


def hourly_tco(
    capex: float,
    lifetime_years: float,
    opex_fraction_per_year: float,
    power_w: float,
    pue: float,
    price_per_kwh: float,
) -> float:
    """Total cost of ownership per hour: amortized capex plus opex."""
    return hourly_capex_cost(capex, lifetime_years) + hourly_opex_cost(
        capex, opex_fraction_per_year, power_w, pue, price_per_kwh
    )


def _tokens_per_hour_millions(tokens_per_s: float, utilization: float) -> float:
    _require_positive(tokens_per_s=tokens_per_s)
    if not 0 < utilization <= 1:
        raise ValueError(f"utilization must be in (0, 1], got {utilization}")
    return tokens_per_s * 3600.0 * utilization / 1e6


def cost_per_million_tokens(
    hourly_cost: float, tokens_per_s: float, utilization: float
) -> float:
    """Cost to produce a million tokens, in USD.

    cost = hourly cost / (tokens/s * 3600 * utilization / 1e6)

    Idle time still costs money, so low utilization raises the cost per token.
    """
    _require_positive(hourly_cost=hourly_cost)
    return hourly_cost / _tokens_per_hour_millions(tokens_per_s, utilization)


def revenue_per_hour(
    tokens_per_s: float, utilization: float, price_per_million_tokens: float
) -> float:
    """Income per hour from selling tokens at a price per million, in USD."""
    _require_positive(price_per_million_tokens=price_per_million_tokens)
    return _tokens_per_hour_millions(tokens_per_s, utilization) * (
        price_per_million_tokens
    )


def payback_months(capex: float, net_per_hour: float) -> float:
    """Months of operation to earn back the purchase price.

    payback = capex / (net per hour * 730). Infinite if net per hour is not
    positive. ``net_per_hour`` is revenue minus opex (not minus amortized capex,
    which would count the purchase twice).
    """
    _require_positive(capex=capex)
    if net_per_hour <= 0:
        return math.inf
    return capex / (net_per_hour * HOURS_PER_MONTH)


def percentiles(values: list[float], qs: list[float]) -> list[float]:
    """Percentiles (0 to 100) by linear interpolation between sorted values."""
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    last = len(ordered) - 1
    out = []
    for q in qs:
        if not 0 <= q <= 100:
            raise ValueError(f"percentile must be in [0, 100], got {q}")
        rank = q / 100.0 * last
        lo = math.floor(rank)
        frac = rank - lo
        if frac == 0 or ordered[lo] == ordered[min(lo + 1, last)]:
            out.append(ordered[lo])
        else:
            out.append(ordered[lo] + frac * (ordered[lo + 1] - ordered[lo]))
    return out


def monte_carlo(
    model_fn: Callable[..., float],
    distributions: dict[str, tuple[float, float, float]],
    n: int,
    seed: int,
) -> list[float]:
    """Run a model many times with uncertain inputs.

    Each input is drawn from a triangular distribution (low, mode, high): most
    likely near the mode, never outside the range. The same seed gives the same
    draws, so results are reproducible.

    Args:
        model_fn: Function taking the inputs as keyword arguments, returning a number.
        distributions: Input name -> (low, mode, high).
        n: Number of runs.
        seed: Random seed.

    Returns:
        The n outputs, in draw order.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    for name, (low, mode, high) in distributions.items():
        if not low <= mode <= high:
            raise ValueError(
                f"{name}: need low <= mode <= high, got {(low, mode, high)}"
            )
    rng = random.Random(seed)
    return [
        model_fn(
            **{
                name: rng.triangular(low, high, mode)
                for name, (low, mode, high) in distributions.items()
            }
        )
        for _ in range(n)
    ]
