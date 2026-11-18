"""Die cost, yield and chiplets (M5).

A wafer is a round slice of silicon cut into rectangular dies. Some dies have a
fatal defect, so you pay for the whole wafer but sell only the good ones:

    cost per good die = wafer cost / (dies per wafer * yield)

Big dies lose twice: fewer fit on the wafer, and each is more likely to catch a
defect. Splitting a big chip into smaller chiplets recovers yield but spends
extra area on die-to-die links and adds packaging complexity.

Units: areas in mm^2, wafer diameter in mm, defect density in defects per cm^2,
costs in USD.

What this models: the standard dies-per-wafer estimate, Poisson and Murphy
yield, equal-sized chiplets with known-good-die testing.
What it ignores: wafer edge exclusion, scribe lanes, reticle-size limits,
defect clustering beyond Murphy's formula, test cost, packaging and assembly
yield (see ``economics`` for packaging), and parametric yield (speed binning).
"""

from __future__ import annotations

import math


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def dies_per_wafer(wafer_diameter_mm: float, die_area_mm2: float) -> int:
    """Whole dies that fit on a round wafer.

    N = floor( pi * (d/2)^2 / A  -  pi * d / sqrt(2 * A) )

    The second term removes partial dies lost around the edge.

    Raises:
        ValueError: If not even one die fits.
    """
    _require_positive(wafer_diameter_mm=wafer_diameter_mm, die_area_mm2=die_area_mm2)
    gross = math.pi * (wafer_diameter_mm / 2) ** 2 / die_area_mm2
    edge = math.pi * wafer_diameter_mm / math.sqrt(2 * die_area_mm2)
    n = math.floor(gross - edge)
    if n < 1:
        raise ValueError(
            f"a {die_area_mm2:g} mm^2 die does not fit "
            f"on a {wafer_diameter_mm:g} mm wafer"
        )
    return n


def _area_times_density(die_area_mm2: float, defect_density_per_cm2: float) -> float:
    _require_positive(die_area_mm2=die_area_mm2)
    if defect_density_per_cm2 < 0:
        raise ValueError(
            f"defect_density_per_cm2 must be >= 0, got {defect_density_per_cm2}"
        )
    return die_area_mm2 / 100.0 * defect_density_per_cm2


def poisson_yield(die_area_mm2: float, defect_density_per_cm2: float) -> float:
    """Fraction of dies with no defect, assuming defects land independently.

    Y = e^(-A*D)
    """
    return math.exp(-_area_times_density(die_area_mm2, defect_density_per_cm2))


def murphy_yield(die_area_mm2: float, defect_density_per_cm2: float) -> float:
    """Murphy yield, which allows for defects clustering on the wafer.

    Y = ((1 - e^(-A*D)) / (A*D))^2   (A in cm^2, D in defects per cm^2)

    Less pessimistic than Poisson for large dies.
    """
    ad = _area_times_density(die_area_mm2, defect_density_per_cm2)
    if ad == 0:
        return 1.0
    return ((1.0 - math.exp(-ad)) / ad) ** 2


def cost_per_good_die(
    wafer_cost: float, dies_per_wafer_: int, yield_fraction: float
) -> float:
    """Wafer cost spread over the good dies only, in USD."""
    _require_positive(wafer_cost=wafer_cost)
    if dies_per_wafer_ < 1:
        raise ValueError(f"dies_per_wafer_ must be >= 1, got {dies_per_wafer_}")
    if not 0 < yield_fraction <= 1:
        raise ValueError(f"yield_fraction must be in (0, 1], got {yield_fraction}")
    return wafer_cost / (dies_per_wafer_ * yield_fraction)


def chiplet_silicon_cost(
    total_area_mm2: float,
    n_dies: int,
    wafer_cost: float,
    wafer_diameter_mm: float,
    defect_density_per_cm2: float,
    d2d_overhead: float = 0.0,
) -> float:
    """Silicon cost of one chip built from ``n_dies`` equal chiplets, in USD.

    Each chiplet has area (total / n) * (1 + overhead) when n > 1, where
    ``d2d_overhead`` is the extra area fraction spent on die-to-die links. A
    single die pays no overhead. Uses Murphy yield and assumes bad chiplets are
    discarded before assembly (known-good-die).

    Args:
        total_area_mm2: Logic area you need, before link overhead.
        n_dies: Chiplets per chip (1 means monolithic).
        wafer_cost: Cost of one wafer in USD.
        wafer_diameter_mm: Wafer diameter in mm.
        defect_density_per_cm2: Defects per cm^2.
        d2d_overhead: Extra area fraction per chiplet, e.g. 0.1 for 10%.

    Returns:
        Silicon cost in USD (packaging not included).
    """
    if n_dies < 1:
        raise ValueError(f"n_dies must be >= 1, got {n_dies}")
    _require_positive(total_area_mm2=total_area_mm2)
    area = total_area_mm2 / n_dies
    if n_dies > 1:
        area *= 1.0 + d2d_overhead
    dpw = dies_per_wafer(wafer_diameter_mm, area)
    y = murphy_yield(area, defect_density_per_cm2)
    return n_dies * cost_per_good_die(wafer_cost, dpw, y)
