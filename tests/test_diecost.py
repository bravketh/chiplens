"""Tests for die cost, yield and chiplets (M5).

Hand-worked, 300 mm wafer, wafer cost $10,000, defect density 0.1 per cm^2.

Dies per wafer = pi*(d/2)^2 / A  -  pi*d / sqrt(2A)
    A = 100 mm^2: 70685.8/100 - 942.48/14.142 = 706.86 - 66.64 = 640.2 -> 640
    A = 600 mm^2: 70685.8/600 - 942.48/34.641 = 117.81 - 27.21 =  90.6 ->  90

Yield with A*D = area_in_cm2 * defects_per_cm2:
    Poisson: e^(-AD)                     A=100 mm^2 -> e^-0.1 = 0.904837
    Murphy:  ((1 - e^(-AD)) / AD)^2      A=100 -> (0.095163/0.1)^2 = 0.905597
                                         A=600 -> (0.451188/0.6)^2 = 0.565474

Monolithic 600 mm^2: 10000 / (90 * 0.565474)                = $196.49
Two 300 mm^2 chiplets (no overhead):
    dies per wafer = 235.62 - 38.48 = 197.1 -> 197
    Murphy(AD=0.3) = (0.259182/0.3)^2 = 0.746393
    one die = 10000 / (197 * 0.746393) = $68.01   -> two = $136.02
With 10% extra area per chiplet for die-to-die links (330 mm^2): about $155.8 total.
Small case, 100 mm^2: monolithic $17.25, 2 x 55 mm^2 chiplets about $17.68 (worse).
"""

import pytest

from chiplens.diecost import (
    chiplet_silicon_cost,
    cost_per_good_die,
    dies_per_wafer,
    murphy_yield,
    poisson_yield,
)

WAFER = 300.0
COST = 10_000.0
D0 = 0.1


def test_dies_per_wafer_small_die() -> None:
    assert dies_per_wafer(WAFER, 100.0) == 640


def test_dies_per_wafer_big_die() -> None:
    assert dies_per_wafer(WAFER, 600.0) == 90


def test_die_bigger_than_wafer_raises() -> None:
    with pytest.raises(ValueError):
        dies_per_wafer(WAFER, 100_000.0)


def test_poisson_yield() -> None:
    assert poisson_yield(100.0, D0) == pytest.approx(0.904837, abs=1e-6)


def test_murphy_yield() -> None:
    assert murphy_yield(100.0, D0) == pytest.approx(0.905597, abs=1e-5)
    assert murphy_yield(600.0, D0) == pytest.approx(0.565474, abs=1e-5)


def test_zero_defects_means_perfect_yield() -> None:
    assert murphy_yield(600.0, 0.0) == 1.0
    assert poisson_yield(600.0, 0.0) == 1.0


def test_bigger_die_yields_worse() -> None:
    assert murphy_yield(800.0, D0) < murphy_yield(100.0, D0)


def test_cost_per_good_die_monolithic() -> None:
    assert cost_per_good_die(COST, 90, 0.565474) == pytest.approx(196.49, abs=0.01)


def test_chiplets_beat_big_die() -> None:
    mono = chiplet_silicon_cost(600.0, 1, COST, WAFER, D0)
    two = chiplet_silicon_cost(600.0, 2, COST, WAFER, D0)
    assert mono == pytest.approx(196.49, abs=0.05)
    assert two == pytest.approx(136.02, abs=0.1)


def test_chiplet_overhead_raises_cost() -> None:
    two_overhead = chiplet_silicon_cost(600.0, 2, COST, WAFER, D0, d2d_overhead=0.1)
    assert two_overhead == pytest.approx(155.8, abs=0.5)


def test_chiplets_lose_for_small_chips() -> None:
    mono = chiplet_silicon_cost(100.0, 1, COST, WAFER, D0)
    two = chiplet_silicon_cost(100.0, 2, COST, WAFER, D0, d2d_overhead=0.1)
    assert mono == pytest.approx(17.25, abs=0.05)
    assert two > mono


def test_overhead_ignored_for_single_die() -> None:
    assert chiplet_silicon_cost(
        600.0, 1, COST, WAFER, D0, d2d_overhead=0.5
    ) == pytest.approx(chiplet_silicon_cost(600.0, 1, COST, WAFER, D0))


def test_bad_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        murphy_yield(0.0, D0)
    with pytest.raises(ValueError):
        murphy_yield(100.0, -0.1)
    with pytest.raises(ValueError):
        cost_per_good_die(COST, 0, 0.5)
    with pytest.raises(ValueError):
        cost_per_good_die(COST, 10, 0.0)
    with pytest.raises(ValueError):
        chiplet_silicon_cost(600.0, 0, COST, WAFER, D0)
