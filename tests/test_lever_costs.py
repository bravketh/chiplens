"""Tests for lever costs: what each way of closing a gap costs, per chip.

Hand cases. Unit costs (USD of extra cost per chip for +100% of a lever):
    bandwidth 1000, compute 400, capacity 300.
Cost is linear in the multiplier: cost = (m - 1) * unit_cost.
    m = 1.5 bandwidth -> 0.5 * 1000 = 500
    m = 3.0 compute   -> 2.0 * 400  = 800
    m = 1.0 anything  -> 0 (not needed)
    m = None          -> None (cannot fix, so no price)
Cheapest is the lowest cost among levers that can fix the gap.
"""

import pytest

from chiplens.catalog import load
from chiplens.lever_costs import cheapest_lever, lever_costs, unit_costs_from

UNIT = {"bandwidth": 1000.0, "compute": 400.0, "capacity": 300.0}


def test_cost_is_linear_in_multiplier() -> None:
    need = {"bandwidth": 1.5, "compute": 3.0, "capacity": 2.0}
    cost = lever_costs(need, UNIT)
    assert cost["bandwidth"] == pytest.approx(500.0)
    assert cost["compute"] == pytest.approx(800.0)
    assert cost["capacity"] == pytest.approx(300.0)


def test_met_requirement_costs_nothing() -> None:
    need = {"bandwidth": 1.0, "compute": 1.0, "capacity": 1.0}
    assert set(lever_costs(need, UNIT).values()) == {0.0}


def test_levers_that_cannot_fix_have_no_price() -> None:
    need = {"bandwidth": 2.0, "compute": None, "capacity": None}
    cost = lever_costs(need, UNIT)
    assert cost["bandwidth"] == pytest.approx(1000.0)
    assert cost["compute"] is None and cost["capacity"] is None


def test_cheapest_picks_lowest_cost_not_lowest_multiplier() -> None:
    # Bandwidth needs the smaller multiplier (1.5x) but compute is cheaper per unit.
    need = {"bandwidth": 1.5, "compute": 1.6, "capacity": None}
    assert cheapest_lever(need, UNIT) == ("compute", pytest.approx(240.0))


def test_cheapest_is_none_when_nothing_helps() -> None:
    need = {"bandwidth": None, "compute": None, "capacity": None}
    assert cheapest_lever(need, UNIT) is None


def test_cheapest_is_none_when_nothing_needed() -> None:
    need = {"bandwidth": 1.0, "compute": 1.0, "capacity": 1.0}
    assert cheapest_lever(need, UNIT) is None


def test_missing_unit_cost_rejected() -> None:
    with pytest.raises(KeyError):
        lever_costs(
            {"bandwidth": 2.0, "compute": 1.0, "capacity": 1.0}, {"bandwidth": 1}
        )


def test_data_file_loads_all_assumed() -> None:
    data = load("assumptions", "lever_costs")
    costs = unit_costs_from(data)
    assert set(costs) == {"bandwidth", "compute", "capacity"}
    assert all(v > 0 for v in costs.values())
    assert all(s["confidence"] == "assumed" for s in data["specs"].values())
