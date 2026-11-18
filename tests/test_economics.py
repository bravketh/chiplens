"""Tests for the economics model (M7).

Hand-worked: a chip bought for $30,000, kept 4 years, drawing 700 W.
    capex per hour   = 30000 / (4 * 8760)                 = 0.856164 $/h
    opex per hour    = 10%/yr * 30000 / 8760              = 0.342466 $/h
                       + 0.7 kW * PUE 1.3 * $0.08         = 0.072800 $/h
                                                total opex = 0.415266 $/h
    total cost/hour  = 0.856164 + 0.415266                = 1.271430 $/h
Making 4557.14 tokens/s at 50% busy:
    tokens per hour  = 4557.14 * 3600 * 0.5               = 8,202,852
    cost per million = 1.271430 / 8.202852                = $0.15499
At a price of $0.60 per million tokens:
    revenue per hour = 8.202852 * 0.60                    = 4.921711 $/h
    net of opex      = 4.921711 - 0.415266                = 4.506445 $/h
    payback          = 30000 / (4.506445 * 730 h/month)   = 9.119 months
"""

import math

import pytest

from chiplens.economics import (
    chip_price,
    cogs_per_chip,
    cost_per_million_tokens,
    hourly_capex_cost,
    hourly_opex_cost,
    hourly_tco,
    monte_carlo,
    payback_months,
    percentiles,
    revenue_per_hour,
)


def test_cogs_and_price() -> None:
    assert cogs_per_chip(196.49, 4000.0) == pytest.approx(4196.49)
    assert chip_price(4196.49, 2.0) == pytest.approx(8392.98)


def test_hourly_capex() -> None:
    assert hourly_capex_cost(30000.0, 4.0) == pytest.approx(0.856164, abs=1e-6)


def test_hourly_opex() -> None:
    got = hourly_opex_cost(30000.0, 0.10, 700.0, 1.3, 0.08)
    assert got == pytest.approx(0.415266, abs=1e-6)


def test_hourly_tco() -> None:
    got = hourly_tco(30000.0, 4.0, 0.10, 700.0, 1.3, 0.08)
    assert got == pytest.approx(1.271430, abs=1e-6)


def test_cost_per_million_tokens() -> None:
    assert cost_per_million_tokens(1.271430, 4557.14, 0.5) == pytest.approx(
        0.15499, abs=1e-5
    )


def test_idle_chips_cost_more_per_token() -> None:
    busy = cost_per_million_tokens(1.27, 4557.14, 0.9)
    idle = cost_per_million_tokens(1.27, 4557.14, 0.1)
    assert idle == pytest.approx(9 * busy)


def test_revenue_per_hour() -> None:
    assert revenue_per_hour(4557.14, 0.5, 0.60) == pytest.approx(4.921711, abs=1e-5)


def test_payback_months() -> None:
    net = 4.921711 - 0.415266
    assert payback_months(30000.0, net) == pytest.approx(9.119, abs=1e-3)


def test_payback_infinite_when_losing_money() -> None:
    assert payback_months(30000.0, 0.0) == math.inf
    assert payback_months(30000.0, -1.0) == math.inf


def test_percentiles_interpolate() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentiles(values, [0, 25, 50, 100]) == [1.0, 2.0, 3.0, 5.0]
    assert percentiles([1.0, 2.0, 3.0, 4.0], [50]) == [2.5]


def test_percentiles_ignore_input_order() -> None:
    assert percentiles([5.0, 1.0, 3.0], [50]) == [3.0]


def test_monte_carlo_stays_in_range_and_centers() -> None:
    out = monte_carlo(lambda x: x, {"x": (1.0, 2.0, 3.0)}, n=4000, seed=7)
    assert len(out) == 4000
    assert all(1.0 <= v <= 3.0 for v in out)
    # triangular mean = (1 + 2 + 3) / 3 = 2
    assert sum(out) / len(out) == pytest.approx(2.0, abs=0.05)


def test_monte_carlo_same_seed_same_answer() -> None:
    a = monte_carlo(lambda x: x, {"x": (1.0, 2.0, 3.0)}, n=50, seed=1)
    b = monte_carlo(lambda x: x, {"x": (1.0, 2.0, 3.0)}, n=50, seed=1)
    c = monte_carlo(lambda x: x, {"x": (1.0, 2.0, 3.0)}, n=50, seed=2)
    assert a == b
    assert a != c


def test_monte_carlo_constant_input_is_deterministic() -> None:
    out = monte_carlo(
        lambda x, y: x * y, {"x": (2.0, 2.0, 2.0), "y": (3.0, 3.0, 3.0)}, 10, 0
    )
    assert out == [6.0] * 10


def test_bad_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        hourly_capex_cost(30000.0, 0.0)
    with pytest.raises(ValueError):
        cost_per_million_tokens(1.0, 100.0, 0.0)
    with pytest.raises(ValueError):
        cost_per_million_tokens(1.0, 100.0, 1.5)
    with pytest.raises(ValueError):
        monte_carlo(lambda x: x, {"x": (3.0, 2.0, 1.0)}, 10, 0)
