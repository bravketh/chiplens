"""Tests for power and efficiency (M4).

Hand-worked: a 700 W chip making 4557.14 tokens/s.
    energy per token = 700 / 4557.14      = 0.15360 J
    tokens per joule = 4557.14 / 700      = 6.5102
    rack: 40 kW budget, each chip really draws 700 * 1.5 = 1050 W
          (1.5 covers host, network, cooling share) -> floor(40000/1050) = 38
    rack throughput = 38 * 4557.14        = 173,171
    electricity: 700 W * PUE 1.3 = 910 W; 910 / 4557.14 = 0.19968 J per token
          per million tokens: 199,685 J / 3.6e6 = 0.055468 kWh * $0.08 = $0.004437
"""

import pytest

from chiplens.power import (
    chips_per_rack,
    electricity_cost_per_million_tokens,
    energy_per_token_j,
    rack_power_used_kw,
    rack_throughput,
    tokens_per_joule,
)


def test_energy_per_token() -> None:
    assert energy_per_token_j(700.0, 4557.14) == pytest.approx(0.1536, abs=1e-4)


def test_tokens_per_joule() -> None:
    assert tokens_per_joule(700.0, 4557.14) == pytest.approx(6.5102, abs=1e-4)


def test_energy_and_efficiency_are_inverses() -> None:
    assert energy_per_token_j(700.0, 100.0) * tokens_per_joule(700.0, 100.0) == (
        pytest.approx(1.0)
    )


def test_chips_per_rack_rounds_down() -> None:
    assert chips_per_rack(40.0, 700.0, 1.5) == 38


def test_rack_power_used() -> None:
    assert rack_power_used_kw(38, 700.0, 1.5) == pytest.approx(39.9)


def test_rack_throughput() -> None:
    assert rack_throughput(40.0, 700.0, 1.5, 4557.14) == pytest.approx(38 * 4557.14)


def test_chip_too_big_for_rack_gives_zero() -> None:
    assert chips_per_rack(0.5, 700.0, 1.5) == 0


def test_electricity_cost_per_million_tokens() -> None:
    got = electricity_cost_per_million_tokens(700.0, 4557.14, 0.08, 1.3)
    assert got == pytest.approx(0.004437, abs=2e-6)


def test_non_positive_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        energy_per_token_j(0.0, 10.0)
    with pytest.raises(ValueError):
        tokens_per_joule(700.0, 0.0)
    with pytest.raises(ValueError):
        chips_per_rack(40.0, 700.0, 0.5)  # overhead below 1 is impossible
