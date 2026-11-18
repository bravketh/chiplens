"""Tests for the memory tier model (M2).

Hand-worked case: HBM 64 GB at 1.8 TB/s, DDR 1500 GB at 0.2 TB/s.

Weights 140 GB do not fit in HBM, so the fast tier fills first:
    64 GB in HBM, 76 GB in DDR
    time per token = 64 / 1800 + 76 / 200 = 0.035556 + 0.38 = 0.415556 s
    tokens/s       = 1 / 0.415556 = 2.4064
Weights 50 GB fit entirely in HBM: 1800 / 50 = 36 tokens/s.
"""

import pytest

from chiplens.memory_tiers import (
    Tier,
    effective_bandwidth_tb_s,
    place_weights,
    step_time_s,
    tokens_per_s,
)

HBM = Tier("hbm", 64.0, 1.8)
DDR = Tier("ddr", 1500.0, 0.2)


def test_all_weights_fit_in_fast_tier() -> None:
    assert place_weights(50.0, [DDR, HBM]) == [("hbm", 50.0)]


def test_overflow_spills_to_slower_tier_in_speed_order() -> None:
    # tiers passed in slow-first order on purpose
    assert place_weights(140.0, [DDR, HBM]) == [("hbm", 64.0), ("ddr", 76.0)]


def test_weights_that_fit_nowhere_raise() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        place_weights(2000.0, [HBM, DDR])


def test_step_time_one_tier() -> None:
    assert step_time_s(50.0, [HBM]) == pytest.approx(50 / 1800)


def test_step_time_two_tiers() -> None:
    assert step_time_s(140.0, [HBM, DDR]) == pytest.approx(0.415556, abs=1e-5)


def test_tokens_per_s_single_tier() -> None:
    assert tokens_per_s(50.0, [HBM]) == pytest.approx(36.0)


def test_tokens_per_s_spilled() -> None:
    assert tokens_per_s(140.0, [HBM, DDR]) == pytest.approx(2.4064, abs=1e-3)


def test_effective_bandwidth_between_the_two_tiers() -> None:
    # 140 GB / 0.415556 s = 336.9 GB/s = 0.3369 TB/s
    eff = effective_bandwidth_tb_s(140.0, [HBM, DDR])
    assert eff == pytest.approx(0.3369, abs=1e-3)
    assert DDR.bandwidth_tb_s < eff < HBM.bandwidth_tb_s


def test_effective_bandwidth_equals_tier_bandwidth_when_all_fit() -> None:
    assert effective_bandwidth_tb_s(50.0, [HBM]) == pytest.approx(1.8)


def test_bad_tier_rejected() -> None:
    with pytest.raises(ValueError):
        Tier("bad", 0.0, 1.0)
    with pytest.raises(ValueError):
        Tier("bad", 1.0, -1.0)
