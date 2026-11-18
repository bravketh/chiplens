"""Tests for the roofline and batching model (M1).

Reference case, worked by hand:
    70B params, 8-bit (1 byte each), 3 TB/s, 638 TFLOPS peak.
    One token reads every weight once and needs 2 ops per weight.

    weights          = 70 * 1                    = 70 GB
    memory limit     = 3000 GB/s / 70 GB         = 42.857 tokens/s
    compute limit    = 638e12 / (2 * 70e9)       = 4557.14 tokens/s
    balanced batch   = 4557.14 / 42.857          = 106.33
    at 6 TB/s        = 6000 / 70                 = 85.714 tokens/s, batch 53.17
    batch 512 / user = 4557.14 / 512             = 8.90 tokens/s
"""

import pytest

from chiplens.roofline import (
    arithmetic_intensity,
    balanced_batch,
    compute_limit_tokens_per_s,
    compute_utilization,
    memory_limit_tokens_per_s,
    ridge_point,
    system_throughput,
    tokens_per_s_per_user,
    weights_gb,
)

P = 70.0  # billions of parameters
B_PARAM = 1.0  # bytes per parameter (8-bit)
BW = 3.0  # TB/s
F = 638.0  # TFLOPS


def test_weights_gb() -> None:
    assert weights_gb(P, B_PARAM) == pytest.approx(70.0)


def test_weights_gb_16bit_doubles() -> None:
    assert weights_gb(P, 2.0) == pytest.approx(140.0)


def test_memory_limit() -> None:
    assert memory_limit_tokens_per_s(BW, 70.0) == pytest.approx(3000 / 70)
    assert memory_limit_tokens_per_s(BW, 70.0) == pytest.approx(42.86, abs=0.01)


def test_compute_limit() -> None:
    assert compute_limit_tokens_per_s(F, P) == pytest.approx(638000 / 140)
    assert compute_limit_tokens_per_s(F, P) == pytest.approx(4557.14, abs=0.01)


def test_balanced_batch() -> None:
    assert balanced_batch(BW, F, P, B_PARAM) == pytest.approx(44660 / 420)
    assert balanced_batch(BW, F, P, B_PARAM) == pytest.approx(106.33, abs=0.01)


def test_doubling_bandwidth_doubles_batch1_speed() -> None:
    assert tokens_per_s_per_user(1, 6.0, F, P, B_PARAM) == pytest.approx(
        85.71, abs=0.01
    )


def test_doubling_bandwidth_halves_balanced_batch() -> None:
    assert balanced_batch(6.0, F, P, B_PARAM) == pytest.approx(53.17, abs=0.01)


def test_batch1_is_memory_bound() -> None:
    assert tokens_per_s_per_user(1, BW, F, P, B_PARAM) == pytest.approx(42.86, abs=0.01)


def test_per_user_speed_flat_below_balanced_batch() -> None:
    at_1 = tokens_per_s_per_user(1, BW, F, P, B_PARAM)
    at_50 = tokens_per_s_per_user(50, BW, F, P, B_PARAM)
    assert at_50 == pytest.approx(at_1)


def test_per_user_speed_at_batch_512() -> None:
    assert tokens_per_s_per_user(512, BW, F, P, B_PARAM) == pytest.approx(
        8.90, abs=0.01
    )


def test_system_throughput_grows_then_caps() -> None:
    assert system_throughput(1, BW, F, P, B_PARAM) == pytest.approx(3000 / 70)
    assert system_throughput(10, BW, F, P, B_PARAM) == pytest.approx(10 * 3000 / 70)
    assert system_throughput(512, BW, F, P, B_PARAM) == pytest.approx(4557.14, abs=0.01)


def test_compute_utilization_at_batch_1() -> None:
    # (3000/70) / (638000/140) = 0.009404...
    assert compute_utilization(1, BW, F, P, B_PARAM) == pytest.approx(0.0094, abs=1e-4)


def test_compute_utilization_is_full_at_and_above_balanced_batch() -> None:
    b_star = balanced_batch(BW, F, P, B_PARAM)
    assert compute_utilization(b_star, BW, F, P, B_PARAM) == pytest.approx(1.0)
    assert compute_utilization(512, BW, F, P, B_PARAM) == pytest.approx(1.0)


def test_arithmetic_intensity_8bit() -> None:
    # 2 ops per weight per token, 1 byte per weight: 2 ops/byte per batch item
    assert arithmetic_intensity(1, B_PARAM) == pytest.approx(2.0)
    assert arithmetic_intensity(100, B_PARAM) == pytest.approx(200.0)


def test_arithmetic_intensity_16bit_is_half() -> None:
    assert arithmetic_intensity(1, 2.0) == pytest.approx(1.0)


def test_ridge_point() -> None:
    # 638 TFLOPS / 3 TB/s = 212.67 ops/byte
    assert ridge_point(BW, F) == pytest.approx(212.67, abs=0.01)


def test_ridge_point_matches_balanced_batch() -> None:
    # B* = ridge / (2 / bytes_per_param) for 8-bit: 212.67 / 2 = 106.33
    assert ridge_point(BW, F) / arithmetic_intensity(1, B_PARAM) == pytest.approx(
        balanced_batch(BW, F, P, B_PARAM)
    )


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_inputs_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        weights_gb(bad, B_PARAM)
    with pytest.raises(ValueError):
        memory_limit_tokens_per_s(bad, 70.0)
    with pytest.raises(ValueError):
        compute_limit_tokens_per_s(bad, P)
    with pytest.raises(ValueError):
        tokens_per_s_per_user(bad, BW, F, P, B_PARAM)
