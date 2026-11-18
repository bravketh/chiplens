"""Tests for the operator-fusion model (M6).

Unfused: every intermediate tensor is written to memory and read back.
Fused: intermediates stay on chip, so that traffic disappears.

Hand-worked, Llama 3 70B (80 layers, hidden 8192), 2-byte activations, 12
intermediate tensors per layer, batch 512:
    one tensor = 512 * 8192 * 2           = 8,388,608 bytes
    written + read = 2 * 12 * 80 tensors  -> 2*12*80*8,388,608 = 16,106,127,360 B
                                                                = 16.106 GB
Launch overhead: 80 * 12 = 960 launches * 5 us = 4.8 ms.
Step time (weights 140 GB, 3 TB/s, compute 30 ms):
    unfused = max(30 ms, (140 + 16.106)/3000 = 52.035 ms) + 4.8 ms = 56.835 ms
    fused   = max(30 ms, 140/3000 = 46.667 ms) + 80 * 5 us (0.4 ms) = 47.067 ms
    speedup = 56.835 / 47.067 = 1.2075
"""

import pytest

from chiplens.fusion import (
    activation_traffic_gb,
    decode_step_time_s,
    kernel_launch_overhead_s,
    speedup,
)


def test_activation_traffic_unfused() -> None:
    got = activation_traffic_gb(512, 80, 8192, 2.0, 12, fused_fraction=0.0)
    assert got == pytest.approx(16.10612736)


def test_activation_traffic_fully_fused_is_zero() -> None:
    assert activation_traffic_gb(512, 80, 8192, 2.0, 12, fused_fraction=1.0) == 0.0


def test_activation_traffic_half_fused() -> None:
    got = activation_traffic_gb(512, 80, 8192, 2.0, 12, fused_fraction=0.5)
    assert got == pytest.approx(16.10612736 / 2)


def test_traffic_scales_with_batch() -> None:
    one = activation_traffic_gb(1, 80, 8192, 2.0, 12)
    big = activation_traffic_gb(512, 80, 8192, 2.0, 12)
    assert big == pytest.approx(512 * one)


def test_launch_overhead() -> None:
    assert kernel_launch_overhead_s(960, 5.0) == pytest.approx(0.0048)


def test_step_time_memory_bound_adds_launch() -> None:
    got = decode_step_time_s(140.0, 16.10612736, 3.0, 0.030, 0.0048)
    assert got == pytest.approx(0.056835, abs=1e-5)


def test_step_time_compute_bound() -> None:
    got = decode_step_time_s(140.0, 0.0, 3.0, 0.100, 0.0)
    assert got == pytest.approx(0.100)


def test_fusion_speedup_worked_example() -> None:
    unfused = decode_step_time_s(140.0, 16.10612736, 3.0, 0.030, 0.0048)
    fused = decode_step_time_s(140.0, 0.0, 3.0, 0.030, 0.0004)
    assert speedup(unfused, fused) == pytest.approx(1.2075, abs=1e-3)


def test_fusion_barely_helps_at_batch_1() -> None:
    """At batch 1 activations are tiny next to 140 GB of weights."""
    act = activation_traffic_gb(1, 80, 8192, 2.0, 12)  # 0.0315 GB
    unfused = decode_step_time_s(140.0, act, 3.0, 0.0, 0.0)
    fused = decode_step_time_s(140.0, 0.0, 3.0, 0.0, 0.0)
    assert speedup(unfused, fused) == pytest.approx(1.0, abs=0.001)


def test_bad_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        activation_traffic_gb(1, 80, 8192, 2.0, 12, fused_fraction=1.5)
    with pytest.raises(ValueError):
        kernel_launch_overhead_s(-1, 5.0)
    with pytest.raises(ValueError):
        speedup(1.0, 0.0)
