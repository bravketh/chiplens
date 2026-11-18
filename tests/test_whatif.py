"""Tests for what-if chips: scale a chip's bandwidth or compute and compare.

Roofline facts to confirm with the real data (H100 SXM, Llama 3 70B, fp8):
    doubling bandwidth doubles batch-1 speed and halves the balanced batch;
    doubling compute leaves batch-1 speed alone and doubles the balanced batch.
Anything scaled is relabelled `assumed`, because it is no longer a sourced spec.
"""

import copy

import pytest

from chiplens.catalog import load, spec_confidence, spec_value
from chiplens.scenario import analyze, with_multipliers

CHIP = load("chips", "h100_sxm")
MODEL = load("models", "llama3_70b")


def run(chip):
    return analyze(chip, MODEL, precision="fp8", batch=1)


def test_double_bandwidth_doubles_batch1_speed() -> None:
    base = run(CHIP)
    fast = run(with_multipliers(CHIP, bandwidth=2.0))
    assert fast.per_user_tokens_per_s == pytest.approx(2 * base.per_user_tokens_per_s)


def test_double_bandwidth_halves_balanced_batch() -> None:
    base = run(CHIP)
    fast = run(with_multipliers(CHIP, bandwidth=2.0))
    assert fast.balanced_batch == pytest.approx(base.balanced_batch / 2)


def test_double_compute_keeps_batch1_speed_and_doubles_balanced_batch() -> None:
    base = run(CHIP)
    strong = run(with_multipliers(CHIP, compute=2.0))
    assert strong.per_user_tokens_per_s == pytest.approx(base.per_user_tokens_per_s)
    assert strong.balanced_batch == pytest.approx(2 * base.balanced_batch)


def test_original_chip_is_not_changed() -> None:
    before = copy.deepcopy(CHIP)
    with_multipliers(CHIP, bandwidth=3.0, compute=0.5)
    assert CHIP == before


def test_scaled_specs_are_marked_assumed_and_named() -> None:
    chip = with_multipliers(CHIP, bandwidth=2.0)
    assert spec_confidence(chip, "memory_bandwidth") == "assumed"
    assert spec_value(chip, "memory_bandwidth") == pytest.approx(6.7)
    assert "what-if" in chip["name"].lower()
    # untouched specs keep their tags
    assert spec_confidence(chip, "memory_capacity") == "published"


def test_multiplier_of_one_changes_nothing() -> None:
    chip = with_multipliers(CHIP)
    assert chip["specs"] == CHIP["specs"]
    assert chip["name"] == CHIP["name"]


def test_tiered_chip_scales_both_tiers() -> None:
    sn = load("chips", "sn40l")
    chip = with_multipliers(sn, bandwidth=2.0)
    assert spec_value(chip, "hbm_bandwidth", "TB/s") == pytest.approx(3.6)
    assert spec_value(chip, "ddr_bandwidth", "TB/s") == pytest.approx(0.4)


def test_whatif_result_reports_assumed_input() -> None:
    r = run(with_multipliers(CHIP, bandwidth=2.0))
    assert "memory_bandwidth" in r.assumed_inputs


def test_non_positive_multiplier_rejected() -> None:
    with pytest.raises(ValueError):
        with_multipliers(CHIP, bandwidth=0.0)
    with pytest.raises(ValueError):
        with_multipliers(CHIP, compute=-1.0)


def test_double_capacity_doubles_max_batch_headroom() -> None:
    """Capacity only matters once the KV cache is in play."""
    base = analyze(CHIP, MODEL, precision="fp8", context_tokens=8192)
    big = analyze(
        with_multipliers(CHIP, capacity=2.0),
        MODEL,
        precision="fp8",
        context_tokens=8192,
    )
    # (80 - 70.552) / 1.342 = 7 users; (160 - 70.552) / 1.342 = 66 users
    assert base.max_batch_by_capacity == 7
    assert big.max_batch_by_capacity == 66


def test_capacity_does_not_change_speed() -> None:
    base = analyze(CHIP, MODEL, precision="fp8", batch=1)
    big = analyze(with_multipliers(CHIP, capacity=4.0), MODEL, precision="fp8", batch=1)
    assert big.per_user_tokens_per_s == pytest.approx(base.per_user_tokens_per_s)


def test_capacity_specs_marked_assumed() -> None:
    chip = with_multipliers(CHIP, capacity=2.0)
    assert spec_value(chip, "memory_capacity") == pytest.approx(160.0)
    assert spec_confidence(chip, "memory_capacity") == "assumed"
    assert spec_confidence(chip, "memory_bandwidth") == "published"


def test_capacity_multiplier_must_be_positive() -> None:
    with pytest.raises(ValueError):
        with_multipliers(CHIP, capacity=0.0)
