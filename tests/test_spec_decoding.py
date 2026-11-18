"""Tests for speculative decoding and calibration.

Speculative decoding: a small model drafts several tokens and the big model checks
them in one pass over the weights, so one weight read can yield more than one
token. Modeled as `tokens_per_read` (k >= 1) multiplying the memory-limited speed:

    per user = efficiency * min( k / memory_step , 1 / compute_step )

Hand-worked, H100 fp8, Llama 3 70B (70.552387584 GB), batch 1:
    base      = 3350 / 70.552387584            = 47.4833 tokens/s
    k = 2.5   = 2.5 * 47.4833                  = 118.708 tokens/s
    batch 2048 is compute-bound, so k does nothing there.
    efficiency 0.5 halves both regimes.

Calibration: efficiency = measured / predicted, e.g. 1042 / 1800 = 0.5789.
"""

import pytest

from chiplens.audit import calibration_factor
from chiplens.catalog import load
from chiplens.scenario import analyze

W = 70.552387584


def run(**kw):
    return analyze(
        load("chips", "h100_sxm"), load("models", "llama3_70b"), precision="fp8", **kw
    )


def test_default_is_unchanged() -> None:
    assert run().per_user_tokens_per_s == pytest.approx(3350 / W)


def test_speculative_decoding_multiplies_memory_bound_speed() -> None:
    assert run(tokens_per_read=2.5).per_user_tokens_per_s == pytest.approx(
        2.5 * 3350 / W
    )


def test_speculative_decoding_does_not_help_when_compute_bound() -> None:
    base = run(batch=2048)
    spec = run(batch=2048, tokens_per_read=3.0)
    assert spec.per_user_tokens_per_s == pytest.approx(base.per_user_tokens_per_s)


def test_efficiency_scales_both_regimes() -> None:
    for batch in (1, 2048):
        base = run(batch=batch)
        half = run(batch=batch, efficiency=0.5)
        assert half.per_user_tokens_per_s == pytest.approx(
            0.5 * base.per_user_tokens_per_s
        )


def test_batch1_bound_stays_the_physical_one_read_bound() -> None:
    """The claim checker needs the bound without speculative decoding."""
    assert run(tokens_per_read=4.0).batch1_bound_tokens_per_s == pytest.approx(3350 / W)


def test_result_records_the_factors() -> None:
    r = run(tokens_per_read=2.0, efficiency=0.8)
    assert r.tokens_per_read == 2.0
    assert r.efficiency == 0.8


def test_bad_factors_rejected() -> None:
    with pytest.raises(ValueError):
        run(tokens_per_read=0.5)
    with pytest.raises(ValueError):
        run(efficiency=0.0)
    with pytest.raises(ValueError):
        run(efficiency=1.5)


def test_calibration_factor() -> None:
    assert calibration_factor(1042.0, 1800.0) == pytest.approx(0.5789, abs=1e-4)


def test_calibration_can_exceed_one_and_says_so_by_value() -> None:
    # measured above the bound: factor above 1 means the bound is broken
    assert calibration_factor(457.0, 204.1) == pytest.approx(2.2391, abs=1e-3)


def test_calibration_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        calibration_factor(0.0, 100.0)
    with pytest.raises(ValueError):
        calibration_factor(100.0, 0.0)


def test_h200_is_present_and_faster_than_h100_at_batch_1() -> None:
    model = load("models", "llama3_70b")
    h100 = analyze(load("chips", "h100_sxm"), model, precision="fp8")
    h200 = analyze(load("chips", "h200_sxm"), model, precision="fp8")
    # 4.8 / 3.35 = 1.4328
    assert h200.per_user_tokens_per_s / h100.per_user_tokens_per_s == pytest.approx(
        4.8 / 3.35
    )
