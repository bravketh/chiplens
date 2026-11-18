"""Tests for the claim auditor.

A decode speed claim per user cannot beat BW / weights unless each weight read
produces more than one token (speculative decoding), weights are smaller than
assumed (quantization, sparsity, mixture of experts), or the claim is wrong.

Hand-worked, 16 sockets at 1.8 TB/s = 28.8 TB/s, 70B at BF16 = 140 GB:
    bound = 28800 / 140 = 205.714 tokens/s per user
    claim 457 -> 457 * 140 / 28800 = 63980 / 28800 = 2.22153 times the bound
Total throughput bound for 638 TFLOPS and 70B: 1000 * 638 / 140 = 4557.14.
"""

import pytest

from chiplens.audit import audit_per_user_claim, audit_total_claim


def test_claim_above_bound_is_flagged() -> None:
    r = audit_per_user_claim(457.0, 28.8, 140.0)
    assert r.bound == pytest.approx(205.714, abs=1e-3)
    assert r.ratio == pytest.approx(2.22153, abs=1e-5)
    assert r.verdict == "exceeds bound"
    assert "speculative" in r.explanation.lower()


def test_claim_below_bound_is_plausible() -> None:
    r = audit_per_user_claim(100.0, 28.8, 140.0)
    assert r.verdict == "within bound"
    assert r.ratio == pytest.approx(100 / 205.714, abs=1e-4)


def test_claim_exactly_at_bound_is_within() -> None:
    bound = 28800 / 140
    assert audit_per_user_claim(bound, 28.8, 140.0).verdict == "within bound"


def test_total_claim_above_compute_bound() -> None:
    r = audit_total_claim(10000.0, 638.0, 70.0)
    assert r.bound == pytest.approx(4557.14, abs=0.01)
    assert r.verdict == "exceeds bound"
    assert r.ratio == pytest.approx(10000 / 4557.14, abs=1e-3)


def test_total_claim_within_bound() -> None:
    assert audit_total_claim(1000.0, 638.0, 70.0).verdict == "within bound"


def test_bad_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        audit_per_user_claim(0.0, 28.8, 140.0)
    with pytest.raises(ValueError):
        audit_total_claim(100.0, 0.0, 70.0)
