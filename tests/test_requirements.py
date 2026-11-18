"""Tests for the requirements check: does a chip meet a customer target, and if
not, how much more of each lever would close the gap?

A requirement is "at least T tokens/s per user, for B users, at context C", and it
also needs everything to fit in memory.

Hand-worked, H100 SXM fp8, Llama 3 70B (70.552387584 GB), 80 GB, 3.35 TB/s:

  batch 1: speed = 3350 / 70.552 = 47.483 tokens/s (memory-bound)
     target 40  -> met.
     target 95  -> bandwidth needed = 95 / 47.483 = 2.0007x.
                   More compute or capacity changes nothing at batch 1.

  batch 2048: compute-bound, per user = 14024.7 / 2048 = 6.848 tokens/s
     target 10  -> compute needed = 10 / 6.848 = 1.460x.
                   Bandwidth does nothing: memory is not the limit here.

  batch 64, 8K context: KV = 1.342 GB per user, total need
     70.552 + 64 * 1.342 = 156.45 GB against 80 GB.
     speed = 3350 / ... = 21.4 tokens/s, so a target of 15 is met on speed but
     the batch does not fit -> capacity needed = 156.45 / 80 = 1.956x.
"""

import pytest

from chiplens.catalog import load
from chiplens.requirements import meets_target, required_multipliers
from chiplens.scenario import analyze

CHIP = load("chips", "h100_sxm")
MODEL = load("models", "llama3_70b")


def need(target, **kw):
    return required_multipliers(CHIP, MODEL, "fp8", target, **kw)


def test_target_below_current_speed_is_met() -> None:
    r = analyze(CHIP, MODEL, precision="fp8")
    assert meets_target(r, 40.0) is True
    assert meets_target(r, 95.0) is False


def test_nothing_needed_when_already_met() -> None:
    assert need(40.0) == {"bandwidth": 1.0, "compute": 1.0, "capacity": 1.0}


def test_memory_bound_gap_needs_bandwidth_only() -> None:
    got = need(95.0)
    assert got["bandwidth"] == pytest.approx(95 / 47.483, abs=0.01)
    assert got["compute"] is None
    assert got["capacity"] is None


def test_compute_bound_gap_needs_compute_only() -> None:
    got = need(10.0, batch=2048)
    assert got["compute"] == pytest.approx(1.460, abs=0.01)
    assert got["bandwidth"] is None


def test_capacity_gap_needs_capacity_only() -> None:
    got = need(15.0, batch=64, context_tokens=8192)
    assert got["capacity"] == pytest.approx(156.45 / 80, abs=0.01)
    assert got["bandwidth"] is None


def test_unreachable_target_returns_none() -> None:
    got = need(1e6, max_multiplier=4.0)
    assert got == {"bandwidth": None, "compute": None, "capacity": None}


def test_speculative_decoding_lowers_the_bandwidth_needed() -> None:
    plain = need(95.0)["bandwidth"]
    spec = need(95.0, tokens_per_read=2.0)
    # with k = 2 the chip already makes 94.97 tokens/s, so almost nothing is needed
    assert spec["bandwidth"] < plain
    assert spec["bandwidth"] == pytest.approx(1.0, abs=0.01)


def test_bad_target_rejected() -> None:
    with pytest.raises(ValueError):
        need(0.0)
