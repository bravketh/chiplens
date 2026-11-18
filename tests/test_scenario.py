"""Tests for the scenario layer, which joins a chip, a model and the roofline.

Hand-worked cases:

A. SN40L x 16 sockets, Llama 3.1 405B, BF16, batch 1, no KV cache
   weights = 405 * 2 = 810 GB. HBM = 16 * 64 = 1024 GB, so all weights fit.
   bandwidth = 16 * 1.8 = 28.8 TB/s -> 28800 / 810 = 35.556 tokens/s.

B. SN40L x 1 socket, Llama 3 70B, BF16 (weights 141.1048 GB)
   64 GB in HBM, 77.1048 GB in DDR:
   time = 64/1800 + 77.1048/200 = 0.035556 + 0.385524 = 0.42108 s -> 2.3748 tok/s

C. H100, Llama 3 70B, FP8 (weights 70.5524 GB), batch 1
   3350 / 70.5524 = 47.48 tokens/s.
"""

import pytest

from chiplens.catalog import load
from chiplens.scenario import analyze


def run(chip: str, model: str, **kw):
    return analyze(load("chips", chip), load("models", model), **kw)


def test_case_a_sn40l_405b_batch1() -> None:
    r = run("sn40l", "llama31_405b", precision="bf16", sockets=16)
    assert r.weights_gb == pytest.approx(810.0)
    assert r.per_user_tokens_per_s == pytest.approx(28800 / 810)
    assert r.placement == [("hbm", 810.0)]


def test_case_b_weights_spill_to_ddr() -> None:
    r = run("sn40l", "llama3_70b", precision="bf16", sockets=1)
    assert r.placement[0][0] == "hbm"
    assert r.placement[1][0] == "ddr"
    assert r.per_user_tokens_per_s == pytest.approx(2.3748, abs=1e-3)


def test_case_c_h100_fp8() -> None:
    r = run("h100_sxm", "llama3_70b", precision="fp8", batch=1)
    assert r.per_user_tokens_per_s == pytest.approx(3350 / 70.552387584)


def test_balanced_batch_h100_fp8() -> None:
    # T_cmp = 1000*1979 / (2*70.552387584) ; T_mem = 3350/70.552387584 * 1000/1000
    # B* = (1979/2) / 3.35 = 295.37
    r = run("h100_sxm", "llama3_70b", precision="fp8")
    assert r.balanced_batch == pytest.approx(1979.0 / 2 / 3.35, rel=1e-6)


def test_throughput_is_capped_by_compute_at_large_batch() -> None:
    r = run("h100_sxm", "llama3_70b", precision="fp8", batch=2048)
    assert r.compute_utilization == pytest.approx(1.0)
    t_cmp = 1000 * 1979.0 / (2 * 70.552387584)
    assert r.system_tokens_per_s == pytest.approx(t_cmp)


def test_kv_cache_reduces_speed_and_is_reported() -> None:
    plain = run("h100_sxm", "llama3_70b", precision="fp8", batch=16)
    with_kv = run(
        "h100_sxm", "llama3_70b", precision="fp8", batch=16, context_tokens=8192
    )
    # 80*8*128*2 bytes per token (fp8 = 1 byte) * 8192 / 1e9 = 1.34218 GB per user
    assert with_kv.kv_gb_per_user == pytest.approx(1.342177, abs=1e-5)
    assert with_kv.per_user_tokens_per_s < plain.per_user_tokens_per_s


def test_max_batch_by_capacity_reported_with_kv() -> None:
    r = run("h100_sxm", "llama3_70b", precision="fp8", context_tokens=8192)
    # (80 - 70.552387584) / 1.342177 = 7.04 -> 7
    assert r.max_batch_by_capacity == 7


def test_kv_overflow_flagged() -> None:
    r = run("h100_sxm", "llama3_70b", precision="fp8", batch=64, context_tokens=8192)
    assert r.kv_fits is False


def test_power_metrics() -> None:
    r = run("h100_sxm", "llama3_70b", precision="fp8", batch=2048)
    assert r.power_w == pytest.approx(700.0)
    assert r.tokens_per_joule == pytest.approx(r.system_tokens_per_s / 700.0)


def test_sockets_scale_power() -> None:
    r = run("sn40l", "llama31_405b", precision="bf16", sockets=16)
    assert r.power_w == pytest.approx(16 * 600.0)


def test_assumed_inputs_are_reported() -> None:
    r = run("sn40l", "llama31_405b", precision="bf16", sockets=16)
    assert "board_power" in r.assumed_inputs
    assert r.worst_confidence == "assumed"


def test_estimated_inputs_are_reported() -> None:
    r = run("h100_sxm", "llama3_70b", precision="fp8")
    assert r.worst_confidence == "estimated"
    assert "peak_compute_fp8" in r.estimated_inputs


def test_missing_precision_spec_raises() -> None:
    with pytest.raises(ValueError, match="peak_compute_bf16"):
        run("example_chip", "llama3_70b", precision="bf16")


def test_weights_too_big_raise() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        run("example_chip", "llama31_405b", precision="int8")


def test_context_needs_architecture() -> None:
    with pytest.raises(ValueError, match="architecture"):
        run("sn40l", "llama31_405b", precision="bf16", sockets=16, context_tokens=8192)


def test_unknown_precision_raises() -> None:
    with pytest.raises(ValueError, match="precision"):
        run("sn40l", "llama3_70b", precision="fp4")
