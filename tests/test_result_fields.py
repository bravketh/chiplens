"""Tests for the extra fields on a scenario Result that the UI draws.

Hand-worked: H100 SXM, Llama 3 70B (70.552387584 GB at 1 byte), fp8, 1979 TFLOPS,
3.35 TB/s, 80 GB.

    memory step  = W / (1000 * BW)   = 70.552387584 / 3350        = 0.0210604 s
    compute step = B * 2P / (1000 F * 1e9...) = B * 141.104775168 / 1,979,000
                 = B * 7.1301e-5 s
    batch 1:    memory step is longer, so the step is memory-bound.
                utilization = 7.1301e-5 / 0.0210604 = 0.003385
                achieved TFLOPS = 1979 * 0.003385  = 6.70
    batch 2048: compute step = 0.14601 s > 0.02106 s, so compute-bound,
                achieved = peak = 1979 TFLOPS.
    KV at 8192 tokens, fp8: 80*8*128*2*8192 / 1e9 = 1.342177 GB per user.
"""

import pytest

from chiplens.catalog import load
from chiplens.scenario import analyze

W = 70.552387584


def run(**kw):
    return analyze(
        load("chips", "h100_sxm"), load("models", "llama3_70b"), precision="fp8", **kw
    )


def test_steps_at_batch_1() -> None:
    r = run(batch=1)
    assert r.memory_step_s == pytest.approx(W / 3350)
    assert r.compute_step_s == pytest.approx(141.104775168 / 1_979_000)
    assert r.bound == "memory"


def test_steps_at_batch_2048() -> None:
    r = run(batch=2048)
    assert r.compute_step_s == pytest.approx(2048 * 141.104775168 / 1_979_000)
    assert r.bound == "compute"


def test_achieved_tflops_is_peak_times_utilization() -> None:
    for batch in (1, 64, 2048):
        r = run(batch=batch)
        assert r.achieved_tflops == pytest.approx(r.peak_tflops * r.compute_utilization)


def test_achieved_tflops_values() -> None:
    assert run(batch=1).achieved_tflops == pytest.approx(6.70, abs=0.01)
    assert run(batch=2048).achieved_tflops == pytest.approx(1979.0)


def test_bandwidth_and_peak_fields() -> None:
    r = run()
    assert r.peak_tflops == pytest.approx(1979.0)
    assert r.effective_bandwidth_tb_s == pytest.approx(3.35)
    assert r.fastest_bandwidth_tb_s == pytest.approx(3.35)
    assert r.batch1_bound_tokens_per_s == pytest.approx(3350 / W)


def test_capacity_and_kv_total() -> None:
    r = run(batch=4, context_tokens=8192)
    assert r.capacity_gb == pytest.approx(80.0)
    assert r.kv_total_gb == pytest.approx(4 * 1.342177, abs=1e-5)
    assert r.batch == 4


def test_effective_bandwidth_drops_when_weights_spill() -> None:
    r = analyze(
        load("chips", "sn40l"),
        load("models", "llama3_70b"),
        precision="bf16",
        sockets=1,
    )
    assert r.fastest_bandwidth_tb_s == pytest.approx(1.8)
    assert r.effective_bandwidth_tb_s < 1.8
    assert r.effective_bandwidth_tb_s == pytest.approx(0.33, abs=0.01)
