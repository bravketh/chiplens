"""Tests for the plain-English verdict that opens the app.

The verdict names the bottleneck (memory, compute or capacity), says how idle the
compute is, and lists what would help. Numbers come from the H100 / Llama 3 70B
hand case in test_result_fields.py:

    batch 1:    utilization 0.003385 -> compute idle 99.66%, shown as 99.7%
    batch 2048: compute-bound
    batch 64 with 8K context: 70.55 + 64 * 1.342 = 156 GB > 80 GB -> capacity;
                largest batch that fits = floor((80 - 70.55) / 1.342) = 7
"""

from chiplens.catalog import load
from chiplens.insight import diagnose
from chiplens.scenario import analyze


def result(chip="h100_sxm", model="llama3_70b", precision="fp8", **kw):
    return analyze(load("chips", chip), load("models", model), precision, **kw)


def test_memory_bound_verdict() -> None:
    v = diagnose(result(batch=1))
    assert v.regime == "memory"
    assert "memory" in v.headline.lower()
    assert "99.7%" in v.headline
    assert any("bandwidth" in lever.lower() for lever in v.levers)
    assert any("batch" in lever.lower() for lever in v.levers)


def test_compute_bound_verdict() -> None:
    v = diagnose(result(batch=2048))
    assert v.regime == "compute"
    assert "compute" in v.headline.lower()
    assert any("compute" in lever.lower() for lever in v.levers)


def test_capacity_verdict_when_kv_does_not_fit() -> None:
    v = diagnose(result(batch=64, context_tokens=8192))
    assert v.regime == "capacity"
    assert "does not fit" in v.headline.lower()
    assert "7" in v.detail
    assert any(
        "capacity" in lever.lower() or "context" in lever.lower() for lever in v.levers
    )


def test_memory_detail_mentions_balanced_batch() -> None:
    v = diagnose(result(batch=1))
    assert "295" in v.detail  # balanced batch 1979 / 2 / 3.35 = 295.4


def test_spill_is_called_out() -> None:
    v = diagnose(result("sn40l", "llama3_70b", "bf16", sockets=1))
    assert "spill" in v.detail.lower()
    assert "ddr" in v.detail.lower()


def test_no_spill_note_when_everything_fits_fast_memory() -> None:
    v = diagnose(result(batch=1))
    assert "spill" not in v.detail.lower()
