"""Pin each real chip's stored values to what its source page or PDF states.

If a number here changes, the source must have changed, and the file's `as_of` date
and note should change with it. Values read on 2026-11-18.

Unit conversions (stored value vs the source's wording):
    TPU v5p HBM: 95 GiB * 1.073741824 = 102.0 GB
    TPU v5e HBM: 16 GiB * 1.073741824 = 17.18 GB (stored 17.2)
    TPU bandwidth: 2765 GB/s = 2.765 TB/s;  819 GB/s = 0.819 TB/s
"""

import pytest

from chiplens.catalog import load, spec_confidence, spec_value
from chiplens.scenario import analyze

CASES = {
    "h200_sxm": {
        "memory_capacity": (141.0, "GB", "published"),
        "memory_bandwidth": (4.8, "TB/s", "published"),
        "board_power": (700.0, "W", "published"),
        "peak_compute_fp8": (1979.0, "TFLOPS", "estimated"),  # 3,958 sparse / 2
    },
    "tpu_v5p": {
        "peak_compute_bf16": (459.0, "TFLOPS", "published"),
        "peak_compute_fp8": (459.0, "TFLOPS", "published"),
        "memory_capacity": (102.0, "GB", "estimated"),  # 95 GiB
        "memory_bandwidth": (2.765, "TB/s", "published"),
        "board_power": (400.0, "W", "assumed"),
    },
    "tpu_v5e": {
        "peak_compute_bf16": (197.0, "TFLOPS", "published"),
        "peak_compute_int8": (393.0, "TFLOPS", "published"),
        "memory_capacity": (17.2, "GB", "estimated"),  # 16 GiB
        "memory_bandwidth": (0.819, "TB/s", "published"),
        "board_power": (200.0, "W", "assumed"),
    },
    "gaudi3": {
        "peak_compute_bf16": (1678.0, "TFLOPS", "published"),
        "peak_compute_fp8": (1678.0, "TFLOPS", "published"),
        "memory_capacity": (128.0, "GB", "published"),
        "memory_bandwidth": (3.7, "TB/s", "published"),
        "board_power": (900.0, "W", "published"),
    },
}


@pytest.mark.parametrize("chip_id", list(CASES))
def test_stored_values_match_the_sources(chip_id: str) -> None:
    chip = load("chips", chip_id)
    for key, (value, unit, confidence) in CASES[chip_id].items():
        assert spec_value(chip, key, unit) == pytest.approx(value, rel=1e-6), key
        assert spec_confidence(chip, key) == confidence, key


def test_every_real_chip_has_a_source_for_its_published_specs() -> None:
    for chip_id in CASES:
        chip = load("chips", chip_id)
        for key, field in chip["specs"].items():
            if field["confidence"] != "assumed":
                assert field["source_url"].startswith("http"), (chip_id, key)


def test_unpublished_power_is_never_labelled_published() -> None:
    for chip_id in ("tpu_v5p", "tpu_v5e"):
        assert spec_confidence(load("chips", chip_id), "board_power") == "assumed"


def test_ordering_at_batch_1_follows_memory_bandwidth() -> None:
    """Batch 1 is memory-bound, so speed should rank like bandwidth (TB/s)."""
    model = load("models", "llama3_70b")
    speed = {
        c: analyze(load("chips", c), model, precision="fp8").per_user_tokens_per_s
        for c in ("h100_sxm", "h200_sxm", "gaudi3", "tpu_v5p")
    }
    ranked = sorted(speed, key=speed.get, reverse=True)
    assert ranked == [
        "h200_sxm",
        "gaudi3",
        "h100_sxm",
        "tpu_v5p",
    ]  # 4.8 > 3.7 > 3.35 > 2.77


def test_a_small_chip_cannot_hold_a_70b_model_and_says_so() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        analyze(
            load("chips", "tpu_v5e"), load("models", "llama3_70b"), precision="int8"
        )


def test_small_chips_can_be_pooled() -> None:
    # 8 chips * 17.2 GB = 137.6 GB holds 70.55 GB of int8 weights
    r = analyze(
        load("chips", "tpu_v5e"),
        load("models", "llama3_70b"),
        precision="int8",
        sockets=8,
    )
    # 8 * 0.819 TB/s = 6.552 TB/s -> 6552 / 70.552 = 92.87 tokens/s
    assert r.per_user_tokens_per_s == pytest.approx(6552 / 70.552387584)
