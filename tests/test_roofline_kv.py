"""Tests for decode speed with a KV cache.

Each decode step reads the weights once plus every user's KV cache:
    bytes per step = W + B * kv
Hand case: 70 GB weights, 3 TB/s, kv = 2 GB per user, batch 10:
    memory-bound speed = 3000 / (70 + 10 * 2) = 33.33 tokens/s per user
    compute-bound speed = 4557.14 / 10 = 455.7  (not the limit)
"""

import pytest

from chiplens.roofline import tokens_per_s_per_user, tokens_per_s_per_user_with_kv


def test_kv_slows_memory_bound_users() -> None:
    got = tokens_per_s_per_user_with_kv(10, 3.0, 638.0, 70.0, 1.0, 2.0)
    assert got == pytest.approx(3000 / 90)


def test_zero_kv_matches_plain_model() -> None:
    for batch in (1, 50, 512):
        assert tokens_per_s_per_user_with_kv(
            batch, 3.0, 638.0, 70.0, 1.0, 0.0
        ) == pytest.approx(tokens_per_s_per_user(batch, 3.0, 638.0, 70.0, 1.0))


def test_kv_does_not_change_compute_bound_regime_much() -> None:
    # batch 512 with tiny kv is still compute bound: 4557.14 / 512 = 8.90
    got = tokens_per_s_per_user_with_kv(512, 3.0, 638.0, 70.0, 1.0, 0.001)
    assert got == pytest.approx(8.90, abs=0.01)


def test_big_kv_pushes_into_memory_bound_at_large_batch() -> None:
    # batch 512, kv 2 GB: memory = 3000 / (70 + 1024) = 2.742 < 8.90
    got = tokens_per_s_per_user_with_kv(512, 3.0, 638.0, 70.0, 1.0, 2.0)
    assert got == pytest.approx(3000 / 1094)


def test_negative_kv_rejected() -> None:
    with pytest.raises(ValueError):
        tokens_per_s_per_user_with_kv(1, 3.0, 638.0, 70.0, 1.0, -1.0)
