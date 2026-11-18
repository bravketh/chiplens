"""Tests for model-size helpers: parameter count and KV cache.

Hand-worked for Llama 3 70B (80 layers, hidden 8192, 64 heads, 8 KV heads,
FFN 28672, vocab 128256):

    head_dim        = 8192 / 64                          = 128
    attention/layer = 2*8192^2 + 2*8192*(8*128)          = 150,994,944
    MLP/layer       = 3 * 8192 * 28672                   = 704,643,072
    per layer       = 855,638,016    x 80 layers         = 68,451,041,280
    embeddings      = 128256 * 8192 = 1,050,673,152 (x2 if input and output
                      matrices are separate)
    total           = 68,451,041,280 + 2,101,346,304     = 70,552,387,584

    KV bytes per token = 2 (K and V) * 80 * 8 * 128 * 2 bytes = 327,680
    KV per user at 8192 tokens = 327,680 * 8192 = 2.68435456 GB
"""

import pytest

from chiplens.llm import (
    head_dim,
    kv_bytes_per_token,
    kv_gb_per_user,
    max_batch_by_capacity,
    params_from_architecture,
)

ARCH = dict(
    layers=80, hidden_size=8192, heads=64, kv_heads=8, ffn_size=28672, vocab_size=128256
)


def test_head_dim() -> None:
    assert head_dim(8192, 64) == 128


def test_params_llama3_70b() -> None:
    assert params_from_architecture(**ARCH) == pytest.approx(70.552387584, abs=1e-6)


def test_params_tied_embeddings_removes_one_matrix() -> None:
    tied = params_from_architecture(**ARCH, tied_embeddings=True)
    assert tied == pytest.approx(70.552387584 - 1.050673152, abs=1e-6)


def test_kv_bytes_per_token() -> None:
    assert kv_bytes_per_token(80, 8, 128, 2.0) == 327_680


def test_kv_gb_per_user() -> None:
    assert kv_gb_per_user(80, 8, 128, 8192, 2.0) == pytest.approx(2.68435456)


def test_kv_scales_linearly_with_context() -> None:
    a = kv_gb_per_user(80, 8, 128, 4096, 2.0)
    b = kv_gb_per_user(80, 8, 128, 8192, 2.0)
    assert b == pytest.approx(2 * a)


def test_max_batch_by_capacity() -> None:
    # (192 - 140) / 2.68435456 = 19.37 -> 19 users fit
    assert max_batch_by_capacity(192.0, 140.0, 2.68435456) == 19


def test_max_batch_zero_when_weights_do_not_fit() -> None:
    assert max_batch_by_capacity(100.0, 140.0, 2.0) == 0


def test_bad_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        head_dim(8192, 0)
    with pytest.raises(ValueError):
        kv_gb_per_user(80, 8, 128, -1, 2.0)
