"""Model-size helpers: parameter count and KV cache.

Models what a transformer must store: its weights, and per user a key/value
(KV) cache that grows with context length.

Ignores: normalization weights (about 0.002% of a 70B model), biases,
mixture-of-experts layouts, and cache compression tricks.
"""

from __future__ import annotations

import math


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def head_dim(hidden_size: int, heads: int) -> int:
    """Width of one attention head.

    head_dim = hidden_size / heads
    """
    _require_positive(hidden_size=hidden_size, heads=heads)
    return hidden_size // heads


def params_from_architecture(
    layers: int,
    hidden_size: int,
    heads: int,
    kv_heads: int,
    ffn_size: int,
    vocab_size: int,
    tied_embeddings: bool = False,
) -> float:
    """Parameter count of a Llama-style transformer (gated MLP, GQA attention).

    Per layer:
        attention = 2 * d^2 + 2 * d * (kv_heads * head_dim)   (Q, O and K, V)
        MLP       = 3 * d * ffn_size                           (gate, up, down)
    Plus embeddings: vocab * d, counted twice unless input and output are tied.

    Args:
        layers: Number of transformer layers.
        hidden_size: Model width d.
        heads: Attention heads.
        kv_heads: Key/value heads (fewer than heads means grouped-query attention).
        ffn_size: MLP intermediate width.
        vocab_size: Vocabulary size.
        tied_embeddings: True if input and output embeddings share one matrix.

    Returns:
        Parameters, in billions.
    """
    _require_positive(
        layers=layers,
        hidden_size=hidden_size,
        heads=heads,
        kv_heads=kv_heads,
        ffn_size=ffn_size,
        vocab_size=vocab_size,
    )
    kv_dim = kv_heads * head_dim(hidden_size, heads)
    attention = 2 * hidden_size**2 + 2 * hidden_size * kv_dim
    mlp = 3 * hidden_size * ffn_size
    embeddings = vocab_size * hidden_size * (1 if tied_embeddings else 2)
    return (layers * (attention + mlp) + embeddings) / 1e9


def kv_bytes_per_token(
    layers: int, kv_heads: int, head_dim_: int, bytes_per_value: float
) -> float:
    """KV cache bytes stored per token of context, for one user.

    bytes = 2 * layers * kv_heads * head_dim * b      (2 = one K and one V)
    """
    _require_positive(
        layers=layers,
        kv_heads=kv_heads,
        head_dim_=head_dim_,
        bytes_per_value=bytes_per_value,
    )
    return 2 * layers * kv_heads * head_dim_ * bytes_per_value


def kv_gb_per_user(
    layers: int,
    kv_heads: int,
    head_dim_: int,
    context_tokens: int,
    bytes_per_value: float,
) -> float:
    """KV cache size for one user at a given context length, in GB."""
    if context_tokens < 0:
        raise ValueError(f"context_tokens must be >= 0, got {context_tokens}")
    return (
        kv_bytes_per_token(layers, kv_heads, head_dim_, bytes_per_value)
        * context_tokens
        / 1e9
    )


def max_batch_by_capacity(
    capacity_gb: float, weights_gb_: float, kv_gb_per_user_: float
) -> int:
    """Most users whose KV caches fit beside the weights.

    B_max = floor((capacity - weights) / kv_per_user), never below 0.
    """
    _require_positive(capacity_gb=capacity_gb, kv_gb_per_user_=kv_gb_per_user_)
    free = capacity_gb - weights_gb_
    if free <= 0:
        return 0
    return math.floor(free / kv_gb_per_user_)
