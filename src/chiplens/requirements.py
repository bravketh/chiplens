"""Requirements check: does a chip meet a customer target, and what closes the gap?

A product manager's question is rarely "how fast is it?" but "the customer needs T
tokens/s per user at this load, so what do we need to change?" This module answers
it by asking, for each lever on its own (memory bandwidth, compute, memory
capacity), the smallest multiplier that makes the target pass.

A target passes when the speed per user is at least T and everything fits in
memory. Each lever's effect is monotonic (more never hurts), so the smallest
multiplier is found by bisection. A lever that cannot fix the gap, however large,
returns None: that is the useful finding ("more compute will not help here").

Scaled specs are hypotheticals, so results use ``scenario.with_multipliers``.
"""

from __future__ import annotations

from typing import Any

from chiplens.scenario import Result, analyze, with_multipliers

LEVERS = ("bandwidth", "compute", "capacity")


def meets_target(result: Result, target_tokens_per_s_per_user: float) -> bool:
    """True if speed per user reaches the target and the KV caches fit."""
    return (
        result.kv_fits and result.per_user_tokens_per_s >= target_tokens_per_s_per_user
    )


def required_multipliers(
    chip: dict[str, Any],
    model: dict[str, Any],
    precision: str,
    target_tokens_per_s_per_user: float,
    sockets: int = 1,
    batch: int = 1,
    context_tokens: int = 0,
    tokens_per_read: float = 1.0,
    efficiency: float = 1.0,
    max_multiplier: float = 16.0,
) -> dict[str, float | None]:
    """Smallest multiplier on each lever that meets a speed target.

    Args:
        chip: Loaded chip data.
        model: Loaded model data.
        precision: fp8, int8, bf16 or fp16.
        target_tokens_per_s_per_user: Required speed per user, in tokens/s.
        sockets, batch, context_tokens, tokens_per_read, efficiency: As in
            ``scenario.analyze``.
        max_multiplier: Largest factor tried for any lever.

    Returns:
        {"bandwidth": x, "compute": y, "capacity": z}. 1.0 for every lever if the
        target is already met. None for a lever that cannot meet the target on
        its own within ``max_multiplier``.
    """
    if target_tokens_per_s_per_user <= 0:
        raise ValueError("target must be positive")

    def passes(**mult: float) -> bool:
        try:
            result = analyze(
                with_multipliers(chip, **mult),
                model,
                precision=precision,
                sockets=sockets,
                batch=batch,
                context_tokens=context_tokens,
                tokens_per_read=tokens_per_read,
                efficiency=efficiency,
            )
        except ValueError:  # for example the weights do not fit
            return False
        return meets_target(result, target_tokens_per_s_per_user)

    if passes():
        return {lever: 1.0 for lever in LEVERS}

    out: dict[str, float | None] = {}
    for lever in LEVERS:
        if not passes(**{lever: max_multiplier}):
            out[lever] = None
            continue
        low, high = 1.0, max_multiplier
        for _ in range(60):
            mid = (low + high) / 2.0
            if passes(**{lever: mid}):
                high = mid
            else:
                low = mid
        out[lever] = high
    return out
