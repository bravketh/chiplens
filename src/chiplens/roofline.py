"""Roofline and batching model for LLM decoding (M1).

Idea: generating one token means streaming every weight from memory once and
doing 2 operations per weight (one multiply, one add). With a batch of B users,
the same weights are read once but used B times. So a chip is limited by
whichever is slower:

    memory:   how fast it can stream the weights   (bytes per second)
    compute:  how fast it can do the arithmetic    (operations per second)

Units used throughout: parameters in billions, weights in GB, bandwidth in
TB/s, compute in TFLOPS (1e12 ops/s), speeds in tokens/s.

What this models: weight streaming and arithmetic, at peak rates.
What it ignores: KV cache and attention, prefill, multi-chip scaling, kernel
overheads, and real utilization below peak. See docs/methodology.md.
"""

from __future__ import annotations


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def weights_gb(params_billion: float, bytes_per_param: float) -> float:
    """Size of the model weights.

    W = P * b

    Args:
        params_billion: Parameter count, in billions.
        bytes_per_param: Bytes per weight (1 for 8-bit, 2 for 16-bit).

    Returns:
        Weight size in GB.
    """
    _require_positive(params_billion=params_billion, bytes_per_param=bytes_per_param)
    return params_billion * bytes_per_param


def memory_limit_tokens_per_s(bandwidth_tb_s: float, weights_gb_: float) -> float:
    """Tokens/s if memory bandwidth were the only limit (batch 1 speed).

    T_mem = 1000 * BW / W

    Args:
        bandwidth_tb_s: Memory bandwidth in TB/s.
        weights_gb_: Weight size in GB.

    Returns:
        Tokens per second for one user.
    """
    _require_positive(bandwidth_tb_s=bandwidth_tb_s, weights_gb_=weights_gb_)
    return 1000.0 * bandwidth_tb_s / weights_gb_


def compute_limit_tokens_per_s(peak_tflops: float, params_billion: float) -> float:
    """Tokens/s if peak compute were the only limit, summed over all users.

    T_cmp = 1000 * F / (2 * P)

    Args:
        peak_tflops: Peak compute in TFLOPS (1e12 ops/s) at the model's precision.
        params_billion: Parameter count, in billions.

    Returns:
        Total tokens per second across the whole batch.
    """
    _require_positive(peak_tflops=peak_tflops, params_billion=params_billion)
    return 1000.0 * peak_tflops / (2.0 * params_billion)


def balanced_batch(
    bandwidth_tb_s: float,
    peak_tflops: float,
    params_billion: float,
    bytes_per_param: float,
) -> float:
    """Batch size where memory and compute limits meet.

    B* = T_cmp / T_mem

    Below B* the chip waits on memory. Above it, compute is the bottleneck.

    Returns:
        Balanced batch size (dimensionless, not rounded).
    """
    w = weights_gb(params_billion, bytes_per_param)
    return compute_limit_tokens_per_s(
        peak_tflops, params_billion
    ) / memory_limit_tokens_per_s(bandwidth_tb_s, w)


def tokens_per_s_per_user(
    batch: float,
    bandwidth_tb_s: float,
    peak_tflops: float,
    params_billion: float,
    bytes_per_param: float,
) -> float:
    """Speed each user sees at a given batch size.

    t(B) = min(T_mem, T_cmp / B)

    Returns:
        Tokens per second for one user.
    """
    _require_positive(batch=batch)
    w = weights_gb(params_billion, bytes_per_param)
    return min(
        memory_limit_tokens_per_s(bandwidth_tb_s, w),
        compute_limit_tokens_per_s(peak_tflops, params_billion) / batch,
    )


def system_throughput(
    batch: float,
    bandwidth_tb_s: float,
    peak_tflops: float,
    params_billion: float,
    bytes_per_param: float,
) -> float:
    """Total tokens/s across all users at a given batch size.

    throughput = B * t(B)

    Returns:
        Tokens per second summed over the batch.
    """
    return batch * tokens_per_s_per_user(
        batch, bandwidth_tb_s, peak_tflops, params_billion, bytes_per_param
    )


def compute_utilization(
    batch: float,
    bandwidth_tb_s: float,
    peak_tflops: float,
    params_billion: float,
    bytes_per_param: float,
) -> float:
    """Fraction of peak compute doing useful work.

    util = min(1, B / B*)

    Returns:
        A number from 0 to 1.
    """
    _require_positive(batch=batch)
    b_star = balanced_batch(
        bandwidth_tb_s, peak_tflops, params_billion, bytes_per_param
    )
    return min(1.0, batch / b_star)


def arithmetic_intensity(batch: float, bytes_per_param: float) -> float:
    """Operations done per byte of weights read from memory.

    I = 2 * B / b

    Args:
        batch: Number of users sharing each weight read.
        bytes_per_param: Bytes per weight.

    Returns:
        Intensity in ops/byte.
    """
    _require_positive(batch=batch, bytes_per_param=bytes_per_param)
    return 2.0 * batch / bytes_per_param


def ridge_point(bandwidth_tb_s: float, peak_tflops: float) -> float:
    """Arithmetic intensity at which a chip stops being memory-bound.

    ridge = F / BW   (TFLOPS divided by TB/s gives ops/byte)

    Returns:
        Intensity in ops/byte.
    """
    _require_positive(bandwidth_tb_s=bandwidth_tb_s, peak_tflops=peak_tflops)
    return peak_tflops / bandwidth_tb_s
