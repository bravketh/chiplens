"""Operator-fusion model (M6).

An LLM layer is many small operations (norm, projections, attention, MLP). If
each runs as its own kernel, every intermediate result is written to memory and
read back, and every kernel launch costs a little fixed time. Fusing operations
keeps intermediates on chip and cuts launches.

Per decode step:
    activation traffic = 2 * (tensors per layer) * layers * (batch * hidden * b)
                         * (1 - fused_fraction)             [bytes; write + read]
    step time = max(compute time, (weights + activation traffic) / bandwidth)
                + launches * time per launch

Units: traffic in GB, bandwidth in TB/s, times in seconds, launch time in
microseconds.

What this models: extra memory traffic and launch overhead from unfused
intermediates, with every intermediate sized batch * hidden * bytes.
What it ignores: the wider MLP intermediates (about 3.5x hidden in Llama 3 70B,
so unfused traffic is understated), attention score matrices that grow with
context, overlap of launches with execution, and fusion that also changes the
amount of compute. The SN40L paper reports much larger fusion speedups on
prefill and training, where activations are big; this model is for decode.
"""

from __future__ import annotations


def activation_traffic_gb(
    batch: float,
    layers: int,
    hidden_size: int,
    bytes_per_act: float,
    intermediates_per_layer: int,
    fused_fraction: float = 0.0,
) -> float:
    """Extra GB moved per decode step by intermediates that go through memory.

    Args:
        batch: Users per step.
        layers: Transformer layers.
        hidden_size: Model width.
        bytes_per_act: Bytes per activation value.
        intermediates_per_layer: Tensors written and read back per layer if unfused.
        fused_fraction: 0 means nothing fused, 1 means all intermediates stay on chip.
    """
    if not 0.0 <= fused_fraction <= 1.0:
        raise ValueError(f"fused_fraction must be in [0, 1], got {fused_fraction}")
    for name, value in dict(
        batch=batch,
        layers=layers,
        hidden_size=hidden_size,
        bytes_per_act=bytes_per_act,
        intermediates_per_layer=intermediates_per_layer,
    ).items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    tensor_bytes = batch * hidden_size * bytes_per_act
    total = 2.0 * intermediates_per_layer * layers * tensor_bytes
    return total * (1.0 - fused_fraction) / 1e9


def kernel_launch_overhead_s(n_launches: int, us_per_launch: float) -> float:
    """Total fixed launch time in seconds."""
    if n_launches < 0 or us_per_launch < 0:
        raise ValueError("n_launches and us_per_launch must be >= 0")
    return n_launches * us_per_launch * 1e-6


def decode_step_time_s(
    weights_gb: float,
    activation_gb: float,
    bandwidth_tb_s: float,
    compute_s: float,
    launch_s: float,
) -> float:
    """Seconds for one decode step.

    t = max(compute, (weights + activations) / (1000 * BW)) + launch
    """
    if bandwidth_tb_s <= 0 or weights_gb <= 0:
        raise ValueError("bandwidth_tb_s and weights_gb must be positive")
    memory_s = (weights_gb + activation_gb) / (1000.0 * bandwidth_tb_s)
    return max(compute_s, memory_s) + launch_s


def speedup(unfused_time_s: float, fused_time_s: float) -> float:
    """How many times faster the fused version is."""
    if unfused_time_s <= 0 or fused_time_s <= 0:
        raise ValueError("times must be positive")
    return unfused_time_s / fused_time_s
