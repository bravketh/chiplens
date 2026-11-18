"""Memory tier model (M2).

Real chips have several memories: small and very fast (on-chip SRAM), medium
(HBM), big and slow (DDR). Weights are placed fastest tier first. One token
streams every weight once, so the time per token is the sum over tiers of
(bytes in tier) / (tier bandwidth):

    t = sum_i  GB_i / (1000 * BW_i)          [seconds]
    tokens/s = 1 / t

Units: capacities in GB, bandwidths in TB/s.

What this models: greedy fastest-first placement, sequential reads.
What it ignores: reads from different tiers overlapping in time (real chips can
overlap, so this is pessimistic when tiers work in parallel), caching of hot
weights, and prefetching from slow tiers into fast ones.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    """One memory level.

    Attributes:
        name: Label such as "hbm" or "ddr".
        capacity_gb: Capacity in GB.
        bandwidth_tb_s: Peak read bandwidth in TB/s.
    """

    name: str
    capacity_gb: float
    bandwidth_tb_s: float

    def __post_init__(self) -> None:
        if self.capacity_gb <= 0:
            raise ValueError(f"capacity_gb must be positive, got {self.capacity_gb}")
        if self.bandwidth_tb_s <= 0:
            raise ValueError(
                f"bandwidth_tb_s must be positive, got {self.bandwidth_tb_s}"
            )


def place_weights(weights_gb: float, tiers: list[Tier]) -> list[tuple[str, float]]:
    """Fill tiers fastest first.

    Args:
        weights_gb: Total weight size in GB.
        tiers: Available tiers, in any order.

    Returns:
        (tier name, GB placed) pairs, fastest tier first, skipping empty tiers.

    Raises:
        ValueError: If the weights do not fit in the combined capacity.
    """
    if weights_gb <= 0:
        raise ValueError(f"weights_gb must be positive, got {weights_gb}")
    remaining = weights_gb
    placement: list[tuple[str, float]] = []
    for tier in sorted(tiers, key=lambda t: t.bandwidth_tb_s, reverse=True):
        if remaining <= 0:
            break
        put = min(remaining, tier.capacity_gb)
        placement.append((tier.name, put))
        remaining -= put
    if remaining > 1e-9:
        total = sum(t.capacity_gb for t in tiers)
        raise ValueError(
            f"weights ({weights_gb:g} GB) do not fit in total capacity ({total:g} GB)"
        )
    return placement


def step_time_s(weights_gb: float, tiers: list[Tier]) -> float:
    """Seconds to stream all weights once (one decode step, ignoring compute)."""
    by_name = {t.name: t for t in tiers}
    return sum(
        gb / (1000.0 * by_name[name].bandwidth_tb_s)
        for name, gb in place_weights(weights_gb, tiers)
    )


def tokens_per_s(weights_gb: float, tiers: list[Tier]) -> float:
    """Memory-bound batch-1 speed with weights spread across tiers."""
    return 1.0 / step_time_s(weights_gb, tiers)


def effective_bandwidth_tb_s(weights_gb: float, tiers: list[Tier]) -> float:
    """Single bandwidth that gives the same step time: weights / time.

    Always between the slowest and fastest tier actually used.
    """
    return weights_gb / (1000.0 * step_time_s(weights_gb, tiers))
