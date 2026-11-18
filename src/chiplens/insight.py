"""Plain-English verdict for a scenario: what is limiting this chip, and what helps.

Reads a ``scenario.Result`` and names one of three regimes:

- capacity: the weights plus every user's KV cache do not fit in memory.
- memory:   each step is limited by streaming bytes from memory.
- compute:  each step is limited by arithmetic.

No new physics here. It only describes numbers the models already computed, so
the wording can be tested.
"""

from __future__ import annotations

from dataclasses import dataclass

from chiplens.scenario import Result


@dataclass(frozen=True)
class Verdict:
    """The headline finding for a scenario.

    Attributes:
        regime: "capacity", "memory" or "compute".
        headline: One sentence for the top of the page.
        detail: Two or three sentences of numbers behind the headline.
        levers: Changes that would help, most useful first.
    """

    regime: str
    headline: str
    detail: str
    levers: list[str]


def _spill_note(r: Result) -> str:
    if len(r.placement) < 2:
        return ""
    slow = r.placement[1:]
    gb = sum(g for _, g in slow)
    names = ", ".join(name for name, _ in slow)
    return (
        f" {gb:.0f} GB of weights spill into {names}, so effective bandwidth is "
        f"{r.effective_bandwidth_tb_s:.2f} TB/s instead of "
        f"{r.fastest_bandwidth_tb_s:.2f} TB/s."
    )


def diagnose(r: Result) -> Verdict:
    """Describe the bottleneck of a scenario in plain English.

    Args:
        r: Output of ``scenario.analyze``.

    Returns:
        A Verdict with the regime, a headline, supporting detail and levers.
    """
    spill = _spill_note(r)

    if not r.kv_fits:
        need = r.weights_gb + r.kv_total_gb
        return Verdict(
            regime="capacity",
            headline=(
                f"Does not fit: {r.batch} users need {need:.0f} GB but the chip "
                f"has {r.capacity_gb:.0f} GB."
            ),
            detail=(
                f"Weights take {r.weights_gb:.1f} GB and each user's KV cache adds "
                f"{r.kv_gb_per_user:.2f} GB. The largest batch that fits is "
                f"{r.max_batch_by_capacity}."
            ),
            levers=[
                "More memory capacity",
                "Shorter context (smaller KV cache per user)",
                "Smaller weights (lower precision)",
                "More chips (they share the weights and the cache)",
            ],
        )

    if r.bound == "memory":
        idle = 1.0 - r.compute_utilization
        mem_ms = r.memory_step_s * 1000.0
        cmp_ms = r.compute_step_s * 1000.0
        if r.batch >= r.balanced_batch:
            why = (
                f"Even above the balanced batch of about {r.balanced_batch:.0f}, "
                f"the KV caches ({r.kv_total_gb:.1f} GB read every step) keep "
                "memory the limit."
            )
        else:
            why = (
                f"Batch {r.batch} is below the balanced batch of about "
                f"{r.balanced_batch:.0f}, so adding users raises total throughput "
                "almost for free."
            )
        levers = [
            "More memory bandwidth",
            "A bigger batch (more users share each weight read)",
            "Smaller weights (lower precision)",
        ]
        if len(r.placement) > 1:
            levers.insert(0, "Fit the weights in the fastest memory tier")
        return Verdict(
            regime="memory",
            headline=f"Memory-bound: the compute is idle {idle:.1%} of the time.",
            detail=(
                f"Each step takes {mem_ms:.1f} ms to read the bytes it needs, "
                f"but the arithmetic needs only {cmp_ms:.2f} ms. {why}{spill}"
            ),
            levers=levers,
        )

    return Verdict(
        regime="compute",
        headline="Compute-bound: the chip is doing all the arithmetic it can.",
        detail=(
            f"Batch {r.batch} is above the balanced batch of about "
            f"{r.balanced_batch:.0f}. Total throughput is capped at "
            f"{r.system_tokens_per_s:,.0f} tokens/s, so each extra user only "
            f"slows the others down (now {r.per_user_tokens_per_s:.1f} tokens/s "
            f"each).{spill}"
        ),
        levers=[
            "More peak compute",
            "Lower precision (more TFLOPS per chip)",
            "A smaller batch, if per-user speed matters more than total throughput",
        ],
    )
