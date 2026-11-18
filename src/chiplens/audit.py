"""Claim auditor (M8): is a published speed claim physically plausible?

Two hard bounds from the roofline model:

1. Per-user decode speed cannot exceed memory bandwidth / weight size, because
   every token needs every weight read once:   bound = 1000 * BW / W.
2. Total throughput cannot exceed peak compute / (2 * params):
                                                 bound = 1000 * F / (2 * P).

A claim above a bound is not necessarily false. It means something outside the
simple model is at work (speculative decoding, smaller effective weights, a
different batch or precision, or an error), and the auditor says what to ask.

Units: BW in TB/s, W in GB, F in TFLOPS, P in billions, speeds in tokens/s.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditResult:
    """Outcome of auditing one claim."""

    claimed: float
    bound: float
    ratio: float  # claimed / bound
    verdict: str  # "within bound" or "exceeds bound"
    explanation: str


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def audit_per_user_claim(
    claimed_tokens_per_s_per_user: float, bandwidth_tb_s: float, weights_gb: float
) -> AuditResult:
    """Check a per-user decode speed against the memory-bandwidth bound."""
    _require_positive(
        claimed=claimed_tokens_per_s_per_user,
        bandwidth_tb_s=bandwidth_tb_s,
        weights_gb=weights_gb,
    )
    bound = 1000.0 * bandwidth_tb_s / weights_gb
    ratio = claimed_tokens_per_s_per_user / bound
    if ratio <= 1.0:
        return AuditResult(
            claimed_tokens_per_s_per_user,
            bound,
            ratio,
            "within bound",
            f"Uses {ratio:.0%} of the bandwidth bound, so the claim is "
            "physically possible.",
        )
    return AuditResult(
        claimed_tokens_per_s_per_user,
        bound,
        ratio,
        "exceeds bound",
        f"The claim is {ratio:.2f}x the bandwidth bound, so each weight read must "
        f"yield about {ratio:.2f} tokens. Ask: is speculative decoding used? Are the "
        "weights smaller than assumed (quantization, sparsity, mixture of experts)? "
        "Is the bandwidth or model size wrong? Otherwise the claim is not credible.",
    )


def audit_total_claim(
    claimed_total_tokens_per_s: float, peak_tflops: float, params_billion: float
) -> AuditResult:
    """Check a total-throughput claim against the compute bound."""
    _require_positive(
        claimed=claimed_total_tokens_per_s,
        peak_tflops=peak_tflops,
        params_billion=params_billion,
    )
    bound = 1000.0 * peak_tflops / (2.0 * params_billion)
    ratio = claimed_total_tokens_per_s / bound
    if ratio <= 1.0:
        return AuditResult(
            claimed_total_tokens_per_s,
            bound,
            ratio,
            "within bound",
            f"Uses {ratio:.0%} of peak compute, so the claim is physically possible.",
        )
    return AuditResult(
        claimed_total_tokens_per_s,
        bound,
        ratio,
        "exceeds bound",
        f"The claim is {ratio:.2f}x the compute bound. Ask: more chips than stated? "
        "Lower precision or sparsity doubling peak? Fewer active parameters "
        "(mixture of experts)? Speculative decoding counting accepted tokens?",
    )
