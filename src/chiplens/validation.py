"""Validation (M8): compare chiplens predictions with published measurements.

Reads a validation file from ``validation/data/`` (published results with
sources), runs the same scenarios through chiplens, and reports signed errors
honestly. Regenerate the committed report with ``chiplens validate --write``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chiplens.audit import audit_per_user_claim
from chiplens.catalog import load
from chiplens.scenario import analyze
from chiplens.schema import load_chip

VALIDATION_DIR = Path(__file__).resolve().parents[2] / "validation"


@dataclass(frozen=True)
class Row:
    """One published result next to the model's prediction."""

    model_id: str
    model_name: str
    measured: float  # tokens/s per user, published
    predicted: float  # tokens/s per user, chiplens upper bound
    error_pct: float  # (predicted - measured) / measured * 100
    ratio: float  # measured / predicted
    verdict: str  # audit verdict: within bound / exceeds bound
    note: str


def load_validation(path: str | Path) -> dict[str, Any]:
    """Load and schema-check a validation file."""
    return load_chip(path)


def run_validation(meta: dict[str, Any]) -> list[Row]:
    """Predict every published result in a validation file.

    Predictions use the file's chip, sockets, precision, batch and context.
    """
    chip = load("chips", meta["chip"])
    rows = []
    for model_id, field in meta["specs"].items():
        model = load("models", model_id)
        result = analyze(
            chip,
            model,
            precision=meta["precision"],
            sockets=int(meta["sockets"]),
            batch=int(meta["batch"]),
            context_tokens=int(meta["context_tokens"]),
        )
        measured = float(field["value"])
        predicted = result.per_user_tokens_per_s
        # Effective bandwidth that reproduces the predicted step time.
        eff_bw = result.weights_gb * predicted / 1000.0
        audit = audit_per_user_claim(measured, eff_bw, result.weights_gb)
        rows.append(
            Row(
                model_id=model_id,
                model_name=model["name"],
                measured=measured,
                predicted=predicted,
                error_pct=(predicted - measured) / measured * 100.0,
                ratio=measured / predicted,
                verdict=audit.verdict,
                note=field.get("note", ""),
            )
        )
    return rows


def format_markdown(meta: dict[str, Any], rows: list[Row]) -> str:
    """Render the validation report as Markdown."""
    src = next(iter(meta["specs"].values()))["source_url"]
    lines = [
        "# Validation: SN40L decode speed",
        "",
        f"Source of the measurements: {src} (Table IV). Published values are "
        "tokens per second per user on "
        f"{meta['sockets']} SN40L sockets with {meta['precision'].upper()} weights.",
        "",
        "chiplens prediction: the batch-1 memory-bandwidth bound with ideal scaling "
        f"across {meta['sockets']} sockets (HBM bandwidth and capacity multiplied by "
        "the socket count, no network cost, no KV cache). It is an upper bound, "
        "not a forecast.",
        "",
        "| Model | Published (tok/s/user) | chiplens bound | Error "
        "| Published / bound | Audit |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.model_name} | {r.measured:,.0f} | {r.predicted:,.1f} | "
            f"{r.error_pct:+.1f}% | {r.ratio:.2f}x | {r.verdict} |"
        )
    lines += [
        "",
        "Error is (chiplens minus published) divided by published.",
        "",
        "## What this shows",
        "",
        "**chiplens does not reproduce these published numbers.** The errors above "
        "are large and go in both directions.",
        "",
    ]
    for r in rows:
        if r.verdict == "exceeds bound":
            lines.append(
                f"- **{r.model_name}** beats the bandwidth bound by {r.ratio:.2f}x. "
                "No decode that reads every weight once per token can do that, so the "
                "published run must produce more than one token per weight read. The "
                "paper states that these runs use speculative decoding (Section VI-B), "
                "which fits. This model does not include speculative decoding."
            )
        else:
            lines.append(
                f"- **{r.model_name}** reaches {r.ratio:.0%} of the bound. The gap is "
                "real-system inefficiency this model ignores. Likely contributors "
                "(untested hypotheses): communication across sockets, kernel and "
                "launch overheads, and bandwidth below peak. A small model spread "
                "over many sockets amplifies all of these."
            )
    lines += [
        "",
        "## Limits of this comparison",
        "",
        "- The batch size of the published runs is not stated. chiplens assumes "
        "batch 1.",
        "- Table values were read from an HTML rendering of the paper that repeated "
        "digits. Verify them against the PDF.",
        "- HBM bandwidth is 1.8 TB/s in Table II and about 2 TB/s in Section IV. "
        "chiplens uses 1.8.",
        "- The 70B row uses the Llama 3 70B model file (same dimensions as 3.1 70B) "
        "with a parameter count derived from its architecture.",
        "- Power is not part of this comparison.",
        "",
    ]
    return "\n".join(lines)
