"""Chart published vs predicted decode speed for the SN40L validation set.

Run from the repo root:
    pip install -e ".[plot]"
    python examples/validation_chart.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chiplens.validation import (  # noqa: E402
    VALIDATION_DIR,
    load_validation,
    run_validation,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "img" / "validation.png"


def main() -> None:
    meta = load_validation(VALIDATION_DIR / "data" / "sn40l_table_iv.yaml")
    rows = run_validation(meta)
    names = [r.model_name.replace("Llama ", "") for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([i - 0.2 for i in x], [r.measured for r in rows], 0.4, label="published")
    ax.bar(
        [i + 0.2 for i in x],
        [r.predicted for r in rows],
        0.4,
        label="chiplens bandwidth bound",
    )
    for i, r in enumerate(rows):
        ax.text(
            i, max(r.measured, r.predicted) * 1.15, f"{r.error_pct:+.0f}%", ha="center"
        )
    ax.set_yscale("log")
    ax.set_ylim(top=6000)
    ax.set_xticks(list(x), names)
    ax.set_ylabel("tokens/s per user (log scale)")
    ax.set_title("SN40L, 16 sockets, BF16: published vs predicted (error above bars)")
    ax.legend()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
