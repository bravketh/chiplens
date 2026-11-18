"""M1 demo: per-user speed and total throughput vs batch size.

Run from the repo root:
    pip install -e ".[plot]"
    python examples/m1_roofline_chart.py

Uses the placeholder numbers from the M1 reference case (70B params, 8-bit,
638 TFLOPS). They are assumptions, not a real chip.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chiplens.roofline import (  # noqa: E402
    balanced_batch,
    system_throughput,
    tokens_per_s_per_user,
)

PARAMS_B = 70.0
BYTES = 1.0
PEAK = 638.0
BATCHES = [2**i for i in range(0, 12)]  # 1 .. 2048
OUT = Path(__file__).resolve().parent.parent / "docs" / "img" / "m1_roofline.png"


def main() -> None:
    fig, (ax_user, ax_sys) = plt.subplots(1, 2, figsize=(11, 4))
    for bw, color in [(3.0, "tab:blue"), (6.0, "tab:orange")]:
        b_star = balanced_batch(bw, PEAK, PARAMS_B, BYTES)
        xs = sorted([*BATCHES, b_star])  # include the exact kink point
        user = [tokens_per_s_per_user(b, bw, PEAK, PARAMS_B, BYTES) for b in xs]
        system = [system_throughput(b, bw, PEAK, PARAMS_B, BYTES) for b in xs]
        ax_user.plot(xs, user, color=color, label=f"{bw:g} TB/s")
        ax_sys.plot(xs, system, color=color, label=f"{bw:g} TB/s")
        for ax in (ax_user, ax_sys):
            ax.axvline(b_star, color=color, linestyle=":", linewidth=1)

    ax_user.set(title="Speed each user sees", ylabel="tokens/s per user")
    ax_sys.set(title="Total throughput", ylabel="tokens/s, all users")
    for ax in (ax_user, ax_sys):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size (dotted line = balanced batch)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("70B model, 8-bit, 638 TFLOPS (assumed placeholder chip)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
