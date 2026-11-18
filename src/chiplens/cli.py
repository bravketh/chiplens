"""Command line interface: ``chiplens list`` and ``chiplens run``."""

from __future__ import annotations

import argparse
import sys

from chiplens.catalog import list_ids, load
from chiplens.scenario import analyze


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chiplens", description="AI inference chip tradeoff model."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show available chips and models")

    run = sub.add_parser("run", help="run one chip and model scenario")
    run.add_argument("--chip", required=True)
    run.add_argument("--model", required=True)
    run.add_argument(
        "--precision", required=True, choices=["fp8", "int8", "bf16", "fp16"]
    )
    run.add_argument("--sockets", type=int, default=1)
    run.add_argument("--batch", type=int, default=1)
    run.add_argument("--context", type=int, default=0, help="tokens per user")
    return parser


def _cmd_list() -> int:
    print("chips:  ", ", ".join(list_ids("chips")))
    print("models: ", ", ".join(list_ids("models")))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    chip = load("chips", args.chip)
    model = load("models", args.model)
    r = analyze(
        chip,
        model,
        precision=args.precision,
        sockets=args.sockets,
        batch=args.batch,
        context_tokens=args.context,
    )
    placement = ", ".join(f"{gb:.1f} GB in {tier}" for tier, gb in r.placement)
    print(f"{chip['name']} x{args.sockets} running {model['name']} ({args.precision})")
    print(f"  weights:               {r.weights_gb:.1f} GB ({placement})")
    print(f"  per-user speed:        {r.per_user_tokens_per_s:.2f} tokens/s")
    print(f"  system throughput:     {r.system_tokens_per_s:.1f} tokens/s")
    print(f"  balanced batch:        {r.balanced_batch:.1f} (without KV cache)")
    print(f"  compute utilization:   {r.compute_utilization:.1%}")
    if r.kv_gb_per_user > 0:
        print(f"  KV cache per user:     {r.kv_gb_per_user:.2f} GB (fits: {r.kv_fits})")
    print(f"  power:                 {r.power_w:.0f} W")
    print(f"  tokens per joule:      {r.tokens_per_joule:.3f}")
    print(f"  input confidence:      worst = {r.worst_confidence}")
    if r.assumed_inputs:
        print(f"  ASSUMED inputs:        {', '.join(r.assumed_inputs)}")
    if r.estimated_inputs:
        print(f"  estimated inputs:      {', '.join(r.estimated_inputs)}")
    print("  note: an optimistic upper bound, not a measurement (docs/methodology.md)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return _cmd_list()
        return _cmd_run(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
