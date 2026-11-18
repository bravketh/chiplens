"""Command line interface: list, run, economics, audit-user, audit-total, validate."""

from __future__ import annotations

import argparse
import sys

from chiplens.audit import audit_per_user_claim, audit_total_claim
from chiplens.catalog import list_ids, load
from chiplens.report import economics_report
from chiplens.scenario import analyze
from chiplens.validation import (
    VALIDATION_DIR,
    format_markdown,
    load_validation,
    run_validation,
)


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

    eco = sub.add_parser("economics", help="cost per token, payback and uncertainty")
    eco.add_argument("--chip", required=True)
    eco.add_argument("--model", required=True)
    eco.add_argument(
        "--precision", required=True, choices=["fp8", "int8", "bf16", "fp16"]
    )
    eco.add_argument("--sockets", type=int, default=1)
    eco.add_argument("--batch", type=int, default=1)
    eco.add_argument("--context", type=int, default=0)
    eco.add_argument("--die-area", type=float, required=True, help="mm^2 per chip")
    eco.add_argument("--dies", type=int, default=1, help="chiplets per chip")
    eco.add_argument("--d2d-overhead", type=float, default=0.0)
    eco.add_argument("--assumptions", default="cost_example")

    au = sub.add_parser("audit-user", help="check a per-user tokens/s claim")
    au.add_argument("--claim", type=float, required=True, help="tokens/s per user")
    au.add_argument("--bandwidth-tb-s", type=float, required=True)
    au.add_argument("--weights-gb", type=float, required=True)

    at = sub.add_parser("audit-total", help="check a total tokens/s claim")
    at.add_argument("--claim", type=float, required=True, help="total tokens/s")
    at.add_argument("--peak-tflops", type=float, required=True)
    at.add_argument("--params-billion", type=float, required=True)

    val = sub.add_parser("validate", help="compare predictions with published results")
    val.add_argument(
        "--write", action="store_true", help="rewrite validation/results.md"
    )
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


def _cmd_economics(args: argparse.Namespace) -> int:
    chip = load("chips", args.chip)
    model = load("models", args.model)
    result = analyze(
        chip,
        model,
        precision=args.precision,
        sockets=args.sockets,
        batch=args.batch,
        context_tokens=args.context,
    )
    r = economics_report(
        result,
        load("assumptions", args.assumptions),
        die_area_mm2=args.die_area,
        n_dies=args.dies,
        d2d_overhead=args.d2d_overhead,
        sockets=args.sockets,
    )
    lo, mid, hi = r["cost_per_million_tokens_p10_p50_p90"]
    pl, pm, ph = r["payback_months_p10_p50_p90"]
    print(f"{chip['name']} x{args.sockets} running {model['name']} ({args.precision})")
    print(f"  system throughput:        {result.system_tokens_per_s:,.0f} tokens/s")
    print(f"  silicon cost per chip:    ${r['silicon_cost']:,.0f}")
    print(f"  COGS per chip:            ${r['cogs_per_chip']:,.0f}")
    print(f"  price per chip:           ${r['price_per_chip']:,.0f}")
    print(f"  hourly cost (TCO):        ${r['hourly_tco']:.2f}")
    print(f"  cost per million tokens:  ${r['cost_per_million_tokens']:.4f}")
    print(f"    p10 / p50 / p90:        ${lo:.4f} / ${mid:.4f} / ${hi:.4f}")
    print(f"  payback:                  {r['payback_months']:.1f} months")
    print(f"    p10 / p50 / p90:        {pl:.1f} / {pm:.1f} / {ph:.1f} months")
    assumed = r["assumed_inputs"]
    print(f"  ASSUMED cost inputs ({len(assumed)}): {', '.join(assumed)}")
    if result.assumed_inputs:
        print(f"  ASSUMED chip inputs:      {', '.join(result.assumed_inputs)}")
    print("  note: every cost input is an assumption, and throughput is an upper")
    print("  bound, so cost per token is a floor and payback is optimistic. This")
    print("  shows how the model behaves, not what anyone pays (docs/methodology.md).")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    if args.command == "audit-user":
        r = audit_per_user_claim(args.claim, args.bandwidth_tb_s, args.weights_gb)
    else:
        r = audit_total_claim(args.claim, args.peak_tflops, args.params_billion)
    print(f"claim:   {r.claimed:,.1f} tokens/s")
    print(f"bound:   {r.bound:,.1f} tokens/s")
    print(f"ratio:   {r.ratio:.2f}x")
    print(f"verdict: {r.verdict}")
    print(r.explanation)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    meta = load_validation(VALIDATION_DIR / "data" / "sn40l_table_iv.yaml")
    text = format_markdown(meta, run_validation(meta))
    if args.write:
        (VALIDATION_DIR / "results.md").write_text(text, encoding="utf-8")
        print(f"wrote {VALIDATION_DIR / 'results.md'}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return _cmd_list()
        if args.command == "economics":
            return _cmd_economics(args)
        if args.command in ("audit-user", "audit-total"):
            return _cmd_audit(args)
        if args.command == "validate":
            return _cmd_validate(args)
        return _cmd_run(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
