"""Tests for the economics report that wires scenario results to the cost models."""

import pytest

from chiplens.catalog import load
from chiplens.report import economics_report
from chiplens.scenario import analyze


def make(**kw):
    chip = load("chips", "h100_sxm")
    model = load("models", "llama3_70b")
    result = analyze(chip, model, precision="fp8", batch=2048)
    assumptions = load("assumptions", "cost_example")
    return economics_report(result, assumptions, die_area_mm2=800.0, **kw)


def test_cogs_and_price_follow_the_pure_functions() -> None:
    r = make()
    assert r["cogs_per_chip"] == pytest.approx(r["silicon_cost"] + 4000.0)
    assert r["price_per_chip"] == pytest.approx(2.0 * r["cogs_per_chip"])


def test_cost_per_million_tokens_is_positive_and_ordered() -> None:
    r = make()
    assert r["cost_per_million_tokens"] > 0
    p = r["cost_per_million_tokens_p10_p50_p90"]
    assert p[0] <= p[1] <= p[2]


def test_chiplets_lower_silicon_cost_at_800_mm2() -> None:
    mono = make(n_dies=1)
    split = make(n_dies=2, d2d_overhead=0.1)
    assert split["silicon_cost"] < mono["silicon_cost"]


def test_report_is_reproducible_with_a_seed() -> None:
    assert make(seed=3) == make(seed=3)


def test_sockets_scale_capex() -> None:
    one = make(sockets=1)
    eight = make(sockets=8)
    assert eight["capex_total"] == pytest.approx(8 * one["capex_total"])


def test_report_marks_all_inputs_assumed() -> None:
    r = make()
    assert r["worst_confidence"] == "assumed"
    assert "wafer_cost" in r["assumed_inputs"]
