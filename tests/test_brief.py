"""Tests for the shareable one-page brief."""

from chiplens.brief import scenario_brief
from chiplens.catalog import load
from chiplens.insight import diagnose
from chiplens.scenario import analyze


def make(chip="sn40l", model="llama3_70b", precision="bf16", **kw):
    c, m = load("chips", chip), load("models", model)
    r = analyze(c, m, precision=precision, **kw)
    return c, m, r, diagnose(r)


def test_brief_has_headline_numbers_and_names() -> None:
    c, m, r, v = make(sockets=16)
    text = scenario_brief(c, m, "bf16", r, v, sockets=16)
    assert c["name"] in text and m["name"] in text
    assert v.headline in text
    assert f"{r.per_user_tokens_per_s:.1f}" in text
    assert "16" in text


def test_brief_lists_inputs_with_confidence_tags() -> None:
    c, m, r, v = make(sockets=16)
    text = scenario_brief(c, m, "bf16", r, v, sockets=16)
    assert "board_power" in text
    assert "assumed" in text
    assert "peak_compute_bf16" in text and "published" in text


def test_brief_states_it_is_an_upper_bound() -> None:
    c, m, r, v = make()
    text = scenario_brief(c, m, "bf16", r, v)
    assert "upper bound" in text.lower()


def test_brief_includes_upgrade_comparison_when_given() -> None:
    c, m, r, v = make()
    text = scenario_brief(
        c,
        m,
        "bf16",
        r,
        v,
        upgrades=[("2x memory bandwidth", 100.0, 100.0), ("2x compute", 0.0, 0.0)],
    )
    assert "2x memory bandwidth" in text
    assert "+100%" in text


def test_brief_has_no_em_dashes() -> None:
    c, m, r, v = make()
    assert "—" not in scenario_brief(c, m, "bf16", r, v)
