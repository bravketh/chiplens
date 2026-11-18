"""Smoke test: the Streamlit app runs without errors and shows a result."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def test_app_runs_with_defaults() -> None:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert at.metric[0].label == "Speed per user"
    assert len(at.metric) >= 4  # 4 headline metrics plus the economics expander


def test_app_warns_when_inputs_are_assumed() -> None:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.selectbox(key="chip").set_value("sn40l")
    at.run()
    assert not at.exception
    assert any("ASSUMED" in w.value for w in at.warning)


def test_app_shows_a_verdict_and_all_tabs() -> None:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert at.session_state["view"] == "Bottleneck"
    assert any("Memory-bound" in m.value for m in at.markdown)


def test_story_preset_changes_batch_and_context() -> None:
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.selectbox(key="story").set_value("Busy API server")
    at.run()
    assert not at.exception
    assert at.session_state["batch"] == 64
    assert at.session_state["context"] == 4096


def test_sambanova_claim_button_loads_the_demo_and_flags_it() -> None:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["view"] = "Check a claim"
    at.run()
    next(b for b in at.button if "SambaNova" in b.label).click()
    at.run()
    assert not at.exception
    assert at.session_state["chip"] == "sn40l"
    assert at.session_state["view"] == "Check a claim"
    assert any("Exceeds the bound" in e.value for e in at.error)


def test_every_chip_and_model_combination_renders() -> None:
    from chiplens.catalog import list_ids

    for chip in list_ids("chips"):
        for model in list_ids("models"):
            at = AppTest.from_file(str(APP), default_timeout=30)
            at.session_state["chip"] = chip
            at.session_state["model"] = model
            at.run()
            assert not at.exception, (chip, model)


def test_every_view_renders() -> None:
    nav = [
        "Bottleneck",
        "Requirements",
        "Batching",
        "Upgrades",
        "Compare chips",
        "Check a claim",
        "Cost",
        "Validation",
        "Learn",
    ]
    for view in nav:
        at = AppTest.from_file(str(APP), default_timeout=30)
        at.session_state["view"] = view
        at.run()
        assert not at.exception, view


def test_requirements_view_reports_a_miss_and_the_lever() -> None:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["view"] = "Requirements"
    at.session_state["target"] = 95.0
    at.run()
    assert not at.exception
    assert any("Misses it" in e.value for e in at.error)
    shown = {m.label: m.value for m in at.metric}
    assert shown["Memory bandwidth"].endswith("x")  # about 2.00x
    assert shown["Compute"] == "Will not help"


def test_requirements_view_reports_a_pass() -> None:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["view"] = "Requirements"
    at.session_state["target"] = 10.0
    at.run()
    assert not at.exception
    assert any("Meets it" in s.value for s in at.success)


def test_applying_the_implied_tokens_per_read_updates_the_scenario() -> None:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["view"] = "Check a claim"
    at.run()
    next(b for b in at.button if "SambaNova" in b.label).click()
    at.run()
    next(b for b in at.button if "tokens per read" in b.label).click()
    at.run()
    assert not at.exception
    assert 2.0 < at.session_state["spec_k"] < 2.5


def test_calibrating_from_a_within_bound_claim_sets_efficiency() -> None:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["view"] = "Check a claim"
    at.run()
    at.session_state["lib"] = "llama31_8b"
    at.run()
    next(b for b in at.button if "SambaNova" in b.label).click()
    at.run()
    next(b for b in at.button if "Calibrate" in b.label).click()
    at.run()
    assert not at.exception
    # 1042 / 1800 = 0.58
    assert 0.55 < at.session_state["eff"] < 0.62
