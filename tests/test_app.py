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
    chip_box = at.selectbox[0]
    chip_box.set_value("sn40l")
    at.run()
    assert not at.exception
    assert any("ASSUMED" in w.value for w in at.warning)
