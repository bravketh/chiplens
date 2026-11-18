"""Tests for validation against published SN40L results.

Published (arXiv 2405.07518, Table IV, 16 sockets, BF16), tokens/s per user:
    Llama 3.1 8B: 1042    70B: 457    405B: 129

Model prediction at batch 1 (16 sockets = 28.8 TB/s aggregate HBM bandwidth):
    8B:   16 GB      -> 28800 / 16       = 1800.0   (1800-1042)/1042 = +72.7%
    70B:  141.105 GB -> 28800 / 141.1048 = 204.10   (204.10-457)/457  = -55.3%
    405B: 810 GB     -> 28800 / 810      = 35.556   (35.556-129)/129  = -72.4%
"""

from pathlib import Path

import pytest

from chiplens.validation import (
    VALIDATION_DIR,
    format_markdown,
    load_validation,
    run_validation,
)


@pytest.fixture(scope="module")
def rows():
    meta = load_validation(VALIDATION_DIR / "data" / "sn40l_table_iv.yaml")
    return {r.model_id: r for r in run_validation(meta)}


def test_validation_file_records_the_published_numbers(rows) -> None:
    assert rows["llama31_8b"].measured == 1042.0
    assert rows["llama3_70b"].measured == 457.0
    assert rows["llama31_405b"].measured == 129.0


def test_predictions_match_hand_calculation(rows) -> None:
    assert rows["llama31_8b"].predicted == pytest.approx(1800.0)
    assert rows["llama3_70b"].predicted == pytest.approx(
        28800 / 141.104775168, rel=1e-6
    )
    assert rows["llama31_405b"].predicted == pytest.approx(28800 / 810)


def test_errors_are_signed_percentages(rows) -> None:
    assert rows["llama31_8b"].error_pct == pytest.approx(72.74, abs=0.01)
    assert rows["llama3_70b"].error_pct == pytest.approx(-55.34, abs=0.01)
    assert rows["llama31_405b"].error_pct == pytest.approx(-72.44, abs=0.01)


def test_audit_flags_the_two_that_beat_the_bound(rows) -> None:
    assert rows["llama31_8b"].verdict == "within bound"
    assert rows["llama3_70b"].verdict == "exceeds bound"
    assert rows["llama31_405b"].verdict == "exceeds bound"


def test_markdown_table_states_honest_limits(rows) -> None:
    meta = load_validation(VALIDATION_DIR / "data" / "sn40l_table_iv.yaml")
    text = format_markdown(meta, list(rows.values()))
    assert "| Llama 3.1 8B |" in text or "Llama 3.1 8B" in text
    assert "does not reproduce" in text.lower()
    assert "speculative decoding" in text.lower()


def test_committed_results_file_is_current() -> None:
    """validation/results.md must be regenerated when data or models change."""
    meta = load_validation(VALIDATION_DIR / "data" / "sn40l_table_iv.yaml")
    expected = format_markdown(meta, run_validation(meta))
    committed = Path(VALIDATION_DIR / "results.md").read_text(encoding="utf-8")
    assert committed == expected
