"""Tests for the command line interface."""

import pytest

from chiplens.cli import main


def test_list_shows_chips_and_models(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "sn40l" in out and "h100_sxm" in out and "llama3_70b" in out


def test_run_prints_speed_and_flags_assumptions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "run",
            "--chip",
            "sn40l",
            "--model",
            "llama31_405b",
            "--precision",
            "bf16",
            "--sockets",
            "16",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "35.56" in out  # 28800 / 810
    assert "assumed" in out


def test_run_reports_errors_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "run",
            "--chip",
            "example_chip",
            "--model",
            "llama3_70b",
            "--precision",
            "bf16",
        ]
    )
    assert code == 2
    assert "peak_compute_bf16" in capsys.readouterr().err


def test_audit_user_flags_claim_above_bound(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "audit-user",
            "--claim",
            "457",
            "--bandwidth-tb-s",
            "28.8",
            "--weights-gb",
            "140",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "exceeds bound" in out
    assert "2.22" in out


def test_audit_total_within_bound(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "audit-total",
            "--claim",
            "1000",
            "--peak-tflops",
            "638",
            "--params-billion",
            "70",
        ]
    )
    assert code == 0
    assert "within bound" in capsys.readouterr().out


def test_validate_prints_the_honest_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate"]) == 0
    assert "does not reproduce" in capsys.readouterr().out.lower()


def test_economics_prints_cost_and_flags_assumptions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "economics",
            "--chip",
            "h100_sxm",
            "--model",
            "llama3_70b",
            "--precision",
            "fp8",
            "--batch",
            "2048",
            "--die-area",
            "800",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "cost per million tokens" in out.lower()
    assert "ASSUMED" in out
