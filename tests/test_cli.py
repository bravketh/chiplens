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
