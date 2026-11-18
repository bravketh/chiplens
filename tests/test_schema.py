"""Tests for the chip data schema.

Rule being tested: every spec field carries value, unit, source_url, as_of and
confidence (published | estimated | assumed). Nothing may look sourced when it
is not.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from chiplens.schema import load_chip, validate_chip

CHIPS_DIR = Path(__file__).resolve().parent.parent / "data" / "chips"
EXAMPLE = CHIPS_DIR / "example_chip.yaml"


def good_field() -> dict:
    return {
        "value": 3.0,
        "unit": "TB/s",
        "source_url": "https://example.com/spec",
        "as_of": "2024-05-12",
        "confidence": "published",
    }


def good_chip() -> dict:
    return {"name": "Test Chip", "specs": {"memory_bandwidth": good_field()}}


def test_example_file_loads_and_is_valid() -> None:
    chip = load_chip(EXAMPLE)
    assert chip["name"] == "Example Chip (placeholder)"


def test_example_file_is_entirely_assumed() -> None:
    """The placeholder chip must never look like real data."""
    chip = load_chip(EXAMPLE)
    for field in chip["specs"].values():
        assert field["confidence"] == "assumed"


def test_valid_chip_has_no_errors() -> None:
    assert validate_chip(good_chip()) == []


@pytest.mark.parametrize(
    "missing", ["value", "unit", "source_url", "as_of", "confidence"]
)
def test_missing_key_is_reported(missing: str) -> None:
    chip = good_chip()
    del chip["specs"]["memory_bandwidth"][missing]
    errors = validate_chip(chip)
    assert any(missing in e and "memory_bandwidth" in e for e in errors)


def test_bad_confidence_tag_is_reported() -> None:
    chip = good_chip()
    chip["specs"]["memory_bandwidth"]["confidence"] = "pretty sure"
    errors = validate_chip(chip)
    assert any("confidence" in e for e in errors)


def test_bad_date_is_reported() -> None:
    chip = good_chip()
    chip["specs"]["memory_bandwidth"]["as_of"] = "last Tuesday"
    errors = validate_chip(chip)
    assert any("as_of" in e for e in errors)


def test_non_numeric_value_is_reported() -> None:
    chip = good_chip()
    chip["specs"]["memory_bandwidth"]["value"] = "fast"
    errors = validate_chip(chip)
    assert any("value" in e for e in errors)


@pytest.mark.parametrize("tag", ["published", "estimated"])
def test_published_or_estimated_needs_a_source_url(tag: str) -> None:
    chip = good_chip()
    chip["specs"]["memory_bandwidth"]["confidence"] = tag
    chip["specs"]["memory_bandwidth"]["source_url"] = None
    errors = validate_chip(chip)
    assert any("source_url" in e for e in errors)


def test_assumed_may_have_no_source_url() -> None:
    chip = good_chip()
    chip["specs"]["memory_bandwidth"]["confidence"] = "assumed"
    chip["specs"]["memory_bandwidth"]["source_url"] = None
    assert validate_chip(chip) == []


def test_chip_without_name_or_specs_is_reported() -> None:
    errors = validate_chip({})
    assert any("name" in e for e in errors)
    assert any("specs" in e for e in errors)


def test_load_chip_raises_on_invalid_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: Broken\nspecs:\n  x: {value: 1}\n")
    with pytest.raises(ValueError, match="unit"):
        load_chip(bad)


def test_validate_does_not_mutate_input() -> None:
    chip = good_chip()
    before = deepcopy(chip)
    validate_chip(chip)
    assert chip == before
