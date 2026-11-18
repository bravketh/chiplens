"""Chip data schema: loading and validating chip YAML files.

A chip file looks like::

    name: Some Chip
    specs:
      memory_bandwidth:
        value: 3.0
        unit: TB/s
        source_url: https://...
        as_of: "2024-05-12"
        confidence: published   # published | estimated | assumed

This module checks structure only. It does not check that a number is
physically sensible, and it cannot check that a source URL really says what
the file claims. That part stays a human job.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import yaml

CONFIDENCE_TAGS = ("published", "estimated", "assumed")
FIELD_KEYS = ("value", "unit", "source_url", "as_of", "confidence")


def _validate_field(path: str, field: Any) -> list[str]:
    """Return a list of problems with one spec field (empty list if fine)."""
    if not isinstance(field, dict):
        return [f"{path}: must be a mapping with keys {', '.join(FIELD_KEYS)}"]

    errors = [f"{path}: missing key '{key}'" for key in FIELD_KEYS if key not in field]

    if "value" in field:
        value = field["value"]
        # bool is a subclass of int in Python, so exclude it explicitly
        if isinstance(value, bool) or not isinstance(value, int | float):
            errors.append(f"{path}: value must be a number, got {value!r}")

    if "unit" in field and not (isinstance(field["unit"], str) and field["unit"]):
        errors.append(f"{path}: unit must be a non-empty string")

    if "as_of" in field:
        as_of = field["as_of"]
        if isinstance(as_of, dt.date):
            pass  # YAML turns unquoted 2024-05-12 into a date object
        else:
            try:
                dt.date.fromisoformat(str(as_of))
            except ValueError:
                errors.append(f"{path}: as_of must be an ISO date, got {as_of!r}")

    confidence = field.get("confidence")
    if "confidence" in field and confidence not in CONFIDENCE_TAGS:
        errors.append(
            f"{path}: confidence must be one of {CONFIDENCE_TAGS}, got {confidence!r}"
        )

    if "source_url" in field:
        url = field["source_url"]
        if confidence in ("published", "estimated"):
            if not (isinstance(url, str) and url.startswith(("http://", "https://"))):
                errors.append(
                    f"{path}: source_url is required (http/https) when "
                    f"confidence is '{confidence}'"
                )
        elif url is not None and not isinstance(url, str):
            errors.append(f"{path}: source_url must be a string or null")

    return errors


def validate_chip(chip: Any) -> list[str]:
    """Check a parsed chip mapping against the schema.

    Args:
        chip: The mapping produced by loading a chip YAML file.

    Returns:
        A list of human-readable problems. An empty list means the chip is valid.
    """
    if not isinstance(chip, dict):
        return ["chip: top level must be a mapping"]

    errors: list[str] = []

    name = chip.get("name")
    if not (isinstance(name, str) and name):
        errors.append("chip: missing or empty 'name'")

    specs = chip.get("specs")
    if not isinstance(specs, dict) or not specs:
        errors.append("chip: missing or empty 'specs'")
        return errors

    for spec_name, field in specs.items():
        errors.extend(_validate_field(f"specs.{spec_name}", field))

    return errors


def load_chip(path: str | Path) -> dict[str, Any]:
    """Load a chip YAML file and validate it.

    Args:
        path: Location of the YAML file.

    Returns:
        The parsed chip mapping.

    Raises:
        ValueError: If the file breaks the schema. The message lists every problem.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        chip = yaml.safe_load(handle)

    errors = validate_chip(chip)
    if errors:
        raise ValueError(f"{path.name} is invalid:\n  " + "\n  ".join(errors))
    return chip
