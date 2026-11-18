"""Find and read the YAML data files in ``data/``.

Folders: ``chips``, ``models``, ``assumptions``. A file's id is its name without
``.yaml``. Set the ``CHIPLENS_DATA`` environment variable to use another data
folder (for example your own chips).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chiplens.schema import load_chip

KINDS = ("chips", "models", "assumptions")

# (from_unit, to_unit) -> multiply by this factor
_CONVERSIONS = {
    ("MB", "GB"): 1e-3,
    ("GB", "MB"): 1e3,
    ("TB", "GB"): 1e3,
    ("GB", "TB"): 1e-3,
    ("GB/s", "TB/s"): 1e-3,
    ("TB/s", "GB/s"): 1e3,
}


def data_dir() -> Path:
    """Folder holding chips/, models/ and assumptions/."""
    override = os.environ.get("CHIPLENS_DATA")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data"


def _kind_dir(kind: str) -> Path:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return data_dir() / kind


def list_ids(kind: str) -> list[str]:
    """Ids available in a folder, sorted."""
    return sorted(p.stem for p in _kind_dir(kind).glob("*.yaml"))


def load(kind: str, entity_id: str) -> dict[str, Any]:
    """Load and validate one data file.

    Raises:
        FileNotFoundError: If no such id exists. The message lists valid ids.
        ValueError: If the file breaks the schema.
    """
    path = _kind_dir(kind) / f"{entity_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no {kind} file '{entity_id}'. Available: {', '.join(list_ids(kind))}"
        )
    return load_chip(path)


def _field(entity: dict[str, Any], key: str) -> dict[str, Any]:
    specs = entity["specs"]
    if key not in specs:
        raise KeyError(
            f"'{entity['name']}' has no spec '{key}'. Available: {', '.join(specs)}"
        )
    return specs[key]


def has_spec(entity: dict[str, Any], key: str) -> bool:
    """True if the entity defines this spec."""
    return key in entity["specs"]


def spec_value(entity: dict[str, Any], key: str, unit: str | None = None) -> float:
    """Read a spec's value, optionally converting to ``unit``.

    Args:
        entity: A loaded chip, model or assumptions mapping.
        key: Spec name.
        unit: Wanted unit. If None, the stored value is returned as is.

    Raises:
        KeyError: If the spec is missing.
        ValueError: If the stored unit cannot be converted to ``unit``.
    """
    field = _field(entity, key)
    value = float(field["value"])
    if unit is None or field["unit"] == unit:
        return value
    factor = _CONVERSIONS.get((field["unit"], unit))
    if factor is None:
        raise ValueError(
            f"{key} is stored in {field['unit']}; cannot convert to unit {unit}"
        )
    return value * factor


def spec_confidence(entity: dict[str, Any], key: str) -> str:
    """The confidence tag of a spec: published, estimated or assumed."""
    return _field(entity, key)["confidence"]
