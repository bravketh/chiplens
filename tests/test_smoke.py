"""Smoke test: the package imports and exposes a version string."""

import chiplens


def test_package_has_version() -> None:
    assert isinstance(chiplens.__version__, str)
    assert chiplens.__version__.count(".") == 2
