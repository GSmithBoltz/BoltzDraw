"""Smoke tests for the package scaffold.

These confirm the package imports and its public surface is wired up. They are
replaced/expanded with real behavioural tests as each phase lands.
"""

import waterkit


def test_version() -> None:
    """The package exposes a version string."""
    assert isinstance(waterkit.__version__, str)


def test_public_api_present() -> None:
    """The advertised public API names are importable."""
    for name in (
        "hydrate",
        "Molecule",
        "Water",
        "WaterBox",
        "WaterSampler",
        "Map",
        "AutoDockForceField",
        "HydrogenBonds",
        "RotatableBonds",
        "load_structure",
    ):
        assert hasattr(waterkit, name), f"missing public symbol: {name}"
