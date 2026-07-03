"""WaterKit: grid-based explicit-water sampling in protein binding sites, on Biotite.

A reimplementation of the WaterKit method (Eberhardt & Forli, JCTC 2023) built on
Biotite + hydride (structure, bonds, hydrogens) and RDKit (SMARTS perception),
with a pure-Python AutoDock4 force field and grid generator -- no OpenBabel and
no external AutoGrid binary.
"""

__version__ = "0.1.0"

# Public API
from .api import hydrate
from .core import (
    AutoDockForceField,
    Map,
    Molecule,
    SphericalWaterMap,
    Water,
    WaterBox,
    WaterSampler,
    generate_maps,
)
from .perception import HydrogenBonds, RotatableBonds
from .utils import load_structure

__all__ = [
    "AutoDockForceField",
    "HydrogenBonds",
    "Map",
    "Molecule",
    "RotatableBonds",
    "SphericalWaterMap",
    "Water",
    "WaterBox",
    "WaterSampler",
    "__version__",
    "generate_maps",
    "hydrate",
    "load_structure",
]
