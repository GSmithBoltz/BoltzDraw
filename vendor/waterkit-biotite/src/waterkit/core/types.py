"""Core data types for waterkit.

The sampler tracks per-atom records (coordinates, AutoDock atom type, partial
charge) plus per-molecule hydrogen-bond and rotatable-bond tables. We keep the
lightweight record/array layout from upstream waterkit but expose it through
typed containers in pocketeer style.
"""

from dataclasses import dataclass, field

import numpy as np

from ..utils.typing import Float, Scalars, Vec3

# Structured dtype for an atom record (mirrors waterkit's Molecule.atoms layout).
ATOM_DTYPE = np.dtype(
    [
        ("i", "i4"),  # 1-based atom id
        ("name", "U4"),  # atom name
        ("resname", "U3"),  # residue name
        ("resnum", "i4"),  # residue number
        ("chain", "U1"),  # chain id
        ("xyz", "f4", (3,)),  # coordinates
        ("q", "f4"),  # partial charge
        ("t", "U5"),  # AutoDock atom type
    ]
)


@dataclass(frozen=True)
class HydrogenBond:
    """A single hydrogen-bond anchor and its idealized partner vector."""

    atom_i: int  # anchor atom id (1-based)
    vector_xyz: Vec3  # idealized position of the HB partner
    anchor_type: str  # "donor" or "acceptor"
    anchor_name: str  # waterfield label, e.g. "O_L_000"


@dataclass(frozen=True)
class RotatableBond:
    """A rotatable (disordered-hydrogen) bond and its rotamer definition."""

    atom_i: int
    atom_j: int
    reference_xyz: Float[np.ndarray, "4 3"]  # 4 atoms defining the dihedral
    angles: Scalars  # candidate rotamer angles (radians)
    name: str


@dataclass
class ShellRecord:
    """Per-water bookkeeping for one hydration shell entry."""

    shell_id: int
    energy_position: float = np.nan
    energy_orientation: float = np.nan
    info: dict = field(default_factory=dict)
