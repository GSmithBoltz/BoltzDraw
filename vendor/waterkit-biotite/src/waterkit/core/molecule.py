"""Molecule container backed by Biotite/RDKit (replaces waterkit's OpenBabel wrapper).

A ``Molecule`` holds atoms as a structured numpy record array (dtype
:data:`waterkit.core.types.ATOM_DTYPE`) — the same layout the upstream sampler
relies on — and can produce an RDKit ``Mol`` on demand for SMARTS-based
hydrogen-bond / rotatable-bond perception. Atom ids are **1-based** throughout
(matching upstream); the record array is 0-based, so ``coordinates([3])``
returns the row ``atoms[2]``. RDKit atom index ``i`` maps to atom id ``i + 1``
because :meth:`to_rdkit` preserves atom order.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..utils import io
from ..utils.typing import Coords, Indices, Scalars, Shaped, Vec3
from .types import ATOM_DTYPE

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger("waterkit.molecule")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Molecule:
    """A receptor or ligand molecule with AutoDock typing and HB/rotatable tables."""

    def __init__(self, atoms: Shaped[np.ndarray, "n"] | None) -> None:
        """Initialize from a structured atom record array (or ``None`` for an empty molecule).

        Args:
            atoms: A structured array of :data:`ATOM_DTYPE` (one record per atom),
                or ``None`` for an empty molecule.
        """
        self.atoms = atoms
        self.hydrogen_bonds = None  # pandas.DataFrame, filled by the perception layer
        self.rotatable_bonds = None  # pandas.DataFrame, filled by the perception layer

    @classmethod
    def from_file(
        cls,
        fname: str | Path,
        guess_hydrogen_bonds: bool = True,
        guess_disordered_hydrogens: bool = True,
    ) -> Molecule:
        """Build a ``Molecule`` from a structure file (PDBQT preferred).

        Args:
            fname: Path to the structure file. PDBQT carries AutoDock atom types
                and partial charges directly; other formats are read via Biotite
                and assigned best-effort types.
            guess_hydrogen_bonds: Run hydrogen-bond perception (waterfield.par).
            guess_disordered_hydrogens: Run rotatable-bond perception.

        Returns:
            The constructed :class:`Molecule`.
        """
        fname = str(fname)
        ext = os.path.splitext(fname)[1].lower().lstrip(".")

        if ext == "pdbqt":
            rec = io.parse_pdbqt(fname)
        else:
            rec = cls._records_from_biotite(fname)

        n = len(rec["name"])
        atoms = np.zeros(n, dtype=ATOM_DTYPE)
        atoms["i"] = np.arange(1, n + 1)
        atoms["name"] = rec["name"]
        atoms["resname"] = rec["resname"]
        atoms["resnum"] = rec["resnum"]
        atoms["chain"] = rec["chain"]
        atoms["xyz"] = rec["xyz"]
        atoms["q"] = rec["q"]
        atoms["t"] = rec["t"]

        mol = cls(atoms)

        if guess_disordered_hydrogens:
            mol._guess_rotatable_bonds()
        if guess_hydrogen_bonds:
            mol._guess_hydrogen_bond_anchors()

        return mol

    @staticmethod
    def _records_from_biotite(fname: str) -> dict[str, np.ndarray]:
        """Read a non-PDBQT structure via Biotite into waterkit's flat record dict.

        Args:
            fname: Path to a non-PDBQT structure file (PDB / mmCIF / BinaryCIF).

        Returns:
            A dict with keys ``name``, ``resname``, ``resnum``, ``chain``, ``xyz``
            (shape ``(N, 3)``), ``q`` and ``t``, each holding a numpy array.
        """
        atoms = io.load_structure(fname, add_hydrogens=False, perceive_bonds=True)
        n = atoms.array_length()
        types = io.assign_autodock_types(atoms)
        charges = atoms.charge if atoms.charge is not None else np.zeros(n)
        return {
            "name": atoms.atom_name.astype("U4"),
            "resname": atoms.res_name.astype("U3"),
            "resnum": atoms.res_id.astype("i4"),
            "chain": atoms.chain_id.astype("U1"),
            "xyz": atoms.coord.astype("f4"),
            "q": np.asarray(charges, dtype="f4"),
            "t": types,
        }

    def to_rdkit(self) -> Mol | None:
        """Return an RDKit ``Mol`` with 3D coords + perceived bonds.

        Atom order is preserved, so RDKit atom index ``i`` corresponds to this
        molecule's atom id ``i + 1``. Bonds are perceived from element + proximity
        (and standard-residue templates by RDKit's PDB reader).

        Returns:
            An RDKit ``Mol`` (sanitized where possible), or ``None`` on failure.
        """
        from rdkit import Chem

        elements = np.array([io.autodock_to_element(t) for t in self.atoms["t"]], dtype="U2")
        block = io.to_pdb_block(
            self.atoms["name"],
            self.atoms["resname"],
            self.atoms["resnum"],
            self.atoms["chain"],
            np.atleast_2d(self.atoms["xyz"]),
            elements,
        )
        mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False, proximityBonding=True)
        if mol is None:
            logger.warning("RDKit could not parse the molecule for perception")
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            logger.warning("RDKit sanitization failed: %s", exc)
        return mol

    def is_water(self) -> bool:
        """Return False (overridden by :class:`~waterkit.core.water.Water`)."""
        return False

    def coordinates(self, atom_ids: Indices | None = None) -> Coords:
        """Return ``(N, 3)`` coordinates for the given **1-based** atom ids (all atoms if None).

        Args:
            atom_ids: 1-based atom ids (list or integer array); ``None`` for all atoms.

        Returns:
            Coordinates of shape ``(N, 3)``.
        """
        if atom_ids is not None and self.atoms.size > 1:
            atom_ids = np.array(atom_ids, dtype=int)
            xyz = self.atoms[atom_ids - 1]["xyz"]
        else:
            xyz = self.atoms["xyz"]
        return np.atleast_2d(xyz).copy()

    def atom_types(self, atom_ids: Indices | None = None) -> list[str]:
        """Return AutoDock atom types for the given **1-based** atom ids.

        Args:
            atom_ids: 1-based atom ids (list or integer array); ``None`` for all atoms.

        Returns:
            One AutoDock atom-type string per requested atom.
        """
        if atom_ids is not None and self.atoms.size > 1:
            atom_ids = np.array(atom_ids, dtype=int)
            t = self.atoms[atom_ids - 1]["t"]
        else:
            t = self.atoms["t"]
        return t.tolist()

    def partial_charges(self, atom_ids: Indices | None = None) -> Scalars:
        """Return partial charges for the given **1-based** atom ids.

        Args:
            atom_ids: 1-based atom ids (list or integer array); ``None`` for all atoms.

        Returns:
            One partial charge per requested atom.
        """
        if atom_ids is not None and self.atoms.size > 1:
            atom_ids = np.array(atom_ids, dtype=int)
            q = self.atoms[atom_ids - 1]["q"]
        else:
            q = self.atoms["q"]
        return q.copy()

    def atom_informations(self, atom_ids: Indices | None = None) -> Shaped[np.ndarray, "n"]:
        """Return atom records (``i``, ``xyz``, ``q``, ``t``) for force-field scoring.

        Args:
            atom_ids: 1-based atom ids (list or integer array); ``None`` for all atoms.

        Returns:
            A structured (sub-``ATOM_DTYPE``) record array with one record per atom,
            carrying the ``i``, ``xyz``, ``q`` and ``t`` fields.
        """
        if atom_ids is not None:
            atom_ids = np.array(atom_ids, dtype=int)
            atoms = self.atoms[atom_ids - 1][["i", "xyz", "q", "t"]]
        else:
            atoms = self.atoms[["i", "xyz", "q", "t"]]
        return np.atleast_1d(atoms).copy()

    def update_coordinates(self, xyz: Vec3, atom_id: int) -> bool:
        """Update the coordinates of a single (1-based) atom.

        Args:
            xyz: New ``(x, y, z)`` coordinates for the atom.
            atom_id: 1-based id of the atom to move.

        Returns:
            ``True`` if the atom existed and was updated, ``False`` otherwise.
        """
        if atom_id <= self.atoms.size:
            if self.atoms.size > 1:
                self.atoms[atom_id - 1]["xyz"] = xyz
            else:
                self.atoms["xyz"] = xyz
            return True
        return False

    def _guess_rotatable_bonds(self) -> None:
        """Fill ``rotatable_bonds`` using the RDKit-based perception layer."""
        try:
            from ..perception import RotatableBonds

            typer = RotatableBonds(_DATA_DIR / "disordered_hydrogens.par")
            self.rotatable_bonds = typer.match(self)
        except (ImportError, NotImplementedError) as exc:
            logger.warning("Rotatable-bond perception unavailable: %s", exc)
            self.rotatable_bonds = None

    def _guess_hydrogen_bond_anchors(self) -> None:
        """Fill ``hydrogen_bonds`` using the RDKit-based perception layer."""
        try:
            from ..perception import HydrogenBonds

            typer = HydrogenBonds(_DATA_DIR / "waterfield.par")
            self.hydrogen_bonds = typer.match(self)
        except (ImportError, NotImplementedError) as exc:
            logger.warning("Hydrogen-bond perception unavailable: %s", exc)
            self.hydrogen_bonds = None
