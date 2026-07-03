"""Hydrogen-bond perception via RDKit SMARTS.

Port of ``waterkit.typer.hydrogen_bonds`` (originally OpenBabel-based). Loads
typing rules from ``data/waterfield.par`` (SMARTS pattern, donor/acceptor flag,
hybridization, number of HB positions, HB vector length), matches them against
an RDKit molecule derived from the Biotite/PDBQT structure, then computes
idealized hydrogen-bond partner vectors from the local geometry (sp/sp2/sp3
rules). Returns a table of anchors used to seed spherical waters.

RDKit supplies the SMARTS engine that Biotite lacks; geometry uses
:mod:`waterkit.utils.geometry`. Behavior mirrors the upstream OpenBabel
implementation:

* atom types are matched in **reverse** file order so that later (more
  specific) patterns claim an atom before earlier (more general) ones,
* inactive (``hb_type == 0``) patterns still mark atoms as visited, acting as
  negative filters that block more general active patterns,
* the first atom of every SMARTS pattern is the hydrogen-bond atom (the
  hydrogen for donors, the heavy atom for acceptors).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from rdkit import Chem

from ..utils import geometry
from ..utils.exceptions import PerceptionError
from ..utils.typing import Coords

if TYPE_CHECKING:
    from ..core.molecule import Molecule

logger = logging.getLogger("waterkit.perception.hydrogen_bonds")

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "waterfield.par"

#: Output column order of :meth:`HydrogenBonds.match` (matches upstream).
_COLUMNS = ["atom_i", "vector_xyz", "anchor_type", "anchor_name"]


class _AtomType(NamedTuple):
    """A single hydrogen-bond typing rule parsed from ``waterfield.par``."""

    hb_type: int  # 0: inactive, 1: donor, 2: acceptor
    hb_strength: float
    hyb: int  # hybridization {1: sp, 2: sp2, 3: sp3}
    n_water: int  # number of HB partner positions to place
    hb_length: float  # anchor -> water distance
    pattern: Chem.Mol | None  # compiled RDKit SMARTS query (None if invalid)
    smarts: str  # raw SMARTS string


class HydrogenBonds:
    """Detects hydrogen-bond anchors and their idealized partner vectors."""

    def __init__(self, waterfield_file: str | Path | None = None) -> None:
        """Initialize the typer from a waterfield parameter file.

        Args:
            waterfield_file: Path to a ``waterfield.par``-formatted file. If
                ``None``, the packaged ``data/waterfield.par`` is used.
        """
        # A plain dict preserves insertion order (Python >= 3.7), which is
        # essential here: matching iterates the types in reverse file order.
        self._atom_types: dict[str, _AtomType] = {}
        self._load_param_file(Path(waterfield_file) if waterfield_file else _DATA_FILE)

    def _load_param_file(self, fname: Path) -> None:
        """Parse the waterfield file and compile each SMARTS pattern.

        The file format is one rule per ``ATOM`` line::

            ATOM  NAME  HB_TYPE  STRENGTH  HYB  #WATER  HB_LENGTH  SMARTS  # comment

        Args:
            fname: Path to the parameter file.

        Raises:
            PerceptionError: If the file cannot be read.
        """
        try:
            lines = fname.read_text().splitlines()
        except OSError as exc:
            raise PerceptionError(f"Could not read waterfield file {fname}: {exc}") from exc

        for line in lines:
            if not line.startswith("ATOM"):
                continue

            # Split on whitespace and drop empty fields.
            sline = [field for field in line.split(" ") if field]
            if len(sline) < 8:
                logger.warning("Skipping malformed waterfield line: %s", line.strip())
                continue

            smarts = sline[7]
            pattern = Chem.MolFromSmarts(smarts)

            if pattern is None:
                logger.warning("Invalid SMARTS pattern %s for atom type %s.", smarts, sline[1])
                continue

            name = sline[1]
            self._atom_types[name] = _AtomType(
                hb_type=int(sline[2]),
                hb_strength=float(sline[3]),
                hyb=int(sline[4]),
                n_water=int(sline[5]),
                hb_length=float(sline[6]),
                pattern=pattern,
                smarts=smarts,
            )

    @staticmethod
    def _push_atom_to_end(
        lst: list[Chem.Atom], atomic_nums: int | list[int] | tuple[int, ...]
    ) -> list[Chem.Atom]:
        """Move atoms of the given atomic number(s) to the end of the list.

        Relative order is preserved. Used to place hydrogens last so that heavy
        neighbors are always considered first by the geometry rules.

        Args:
            lst: List of RDKit atoms (modified in place and returned).
            atomic_nums: Atomic number (or list/tuple of them) to push back.

        Returns:
            The reordered list.
        """
        if not isinstance(atomic_nums, (list, tuple)):
            atomic_nums = [atomic_nums]

        for atomic_num in atomic_nums:
            pop_count = 0
            idx = [i for i, x in enumerate(lst) if x.GetAtomicNum() == atomic_num]

            for i in idx:
                lst.append(lst.pop(i - pop_count))
                pop_count += 1

        return lst

    def _neighbor_atoms(
        self, mol: Chem.Mol, start_index: int, depth: int = 1, hydrogen: bool = True
    ) -> list[list[Chem.Atom]]:
        """Return neighbor atoms grouped by BFS depth from ``start_index``.

        Replicates the upstream OpenBabel BFS (``OBAtomAtomIter``): nodes are
        marked visited only when popped, hydrogens are pushed to the end of each
        level, and the start atom is always included regardless of ``hydrogen``.

        Args:
            mol: RDKit molecule.
            start_index: 0-based index of the origin atom.
            depth: Maximum BFS depth to traverse.
            hydrogen: If ``True`` keep hydrogens, otherwise skip them while
                enqueuing neighbors.

        Returns:
            A nested list where entry ``d`` holds the atoms at depth ``d``
            (heavy atoms first, hydrogens last).
        """
        n_atoms = mol.GetNumAtoms()
        visited = [False] * n_atoms
        neighbors: list[list[Chem.Atom]] = []
        queue: list[tuple[int, int]] = [(start_index, 0)]
        atomic_num_to_keep = 1 if hydrogen else 2

        while queue:
            i, d = queue.pop(0)
            atom = mol.GetAtomWithIdx(int(i))

            # Grow the structure lazily, one depth level at a time.
            try:
                neighbors[d].append(atom)
            except IndexError:
                neighbors.append([])
                neighbors[d].append(atom)

            visited[i] = True

            if d < depth:
                for a in atom.GetNeighbors():
                    j = a.GetIdx()
                    if not visited[j] and a.GetAtomicNum() >= atomic_num_to_keep:
                        queue.append((j, d + 1))

        # Push all hydrogens (atomic number 1) to the end of each level.
        neighbors = [self._push_atom_to_end(level, 1) for level in neighbors]

        return neighbors

    def _neighbor_atom_coordinates(
        self,
        mol: Chem.Mol,
        coords: Coords,
        id_atom: int,
        depth: int = 1,
        hydrogen: bool = True,
    ) -> list[Coords]:
        """Return neighbor coordinates grouped by BFS depth.

        Args:
            mol: RDKit molecule (for connectivity).
            coords: ``(N, 3)`` array of atom coordinates, indexed by 0-based
                RDKit atom index.
            id_atom: 0-based index of the origin atom.
            depth: Maximum BFS depth to traverse.
            hydrogen: Whether to keep hydrogens.

        Returns:
            A list of ``(M, 3)`` coordinate arrays, one per depth level.
        """
        atoms = self._neighbor_atoms(mol, id_atom, depth, hydrogen)
        return [np.array([coords[atom.GetIdx()] for atom in level]) for level in atoms]

    def _hb_vectors(
        self,
        mol: Chem.Mol,
        coords: Coords,
        idx: int,
        hyb: int,
        n_hbond: int,
        hb_length: float,
    ) -> Coords:
        """Compute idealized hydrogen-bond partner vectors for one anchor atom.

        The geometry is selected from the atom's hybridization and the requested
        number of HB positions, exactly as in upstream WaterKit.

        Args:
            mol: RDKit molecule (for connectivity).
            coords: ``(N, 3)`` array of atom coordinates (0-based RDKit index).
            idx: 0-based index of the anchor atom.
            hyb: Hybridization {1: sp, 2: sp2, 3: sp3}.
            n_hbond: Number of HB partner positions to place.
            hb_length: Anchor -> partner distance.

        Returns:
            An ``(n_hbond, 3)`` array of partner coordinates.
        """
        vectors = []

        # Origin (anchor) atom and the coordinates of its neighbors by depth.
        anchor_xyz = coords[idx]
        neighbors_xyz = self._neighbor_atom_coordinates(mol, coords, idx, depth=2)
        neighbor1_xyz = neighbors_xyz[1][0]

        # Defined to satisfy the rotation loop below for the linear/above cases.
        r = None
        p = None
        angles: list[float] = []

        if hyb == 1:
            # Position of water is linear: only the origin and first neighbor.
            # Example: H donor.
            if n_hbond == 1:
                r = None
                p = anchor_xyz + geometry.vector(neighbor1_xyz, anchor_xyz)
                angles = [0]

            if n_hbond == 3:
                hyb = 3

        elif hyb == 2:
            # Position of water is just above the origin atom; needs the 2 direct
            # neighbors. Example: nitrogen.
            if n_hbond == 1:
                neighbor2_xyz = neighbors_xyz[1][1]

                r = None
                p = geometry.atom_to_move(anchor_xyz, [neighbor1_xyz, neighbor2_xyz])
                angles = [0]

            # Two waters separated by 120 degrees, aligned with the depth-2
            # neighbors of the origin atom. Example: backbone oxygen.
            elif n_hbond == 2:
                neighbor2_xyz = neighbors_xyz[2][0]

                r = geometry.rotation_axis(
                    neighbor1_xyz, anchor_xyz, neighbor2_xyz, origin=anchor_xyz
                )
                p = neighbor1_xyz
                angles = [-np.radians(120), np.radians(120)]

            elif n_hbond == 3:
                hyb = 3

        if hyb == 3:
            # Position of water is just above the origin atom; needs the 3 direct
            # neighbors (tetrahedral). Example: ammonia.
            if n_hbond == 1:
                neighbor2_xyz = neighbors_xyz[1][1]
                neighbor3_xyz = neighbors_xyz[1][2]

                # Normalize the bonds, otherwise the water is poorly placed.
                v1 = anchor_xyz + geometry.normalize(geometry.vector(anchor_xyz, neighbor1_xyz))
                v2 = anchor_xyz + geometry.normalize(geometry.vector(anchor_xyz, neighbor2_xyz))
                v3 = anchor_xyz + geometry.normalize(geometry.vector(anchor_xyz, neighbor3_xyz))

                r = None
                p = geometry.atom_to_move(anchor_xyz, [v1, v2, v3])
                angles = [0]

            # Two waters separated by 109 degrees, perpendicular to the neighbor
            # atoms of the origin. Example: hydroxyl oxygen.
            elif n_hbond == 2:
                neighbor2_xyz = neighbors_xyz[1][1]
                v1 = anchor_xyz + geometry.normalize(geometry.vector(anchor_xyz, neighbor1_xyz))
                v2 = anchor_xyz + geometry.normalize(geometry.vector(anchor_xyz, neighbor2_xyz))

                r = anchor_xyz + geometry.normalize(geometry.vector(v1, v2))
                p = geometry.atom_to_move(anchor_xyz, [v1, v2])
                angles = [-np.radians(60), np.radians(60)]

            # Three waters separated by 109 degrees; no reference, so they are
            # placed using a random perpendicular axis. Example: DMSO.
            elif n_hbond == 3:
                # Vector between the anchor and the only neighbor atom.
                v = geometry.vector(anchor_xyz, neighbor1_xyz)
                v = geometry.normalize(v)

                # Pick a random vector perpendicular to v as the rotation axis.
                r = anchor_xyz + geometry.get_perpendicular_vector(v)

                # Place a pseudo atom (the first water molecule).
                p = geometry.rotate_point(neighbor1_xyz, anchor_xyz, r, np.radians(109.471))
                # The next rotation axis is along the neighbor -> origin bond.
                r = anchor_xyz + geometry.normalize(geometry.vector(neighbor1_xyz, anchor_xyz))
                angles = [0, -np.radians(120), np.radians(120)]

        # Rotate p around the axis to obtain each vector (if needed) and resize.
        for angle in angles:
            vec = p
            if angle != 0.0:
                vec = geometry.rotate_point(vec, anchor_xyz, r, angle)
            vec = geometry.resize_vector(vec, hb_length, anchor_xyz)
            vectors.append(vec)

        return np.array(vectors)

    def match(self, molecule: Molecule) -> pd.DataFrame:
        """Guess all hydrogen-bond anchors (donor/acceptor) in a molecule.

        The molecule is converted to RDKit (atom order preserved, so RDKit atom
        index ``i`` maps to 1-based atom id ``i + 1``), then every SMARTS rule is
        matched in reverse file order. Inactive rules block more general ones,
        and for each accepted anchor the idealized partner vectors are computed.

        Args:
            molecule: A :class:`waterkit.core.molecule.Molecule`.

        Returns:
            A DataFrame with columns ``atom_i`` (1-based anchor atom id),
            ``vector_xyz`` (``(3,)`` partner coordinates), ``anchor_type``
            (``"donor"`` or ``"acceptor"``), and ``anchor_name`` (waterfield
            label), sorted by ``atom_i``.

        Raises:
            PerceptionError: If the molecule cannot be converted to RDKit or the
                atom count does not match (making the index mapping unsafe).
        """
        mol = molecule.to_rdkit()
        if mol is None:
            raise PerceptionError("RDKit could not build a molecule for HB perception")

        n_atoms = mol.GetNumAtoms()
        if n_atoms != len(molecule.atoms):
            raise PerceptionError(
                f"RDKit atom count ({n_atoms}) does not match molecule "
                f"({len(molecule.atoms)}); atom index mapping is unsafe"
            )

        coords = np.asarray(molecule.coordinates(), dtype=np.float64)

        data: list[list] = []
        # Track visited atoms by 0-based RDKit index.
        visited = [False] * n_atoms

        # Iterate types in reverse so later/more-specific patterns win.
        for name in list(self._atom_types.keys())[::-1]:
            atom_type = self._atom_types[name]

            matches = mol.GetSubstructMatches(atom_type.pattern, uniquify=False)

            if atom_type.hb_type == 1:
                hb_type = "donor"
            elif atom_type.hb_type == 2:
                hb_type = "acceptor"
            else:
                hb_type = None

            for match in matches:
                # The first atom of the SMARTS pattern is the HB atom.
                idx = match[0]

                # Inactive types still claim the atom, blocking general types.
                if hb_type is None and not visited[idx]:
                    visited[idx] = True

                if not visited[idx]:
                    visited[idx] = True

                    try:
                        vector_xyzs = self._hb_vectors(
                            mol,
                            coords,
                            idx,
                            atom_type.hyb,
                            atom_type.n_water,
                            atom_type.hb_length,
                        )
                        for vector_xyz in vector_xyzs:
                            # Store the 1-based atom id (RDKit idx + 1).
                            data.append([idx + 1, vector_xyz, hb_type, name])
                    except Exception:
                        logger.warning(
                            "Could not determine hydrogen bond vectors on atom %d of type %s.",
                            idx + 1,
                            name,
                        )

        df = pd.DataFrame(data=data, columns=_COLUMNS)
        df.sort_values(by="atom_i", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
