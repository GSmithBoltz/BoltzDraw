"""Structure I/O and AutoDock atom typing, built on Biotite.

Replaces waterkit's OpenBabel-based reading. The receptor is consumed as a
PDBQT file (AutoDock atom types + partial charges in columns), exactly as in
upstream waterkit; :func:`parse_pdbqt` reads those columns directly so the
typing is byte-for-byte faithful. Biotite is used for general structure loading
(PDB / mmCIF), bond perception and (via hydride) hydrogen placement, and for
producing the clean PDB block that the RDKit perception layer consumes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .exceptions import ValidationError
from .typing import Coords, Int, Scalars, Shaped

if TYPE_CHECKING:
    from biotite.structure import AtomArray

logger = logging.getLogger("waterkit.io")

# Map the leading character(s) of an AutoDock atom type to an element symbol.
# AutoDock types: A (aromatic C), C, N/NA/NS, OA/OS, HD/HS, SA/S, P, the
# halogens, metals, and waterkit's water pseudo-atoms (W/SW/OW/HW/OT/HT/LP).
_AUTODOCK_TO_ELEMENT = {
    "A": "C",
    "C": "C",
    "N": "N",
    "NA": "N",
    "NS": "N",
    "O": "O",
    "OA": "O",
    "OS": "O",
    "OW": "O",
    "OT": "O",
    "SA": "S",
    "S": "S",
    "H": "H",
    "HD": "H",
    "HS": "H",
    "HW": "H",
    "HT": "H",
    "P": "P",
    "F": "F",
    "CL": "Cl",
    "BR": "Br",
    "I": "I",
    "MG": "Mg",
    "ZN": "Zn",
    "CA": "Ca",
    "MN": "Mn",
    "FE": "Fe",
    "W": "O",  # spherical water pseudo-atom -> oxygen
    "SW": "O",
    "LP": "H",  # tip5p lone pair -> treated as a light placeholder
}

# Element guessed from the first letter when the full AutoDock type is unknown.
_ELEMENT_FALLBACK = {
    "C": "C",
    "N": "N",
    "O": "O",
    "H": "H",
    "S": "S",
    "P": "P",
}


def autodock_to_element(atom_type: str) -> str:
    """Return the chemical element for an AutoDock atom type.

    Args:
        atom_type: AutoDock atom type (e.g. ``"OA"``, ``"A"``, ``"HD"``).

    Returns:
        The element symbol (e.g. ``"O"``, ``"C"``, ``"H"``); defaults to ``"C"``.
    """
    t = atom_type.strip().upper()
    if t in _AUTODOCK_TO_ELEMENT:
        return _AUTODOCK_TO_ELEMENT[t]
    if t[:1] in _ELEMENT_FALLBACK:
        return _ELEMENT_FALLBACK[t[:1]]
    return "C"


def parse_pdbqt(path: str | Path) -> dict[str, np.ndarray]:
    """Parse a PDBQT file into flat per-atom arrays.

    Reads ATOM/HETATM records and extracts the fields waterkit needs, taking the
    partial charge and AutoDock atom type from the PDBQT-specific columns
    (charge in columns 71-76, type in columns 78-79) exactly as upstream does.

    Args:
        path: Path to the PDBQT file.

    Returns:
        A dict with keys ``name``, ``resname``, ``resnum``, ``chain``, ``xyz``,
        ``q`` and ``t`` holding numpy arrays (``xyz`` has shape ``(N, 3)``).

    Raises:
        ValidationError: If the file contains no atom records.
    """
    names, resnames, resnums, chains = [], [], [], []
    xyz, charges, types = [], [], []

    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            names.append(line[12:16].strip())
            resnames.append(line[17:20].strip())
            chains.append(line[21:22].strip() or "A")
            try:
                resnums.append(int(line[22:26]))
            except ValueError:
                resnums.append(0)
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            charges.append(float(line[70:77].strip() or 0.0))
            types.append(line[77:79].strip())

    if not names:
        raise ValidationError(f"No ATOM/HETATM records found in {path}")

    return {
        "name": np.array(names, dtype="U4"),
        "resname": np.array(resnames, dtype="U3"),
        "resnum": np.array(resnums, dtype="i4"),
        "chain": np.array(chains, dtype="U1"),
        "xyz": np.array(xyz, dtype="f4"),
        "q": np.array(charges, dtype="f4"),
        "t": np.array(types, dtype="U5"),
    }


def load_structure(
    path: str | Path,
    *,
    add_hydrogens: bool = False,
    perceive_bonds: bool = True,
) -> AtomArray:
    """Load a structure file into a Biotite ``AtomArray``.

    Args:
        path: Path to a PDB, mmCIF/BinaryCIF or PDBQT file.
        add_hydrogens: If True, add hydrogens with ``hydride``.
        perceive_bonds: If True, attach a ``BondList`` via
            ``connect_via_residue_names`` (falling back to distance-based bonds).

    Returns:
        A Biotite ``AtomArray`` (single model).
    """
    import biotite.structure as struc
    from biotite.structure.io import pdb, pdbx

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdbqt":
        # Biotite cannot read the AutoDock columns reliably; treat the
        # coordinate/topology part as PDB by stripping the trailing type column.
        atoms = _read_pdbqt_as_pdb(path)
    elif suffix in (".cif", ".mmcif", ".bcif"):
        cif = (
            pdbx.CIFFile.read(str(path))
            if suffix != ".bcif"
            else pdbx.BinaryCIFFile.read(str(path))
        )
        atoms = pdbx.get_structure(cif, model=1)
    else:
        atoms = pdb.PDBFile.read(str(path)).get_structure(model=1)

    if add_hydrogens:
        import hydride

        atoms, _ = hydride.add_hydrogen(atoms)

    if perceive_bonds and atoms.bonds is None:
        try:
            atoms.bonds = struc.connect_via_residue_names(atoms)
        except Exception:
            atoms.bonds = struc.connect_via_distances(atoms)

    return atoms


def _read_pdbqt_as_pdb(path: str | Path) -> AtomArray:
    """Read a PDBQT file into a Biotite ``AtomArray`` via its PDB-like records."""
    import io as _io

    from biotite.structure.io import pdb

    # Keep only the coordinate records and trim to standard PDB width so the
    # PDB parser does not choke on AutoDock type columns.
    kept = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM", "TER", "END", "MODEL", "ENDMDL")):
            kept.append(line[:66])
    handle = _io.StringIO("\n".join(kept) + "\n")
    return pdb.PDBFile.read(handle).get_structure(model=1)


def to_pdb_block(
    names: Shaped[np.ndarray, "n"],
    resnames: Shaped[np.ndarray, "n"],
    resnums: Int[np.ndarray, "n"],
    chains: Shaped[np.ndarray, "n"],
    xyz: Coords,
    elements: Shaped[np.ndarray, "n"],
) -> str:
    """Build a well-formed PDB block from per-atom arrays (via Biotite).

    Used to hand the structure to RDKit (``Chem.MolFromPDBBlock``) for SMARTS
    perception; atom order is preserved so RDKit index ``i`` maps to atom id
    ``i + 1``. Biotite handles the exact PDB column formatting.

    Args:
        names: Atom names.
        resnames: Residue names.
        resnums: Residue numbers.
        chains: Chain ids.
        xyz: ``(N, 3)`` coordinates.
        elements: Element symbols.

    Returns:
        A PDB-format string.
    """
    import io as _io

    import biotite.structure as struc
    from biotite.structure.io import pdb

    n = len(names)
    arr = struc.AtomArray(n)
    arr.coord = np.atleast_2d(xyz).astype(np.float32)
    arr.atom_name = np.asarray(names, dtype="U6")
    arr.res_name = np.asarray(resnames, dtype="U5")
    arr.res_id = np.asarray(resnums, dtype=int)
    arr.chain_id = np.array([str(c)[:1] or "A" for c in chains], dtype="U4")
    arr.element = np.array([str(e).capitalize()[:2] for e in elements], dtype="U2")
    arr.hetero = np.zeros(n, dtype=bool)

    handle = _io.StringIO()
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(arr)
    pdb_file.write(handle)
    return handle.getvalue()


def write_pdbqt(
    path: str | Path,
    names: Shaped[np.ndarray, "n"],
    resnames: Shaped[np.ndarray, "n"],
    resnums: Int[np.ndarray, "n"],
    chains: Shaped[np.ndarray, "n"],
    xyz: Coords,
    charges: Scalars,
    atom_types: Shaped[np.ndarray, "n"],
) -> None:
    """Write per-atom arrays to a PDBQT file (waterkit column layout).

    Args:
        path: Output path.
        names: Atom names.
        resnames: Residue names.
        resnums: Residue numbers.
        chains: Chain ids.
        xyz: ``(N, 3)`` coordinates.
        charges: Partial charges.
        atom_types: AutoDock atom types.
    """
    fmt = "ATOM  %5d %-4s %-3s %1s%4d    %8.3f%8.3f%8.3f  1.00  1.00    %6.3f %-2s\n"
    with open(path, "w") as w:
        for i in range(len(names)):
            x, y, z = (float(v) for v in xyz[i])
            w.write(
                fmt
                % (
                    (i + 1) % 100000,
                    str(names[i]),
                    str(resnames[i]),
                    str(chains[i])[:1] or "A",
                    int(resnums[i]) % 10000,
                    x,
                    y,
                    z,
                    float(charges[i]),
                    str(atom_types[i]),
                )
            )


def assign_autodock_types(atoms: AtomArray) -> Shaped[np.ndarray, "n"]:
    """Assign best-effort AutoDock4 atom types to a Biotite ``AtomArray``.

    This is a coarse element/topology-based assignment used by the
    ``prepare-receptor`` path; a properly prepared PDBQT (e.g. from AmberTools)
    is preferred for production runs.

    Args:
        atoms: Biotite ``AtomArray`` (ideally with a ``BondList``).

    Returns:
        Array of AutoDock atom types, one per atom.
    """
    import biotite.structure as struc

    types = np.empty(atoms.array_length(), dtype="U5")
    elements = np.char.upper(atoms.element.astype("U2"))

    # Identify H-bond donors: hydrogens bonded to N/O become HD.
    bonded_heavy = {}
    if atoms.bonds is not None:
        bonds, _ = atoms.bonds.get_all_bonds()
    else:
        bonds = None

    for i in range(atoms.array_length()):
        el = elements[i]
        if el == "C":
            types[i] = "A" if _is_aromatic(atoms, i) else "C"
        elif el == "N":
            types[i] = "NA" if _has_lone_pair(atoms, i, bonds) else "N"
        elif el == "O":
            types[i] = "OA"
        elif el == "S":
            types[i] = "SA"
        elif el == "H":
            types[i] = "HD" if _bonded_to_polar(atoms, i, bonds, elements) else "H"
        else:
            types[i] = el.capitalize()[:2]

    _ = (struc, bonded_heavy)  # referenced for clarity; perception kept simple
    return types


def _is_aromatic(atoms: AtomArray, i: int) -> bool:
    """Heuristic: treat ring carbons of aromatic residues as aromatic."""
    aromatic_res = {"PHE", "TYR", "TRP", "HIS", "HIE", "HID", "HIP"}
    aromatic_atoms = {
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "CH2",
        "CE3",
        "CZ2",
        "CZ3",
        "ND1",
        "CE",
        "NE1",
    }
    return (
        str(atoms.res_name[i]).upper() in aromatic_res and str(atoms.atom_name[i]) in aromatic_atoms
    )


def _has_lone_pair(atoms: AtomArray, i: int, bonds: Int[np.ndarray, "n m"] | None) -> bool:
    """Heuristic: nitrogen with fewer than 3 heavy neighbours can accept."""
    if bonds is None:
        return False
    neighbours = bonds[i]
    n = int(np.sum(neighbours != -1))
    return n < 3


def _bonded_to_polar(
    atoms: AtomArray,
    i: int,
    bonds: Int[np.ndarray, "n m"] | None,
    elements: Shaped[np.ndarray, "n"],
) -> bool:
    """Return True if hydrogen ``i`` is bonded to N, O or S."""
    if bonds is None:
        return False
    for j in bonds[i]:
        if j != -1 and elements[j] in ("N", "O", "S"):
            return True
    return False


# Backwards/forwards-compatible alias used by the public surface.
def read_pdbqt(path: str | Path) -> dict[str, np.ndarray]:
    """Alias of :func:`parse_pdbqt` (read AutoDock types + charges from PDBQT)."""
    return parse_pdbqt(path)


_RE_ATOM = re.compile(r"^(ATOM|HETATM)")  # retained for callers that scan lines
