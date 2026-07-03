"""Miscellaneous helpers ported from ``waterkit.utils`` (filesystem, chunking, typing).

Pure-Python utilities that are neither geometry nor Boltzmann sampling: directory
helpers, list chunking for parallel frames, AMBER->AutoDock atom-type conversion,
and water-map electrostatics baking.
"""

from __future__ import annotations

import contextlib
import copy
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.grid import Map
    from ..core.molecule import Molecule

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def data_path(*parts: str) -> Path:
    """Return a path inside the packaged ``data/`` directory.

    Args:
        *parts: Path components to join onto the ``data/`` directory.

    Returns:
        The absolute path ``data/<parts...>``.
    """
    return DATA_DIR.joinpath(*parts)


@contextlib.contextmanager
def temporary_directory(
    suffix: str | None = None,
    prefix: str | None = None,
    dir: str | None = None,
    clean: bool = True,
) -> Iterator[str]:
    """Create, ``chdir`` into, and (optionally) clean up a temporary directory.

    Args:
        suffix: Optional suffix for the temporary directory name.
        prefix: Optional prefix for the temporary directory name.
        dir: Parent directory to create the temporary directory in.
        clean: If ``True`` (default) remove the directory on exit.

    Yields:
        The path to the temporary directory (also the current working directory
        for the duration of the ``with`` block).
    """
    temp_dir = tempfile.mkdtemp(suffix, prefix, dir)
    cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        yield temp_dir
    finally:
        os.chdir(cwd)
        if clean:
            shutil.rmtree(temp_dir)


def is_writable(pathname: str | Path) -> None:
    """Check that a directory is writable.

    Args:
        pathname: Directory to test.

    Raises:
        RuntimeError: If a temporary file cannot be created in ``pathname``.
    """
    try:
        testfile = tempfile.NamedTemporaryFile(dir=str(pathname))
        testfile.close()
    except (PermissionError, FileNotFoundError) as e:
        raise RuntimeError(f"Cannot write in directory {pathname}.") from e


def execute_command(cmd_line: str) -> tuple[str, str]:
    """Run a shell command and capture its output.

    Args:
        cmd_line: The command line to run (split on whitespace into argv).

    Returns:
        A ``(stdout, stderr)`` tuple of the command's decoded output.
    """
    args = cmd_line.split()
    p = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    output, errors = p.communicate()
    return output, errors


def split_list_in_chunks(size: int, n: int) -> list[tuple[int, int]]:
    """Split ``range(size)`` into ``n`` contiguous ``(start, stop)`` chunks.

    Used to partition independent frames across worker processes.

    Args:
        size: Number of items to split (e.g. number of frames).
        n: Number of (roughly equal) chunks to produce.

    Returns:
        A list of ``(start, stop)`` index pairs, both ends inclusive.
    """
    return [(chunk[0], chunk[-1]) for chunk in np.array_split(range(size), n)]


# AMBER atom type -> AutoDock atom type mapping (ported verbatim from waterkit).
AMBER_TO_AUTODOCK = {
    "N3": "N",
    "H": "HD",
    "CX": "C",
    "HP": "H",
    "CT": "C",
    "HC": "H",
    "C": "C",
    "O": "OA",
    "N": "N",
    "H1": "H",
    "C3": "C",
    "3C": "C",
    "C2": "C",
    "2C": "C",
    "CO": "C",
    "O2": "OA",
    "OH": "OA",
    "HO": "HD",
    "SH": "SA",
    "HS": "HD",
    "CA": "A",
    "HA": "H",
    "S": "SA",
    "C8": "C",
    "N2": "N",
    "CC": "A",
    "NB": "NA",
    "CR": "A",
    "CV": "A",
    "H5": "H",
    "NA": "N",
    "CW": "A",
    "H4": "H",
    "C*": "A",
    "CN": "A",
    "CB": "A",
    "Zn2+": "Zn",
    "Zn": "Zn",
    "Mn2+": "Mn",
    "Mn": "Mn",
    "XC": "C",
    "br": "Br",
    "c": "C",
    "c1": "C",
    "c2": "C",
    "c3": "C",
    "ca": "A",
    "cc": "A",
    "cd": "A",
    "ce": "C",
    "cf": "C",
    "cl": "Cl",
    "cp": "A",
    "cq": "A",
    "cu": "C",
    "cv": "C",
    "cx": "C",
    "cy": "C",
    "c5": "C",
    "c6": "C",
    "cz": "C",
    "cs": "C",
    "cg": "C",
    "ch": "C",
    "f": "F",
    "h1": "H",
    "h2": "H",
    "h3": "H",
    "h4": "H",
    "h5": "H",
    "ha": "H",
    "hc": "H",
    "hn": "HD",
    "ho": "HD",
    "hp": "HD",
    "hs": "HD",
    "hx": "H",
    "i": "I",
    "n": "N",
    "n1": "NA",
    "n2": "N",
    "n3": "N",
    "n4": "N",
    "n5": "N",
    "n6": "N",
    "n7": "N",
    "n8": "N",
    "n9": "N",
    "na": "N",
    "nb": "N",
    "nc": "N",
    "nd": "N",
    "nh": "N",
    "ne": "N",
    "nf": "N",
    "no": "N",
    "n+": "N",
    "nx": "N",
    "ny": "N",
    "nz": "N",
    "ns": "N",
    "nt": "N",
    "nu": "N",
    "nv": "N",
    "ni": "N",
    "nj": "N",
    "nk": "N",
    "nl": "N",
    "nm": "N",
    "nn": "N",
    "np": "N",
    "nq": "N",
    "o": "OA",
    "oh": "OA",
    "os": "OA",
    "op": "OA",
    "oq": "OA",
    "p2": "P",
    "p3": "P",
    "p4": "P",
    "p5": "P",
    "pb": "P",
    "pc": "P",
    "pd": "P",
    "pe": "P",
    "pf": "P",
    "px": "P",
    "py": "P",
    "s": "S",
    "s2": "SA",
    "s4": "S",
    "s6": "S",
    "sh": "SA",
    "ss": "SA",
    "sx": "S",
    "sy": "S",
    "sp": "S",
    "sq": "S",
}


def convert_amber_to_autodock_types(molecule: Molecule) -> Molecule:
    """Return a deep copy of ``molecule`` with AMBER atom types mapped to AutoDock.

    Args:
        molecule: A molecule whose ``atoms`` record array carries AMBER types in
            the ``t`` field and residue/atom names in ``resname``/``name``.

    Returns:
        A new molecule with ``atoms['t']`` rewritten to AutoDock atom types.
    """
    molecule = copy.deepcopy(molecule)

    for atom in molecule.atoms:
        if atom["resname"] == "TYR" and atom["name"] == "CZ" and atom["t"] == "C":
            atom["t"] = "A"
        elif atom["resname"] == "ARG" and atom["name"] == "CZ" and atom["t"] == "CA":
            atom["t"] = "C"
        else:
            atom["t"] = AMBER_TO_AUTODOCK[atom["t"]]

    return molecule


def prepare_water_map(ad_map: Map, water_model: str = "tip3p", dielectric: float = 1.0) -> None:
    """Bake electrostatics into the water affinity maps (TIP3P/TIP5P).

    H atoms and lone pairs have no van der Waals radius in TIP3P/TIP5P, so their
    receptor interaction is purely electrostatic: each is the electrostatic map
    scaled by the partial charge. Requires ``ad_map`` to expose ``info``,
    ``apply_operation_on_maps``, ``combine`` and ``delete_map`` (see
    :class:`waterkit.core.grid.Map`).

    Args:
        ad_map: The receptor :class:`~waterkit.core.grid.Map`.
        water_model: ``"tip3p"`` or ``"tip5p"``.
        dielectric: Dielectric constant used to scale the electrostatic map.
    """
    e_type = "Electrostatics"
    dielectric = float(dielectric)
    map_info = ad_map.info()

    if water_model == "tip3p":
        ow_type, hw_type = "OW", "HW"
        ow_q, hw_q = -0.834, 0.417
        missing = {ow_type, e_type}.difference(map_info["maps"])
    elif water_model == "tip5p":
        ot_type, hw_type, lw_type = "OT", "HT", "LP"
        hw_q, lw_q = 0.241, -0.241
        missing = {ot_type, lw_type, e_type}.difference(map_info["maps"])
    else:
        raise RuntimeError(f"Water model {water_model} unknown.")

    if missing:
        raise RuntimeError(
            f"Atom type(s) {list(missing)} is(are) missing for {water_model} water model."
        )

    ad_map.apply_operation_on_maps(hw_type, e_type, f"x * {hw_q} / {dielectric}")

    if water_model == "tip3p":
        ad_map.apply_operation_on_maps(e_type, e_type, f"x * {ow_q} / {dielectric}")
        ad_map.combine(ow_type, [ow_type, e_type], how="add")
    elif water_model == "tip5p":
        ad_map.apply_operation_on_maps(lw_type, e_type, f"x * {lw_q} / {dielectric}")

    ad_map.delete_map("Electrostatics")
    ad_map.delete_map("Desolvation")
