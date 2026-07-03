"""Rotatable / disordered-hydrogen perception via RDKit SMARTS.

Port of ``waterkit.typer.rotatable_bonds`` from OpenBabel to RDKit. Loads rules
from ``data/disordered_hydrogens.par`` -- each rule pairs a SMARTS pattern with
four reference atoms (1-based indices into the SMARTS) that define a dihedral and
a list of candidate rotamer angles (in degrees). :meth:`RotatableBonds.match`
applies every rule to an RDKit molecule, deduplicates matches that share the same
first reference atom (e.g. a tyrosine hydroxyl matches twice, once per ortho ring
carbon), and returns a :class:`pandas.DataFrame` describing each disordered
hydrogen the sampler must orient.

The rule-file grammar is unchanged from upstream::

    <name> <SMARTS> <ref1> <ref2> <ref3> <ref4> <angle...> [Delta <step>]

With a trailing ``Delta <step>`` the rotamers become
``numpy.arange(0, 360, step) + <angle1>``; otherwise the remaining integers are
taken verbatim as the rotamer angles (degrees).

RDKit replaces the OpenBabel SMARTS engine: ``OBSmartsPattern.Init`` becomes
:func:`rdkit.Chem.MolFromSmarts` and ``OBSmartsPattern.Match`` /
``GetMapList`` become :meth:`rdkit.Chem.Mol.GetSubstructMatches`. RDKit matches
are 0-based, so each index ``i`` maps to molecule atom id ``i + 1`` (1-based);
coordinates come from :meth:`Molecule.coordinates`.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rdkit import Chem

from ..utils.exceptions import PerceptionError

if TYPE_CHECKING:
    from ..core.molecule import Molecule

logger = logging.getLogger("waterkit.perception.rotatable_bonds")

# Mirrors the upstream rule namedtuple; ``ob_smarts`` becomes a compiled RDKit query.
_RotatableBondRule = namedtuple(
    "RotatableBondRule", ["name", "reference_atoms", "rotamers", "query"]
)

# Matches a rule line: an alphanumeric start (skips ``#`` comments and blanks),
# four bracketed SMARTS atoms, four reference-atom indices, then at least one
# (optionally signed) torsion value. Same expression as upstream.
_RULE_RE = re.compile(r"^[A-Za-z0-9].*?(\[.*?\]){4}( [0-9]){4} -?[0-9]{1,3}")

# DataFrame schema returned by :meth:`RotatableBonds.match` (unchanged from upstream).
_COLUMNS = [
    "atom_i",
    "atom_j",
    "atom_i_xyz",
    "atom_j_xyz",
    "atom_k_xyz",
    "atom_l_xyz",
    "rotamers",
    "name",
]


class RotatableBonds:
    """Detects rotatable bonds bearing disordered hydrogens (Ser/Thr/Tyr/Lys/Cys/Sec)."""

    def __init__(self, disordered_file: str | Path) -> None:
        """Load disordered-hydrogen rotamer rules from a parameter file.

        Args:
            disordered_file: Path to ``disordered_hydrogens.par`` (upstream format).

        Raises:
            PerceptionError: If the rule file cannot be read.
        """
        self._rotatable_bonds: OrderedDict[str, _RotatableBondRule] = OrderedDict()
        self._load_param_file(disordered_file)

    def _load_param_file(self, disordered_file: str | Path) -> None:
        """Parse the rule file into compiled :data:`_RotatableBondRule` entries.

        Args:
            disordered_file: Path to the parameter file.

        Raises:
            PerceptionError: If the file is missing or unreadable.
        """
        try:
            with open(disordered_file) as f:
                lines = f.readlines()
        except OSError as exc:
            raise PerceptionError(
                f"Could not read disordered-hydrogen file: {disordered_file}"
            ) from exc

        for line in lines:
            if not _RULE_RE.search(line):
                continue

            sline = line.split()
            name = sline[0]
            smarts = sline[1]

            query = Chem.MolFromSmarts(smarts)
            if query is None:
                # Upstream silently dropped patterns that failed to compile.
                logger.warning("Skipping rule %r: invalid SMARTS %r", name, smarts)
                continue

            reference_atoms = np.array(sline[2:6]).astype(int)

            if len(sline) > 7 and sline[7] == "Delta":
                rotamers = np.arange(0, 360, int(sline[-1])) + int(sline[6])
            else:
                rotamers = np.array(sline[6:]).astype(int)

            self._rotatable_bonds[name] = _RotatableBondRule(name, reference_atoms, rotamers, query)

    def match(self, molecule: Molecule) -> pd.DataFrame:
        """Type every rotatable bond bearing a disordered hydrogen.

        Args:
            molecule: A :class:`~waterkit.core.molecule.Molecule`. It is converted
                to an RDKit ``Mol`` via :meth:`Molecule.to_rdkit`; atom order is
                preserved, so RDKit atom index ``i`` maps to molecule atom id
                ``i + 1`` (1-based).

        Returns:
            A DataFrame with columns ``atom_i``, ``atom_j``, ``atom_i_xyz``,
            ``atom_j_xyz``, ``atom_k_xyz``, ``atom_l_xyz``, ``rotamers`` and
            ``name``. ``atom_i``/``atom_j`` are 1-based atom ids; the four
            ``*_xyz`` columns hold the dihedral reference-atom coordinates;
            ``rotamers`` is the candidate-angle array (degrees); ``name`` is the
            rule name. Rows are sorted ascending by ``atom_i``.

        Raises:
            PerceptionError: If the molecule cannot be converted to RDKit.
        """
        rdmol = molecule.to_rdkit()
        if rdmol is None:
            raise PerceptionError("Could not build an RDKit molecule for rotatable-bond perception")

        data = []

        for name, rule in self._rotatable_bonds.items():
            unique: list[int] = []

            for match in rdmol.GetSubstructMatches(rule.query):
                # 1-based molecule atom ids for the four reference atoms. For every
                # rule shipped with waterkit ``reference_atoms == [1, 2, 3, 4]``,
                # so this reproduces upstream's use of ``match[0:4]`` directly.
                ref_ids = [int(match[ref - 1]) + 1 for ref in rule.reference_atoms]
                atom_i, atom_j, _atom_k, _atom_l = ref_ids

                # Deduplicate patterns that match the same rotatable bond more than
                # once (e.g. a tyrosine hydroxyl, which matches once per ortho ring
                # carbon). Keep the first match per first reference atom -- exactly
                # what upstream did with OpenBabel's GetMapList.
                if atom_i in unique:
                    continue
                unique.append(atom_i)

                xyz = molecule.coordinates(ref_ids)
                data.append(
                    (
                        atom_i,
                        atom_j,
                        xyz[0],
                        xyz[1],
                        xyz[2],
                        xyz[3],
                        rule.rotamers,
                        name,
                    )
                )

        df = pd.DataFrame(data=data, columns=_COLUMNS)
        df.sort_values(by="atom_i", inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
