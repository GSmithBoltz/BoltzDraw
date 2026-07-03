"""AutoDock4 force field (pure numpy / pandas).

Faithful port of ``waterkit.forcefield`` -- no OpenBabel, no external AutoGrid
binary. :class:`AutoDockForceField` reads atom parameters and free-energy
weights from ``data/AD4_parameters.dat`` (resolved via
:func:`waterkit.utils.misc.data_path`), pre-computes the full pairwise
interaction table, and scores the four AutoDock4 terms between two atom sets:

* van der Waals 12-6,
* hydrogen bonds 12-10 with an optional directional (angular) factor,
* electrostatics with a distance-dependent dielectric (Mehler & Solmajer),
* desolvation (Gaussian-weighted volume/charge term).

Optional custom non-bonded pairs (``nbp_r_eps``) can be layered on top from an
AutoGrid ``.gpf`` file via :meth:`AutoDockForceField.load_nbp_r_eps_from_gpf`.

The pure-Python grid generator and the water sampler use this class as their
energy model, so every public method the upstream class exposed is preserved
with its original name and signature. The numerical behaviour mirrors the
original line for line; the only changes are typing, docstrings, logging, and a
swap of ``waterkit.utils`` geometry helpers for
:mod:`waterkit.utils.geometry`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import geometry
from ..utils.exceptions import ForceFieldError
from ..utils.misc import data_path
from ..utils.typing import Float, Point3, Scalars, Shaped

logger = logging.getLogger("waterkit.forcefield")


class AutoDockForceField:
    """AutoDock4 scoring function evaluated over atom-pair interactions.

    The constructor loads the free-energy coefficients and per-type atom
    parameters from ``parameter_file`` and builds :attr:`pairwise`, a pandas
    table indexed by ``(type_i, type_j)`` holding the pre-computed van der Waals
    and hydrogen-bond coefficients for every ordered pair of atom types.

    Attributes:
        weights: Free-energy coefficients keyed by term name (``vdW``,
            ``hbond``, ``estat``, ``desolv``, ``tors``).
        atom_par: Per-type atom parameters indexed by AutoDock atom type.
        pairwise: Pre-computed pairwise interaction table indexed by
            ``(i, j)`` atom types.
    """

    def __init__(
        self,
        parameter_file: str | Path | None = None,
        dielectric: float = -0.1465,
        smooth: float = 0.5,
        hb_cutoff: float = 8,
        elec_cutoff: float = 20,
        weighted: bool = True,
    ) -> None:
        """Initialize the force field from an AutoDock parameter file.

        Args:
            parameter_file: Path to an ``AD4_parameters.dat``-style file. When
                ``None`` (the default), the packaged
                ``data/AD4_parameters.dat`` is used.
            dielectric: Dielectric constant. A negative value selects the
                distance-dependent dielectric of Mehler & Solmajer; a positive
                value selects a constant dielectric.
            smooth: AutoDock energy-smoothing window (Angstrom). ``0`` disables
                smoothing.
            hb_cutoff: Distance cutoff (Angstrom) beyond which the hydrogen-bond
                term is zero.
            elec_cutoff: Distance cutoff (Angstrom) beyond which the
                electrostatic term is zero.
            weighted: If ``True`` use the free-energy coefficients from the
                parameter file; otherwise set every weight to ``1``.

        Raises:
            ForceFieldError: If the parameter file does not exist or contains no
                ``atom_par`` records.
        """
        if parameter_file is None:
            parameter_file = data_path("AD4_parameters.dat")
        parameter_file = Path(parameter_file)

        if not parameter_file.is_file():
            raise ForceFieldError(f"Parameter file {parameter_file} does not exist.")

        self.weights = self._set_weights(parameter_file, weighted)
        self.atom_par = self._set_atom_parameters(parameter_file)
        self.pairwise = self._build_pairwise_table()

        # Parameters
        self.hb_cutoff = hb_cutoff
        self.elec_cutoff = elec_cutoff

        # VdW and hydrogen bond
        self.smooth = smooth

        # Desolvation constants
        self.desolvation_k = 0.01097

        # Dielectric constants
        self.dielectric = dielectric
        self.dielectric_epsilon = 78.4
        self.dielectric_A = -8.5525
        self.dielectric_B = self.dielectric_epsilon - self.dielectric_A
        self.dielectric_lambda = 0.003627
        self.dielectric_k = 7.7839
        self.elec_scale = 332.06363

    def _set_weights(self, parameter_file: str | Path, weighted: bool = True) -> dict[str, float]:
        """Read the ``FE_coeff_*`` free-energy coefficients from the parameter file.

        Args:
            parameter_file: Path to the AutoDock parameter file.
            weighted: If ``True`` use the file values; otherwise set each to ``1``.

        Returns:
            A mapping from term name (e.g. ``vdW``, ``hbond``) to its weight.
        """
        weights: dict[str, float] = {}

        with open(parameter_file) as f:
            for line in f.readlines():
                if re.search("^FE_coeff", line):
                    fields = line.split()
                    weight_name = fields[0].split("_")[2]

                    if weighted:
                        weight_value = float(fields[-1])
                    else:
                        weight_value = 1.0

                    weights[weight_name] = weight_value

        return weights

    def _set_atom_parameters(self, parameter_file: str | Path) -> pd.DataFrame:
        """Read the ``atom_par`` records into a per-type parameter table.

        Args:
            parameter_file: Path to the AutoDock parameter file.

        Returns:
            A :class:`pandas.DataFrame` indexed by AutoDock atom type with the
            van der Waals, hydrogen-bond, volume, solvation, and bookkeeping
            columns.

        Raises:
            ForceFieldError: If the file contains no ``atom_par`` records.
        """
        columns = [
            "type",
            "rii",
            "epsii",
            "vol",
            "solpar",
            "rij_hb",
            "epsij_hb",
            "hbond",
            "rec_index",
            "map_index",
            "bond_index",
        ]
        data = []

        with open(parameter_file) as f:
            for line in f.readlines():
                if re.search("^atom_par", line):
                    fields = line.split()
                    data.append(
                        (
                            fields[1],
                            float(fields[2]),
                            float(fields[3]),
                            float(fields[4]),
                            float(fields[5]),
                            float(fields[6]),
                            float(fields[7]),
                            int(fields[8]),
                            int(fields[9]),
                            int(fields[10]),
                            int(fields[11]),
                        )
                    )

        if not data:
            raise ForceFieldError(f"No atom_par records found in parameter file {parameter_file}.")

        atom_par = pd.DataFrame(data=data, columns=columns)
        atom_par.set_index("type", inplace=True)
        return atom_par

    def load_nbp_r_eps_from_gpf(self, gpf_file: str | Path) -> None:
        """Load custom non-bonded pairs (``nbp_r_eps``) from a ``.gpf`` file.

        Each matching ``nbp_r_eps`` line overrides the interaction between an
        atom-type pair with a custom ``n``-``m`` potential. The ``cn`` and ``cm``
        coefficients are derived from the requested equilibrium distance and
        well depth, then written into both ``(i, j)`` and ``(j, i)`` rows of
        :attr:`pairwise` with status ``"nbp"``.

        Args:
            gpf_file: Path to the AutoGrid grid-parameter file.
        """
        columns_1 = ["vdw_rij", "A", "B", "hb_rij", "C", "D"]
        columns_2 = ["nbp_ij", "cn", "cm", "n", "m", "statut"]
        added = []

        with open(gpf_file) as f:
            lines = f.readlines()

            for line in lines:
                if re.search("^nbp_r_eps", line):
                    sline = line.split()
                    req = float(sline[1])
                    eps = float(sline[2])
                    n = int(sline[3])
                    m = int(sline[4])
                    i = sline[5]
                    j = sline[6]

                    cn = (m / (n - m)) * (eps * req**n)
                    cm = (n / (n - m)) * (eps * req**m)

                    data = [req, cn, cm, n, m, "nbp"]

                    if (i, j) in self.pairwise.index:
                        columns = columns_2
                    else:
                        columns = columns_1 + columns_2
                        data = [0] * 6 + data

                    self.pairwise.loc[(i, j), columns] = data
                    self.pairwise.loc[(j, i), columns] = data

                    added = True

        if added:
            self.pairwise.sort_index(inplace=True)

    def deactivate_pairs(self, pairs: list[list[str]]) -> None:
        """Deactivate the pairwise interaction between the given atom-type pairs.

        Args:
            pairs: An iterable of two-element sequences ``[type_i, type_j]``.
                The status of both ``(i, j)`` and ``(j, i)`` is set to
                ``"inactive"``.

        Raises:
            ForceFieldError: If ``pairs`` is not a sequence of 2-element
                sequences, or if a requested pair does not exist.
        """
        if not all(isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in pairs):
            raise ForceFieldError("pairs argument must be a list of list(2).")

        for pair in pairs:
            # Upstream relied on the .loc assignment raising for a missing pair;
            # modern pandas instead enlarges the table, so check membership
            # explicitly to preserve the original "pair does not exist" guard.
            if (pair[0], pair[1]) not in self.pairwise.index:
                raise ForceFieldError(f"pair {pair[0]} - {pair[1]} does not exist.")

            self.pairwise.loc[(pair[0], pair[1]), "statut"] = "inactive"
            self.pairwise.loc[(pair[1], pair[0]), "statut"] = "inactive"

    def _coefficient(self, epsilon: float, reqm: float, a: float, b: float) -> float:
        """Return an AutoDock 12-6 / 12-10 coefficient.

        Args:
            epsilon: Well depth.
            reqm: Equilibrium distance.
            a: Exponent paired with the prefactor.
            b: Exponent applied to ``reqm``.

        Returns:
            ``(a / |a - b|) * epsilon * reqm**b``.
        """
        return (a / np.abs(a - b)) * epsilon * reqm**b

    def _build_pairwise_table(self) -> pd.DataFrame:
        """Pre-compute the interaction coefficients for every atom-type pair.

        Returns:
            A :class:`pandas.DataFrame` indexed by ``(i, j)`` atom types with the
            van der Waals (``A``, ``B``), hydrogen-bond (``C``, ``D``), and
            (empty) custom non-bonded columns, plus an ``"active"`` status.
        """
        columns = [
            "i",
            "j",  # atom types
            "vdw_rij",
            "A",
            "B",  # vdw parameters
            "hb_rij",
            "C",
            "D",  # hydrogen bond parameters
            "nbp_ij",
            "cn",
            "cm",
            "n",
            "m",  # nbp parameters
            "statut",
        ]
        data = []

        for i in self.atom_par.index:
            for j in self.atom_par.index:
                # vdW
                vdw_rij = (self.atom_par.loc[i]["rii"] + self.atom_par.loc[j]["rii"]) / 2.0
                vdw_epsij = np.sqrt(self.atom_par.loc[i]["epsii"] * self.atom_par.loc[j]["epsii"])
                a = self._coefficient(vdw_epsij, vdw_rij, 6, 12)
                b = self._coefficient(vdw_epsij, vdw_rij, 12, 6)

                # Hydrogen bond
                hbond_i = self.atom_par.loc[i]["hbond"]
                hbond_j = self.atom_par.loc[j]["hbond"]

                if hbond_i in [1, 2] and hbond_j in [3, 4, 5]:
                    hb_rij = self.atom_par.loc[j]["rij_hb"]
                    hb_epsij = self.atom_par.loc[j]["epsij_hb"]
                elif hbond_i in [3, 4, 5] and hbond_j in [1, 2]:
                    hb_rij = self.atom_par.loc[i]["rij_hb"]
                    hb_epsij = self.atom_par.loc[i]["epsij_hb"]
                else:
                    hb_rij = 0
                    hb_epsij = 0

                c = self._coefficient(hb_epsij, hb_rij, 10, 12)
                d = self._coefficient(hb_epsij, hb_rij, 12, 10)

                # The five trailing zeros are the custom non-bonded-pair columns
                # (nbp_ij, cn, cm, n, m). They are initialised as floats (rather
                # than the upstream ints) so that load_nbp_r_eps_from_gpf can
                # write fractional equilibrium distances / coefficients into them
                # under modern pandas without a dtype-cast error. This does not
                # change any computed energy: n and m are only ever used as
                # exponents.
                data.append((i, j, vdw_rij, a, b, hb_rij, c, d, 0.0, 0.0, 0.0, 0.0, 0.0, "active"))

        pairwise = pd.DataFrame(data=data, columns=columns)
        pairwise.set_index(["i", "j"], inplace=True)
        pairwise.sort_index(inplace=True)

        return pairwise

    def smooth_distance(self, r: Scalars, reqm: float, smooth: float = 0.5) -> Scalars:
        """Apply AutoDock distance smoothing around the equilibrium distance.

        Distances within half the smoothing window of ``reqm`` are snapped to
        ``reqm``; distances outside are shifted toward it by half the window.

        Args:
            r: 1D array of pairwise distances (Angstrom).
            reqm: Equilibrium distance.
            smooth: Smoothing window width.

        Returns:
            A copy of ``r`` with smoothing applied (same shape as ``r``).
        """
        # Otherwise we changed r in-place.
        r = r.copy()
        sf = 0.5 * smooth
        r[((reqm - sf) < r) & (r < (reqm + sf))] = reqm
        r[r >= reqm + sf] -= sf
        r[r <= reqm - sf] += sf
        return r

    def nbp_r_eps(self, r: Scalars, reqm: float, cn: float, cm: float, n: float, m: float) -> float:
        """Compute a custom ``n``-``m`` non-bonded pair interaction.

        Args:
            r: 1D array of pairwise distances (Angstrom).
            reqm: Equilibrium distance used for smoothing.
            cn: Coefficient of the ``1 / r**n`` term.
            cm: Coefficient of the ``1 / r**m`` term.
            n: Repulsive exponent.
            m: Attractive exponent.

        Returns:
            The summed non-bonded pair energy.
        """
        if self.smooth > 0:
            r = self.smooth_distance(r, reqm, self.smooth)
        return np.sum((cn / r**n) - (cm / r**m))

    def van_der_waals(self, r: Scalars, reqm: float, A: float, B: float) -> float:
        """Compute the van der Waals 12-6 interaction.

        Args:
            r: 1D array of pairwise distances (Angstrom).
            reqm: Equilibrium distance used for smoothing.
            A: Coefficient of the ``1 / r**12`` term.
            B: Coefficient of the ``1 / r**6`` term.

        Returns:
            The summed van der Waals energy.
        """
        if self.smooth > 0:
            r = self.smooth_distance(r, reqm, self.smooth)
        return np.sum((A / r**12) - (B / r**6))

    def hydrogen_bond_distance(self, r: Scalars, reqm: float, C: float, D: float) -> float:
        """Compute the distance part of the hydrogen-bond 12-10 interaction.

        Args:
            r: 1D array of pairwise distances (Angstrom).
            reqm: Equilibrium distance used for smoothing.
            C: Coefficient of the ``1 / r**12`` term.
            D: Coefficient of the ``1 / r**10`` term.

        Returns:
            The summed hydrogen-bond distance energy, or ``0`` beyond
            :attr:`hb_cutoff`.
        """
        if r <= self.hb_cutoff:
            if self.smooth > 0:
                r = self.smooth_distance(r, reqm, self.smooth)
            return np.sum((C / r**12) - (D / r**10))
        else:
            return 0.0

    def hydrogen_bond_angle(
        self,
        atom_i_xyz: Point3,
        atom_j_xyz: Point3,
        vector_i_xyz: Point3,
        vector_j_xyz: Point3,
    ) -> float:
        """Compute the angular (directional) factor of a hydrogen bond.

        The factor is the product of the cosines of the two donor/acceptor
        vector angles, with each contribution clamped to ``0`` once the
        corresponding angle reaches ``pi / 2``.

        Args:
            atom_i_xyz: Coordinates of atom ``i``.
            atom_j_xyz: Coordinates of atom ``j``.
            vector_i_xyz: Hydrogen-bond vector position attached to atom ``i``.
            vector_j_xyz: Hydrogen-bond vector position attached to atom ``j``.

        Returns:
            The angular weighting factor in ``[0, 1]``.
        """
        beta_1 = geometry.get_angle(atom_i_xyz, atom_j_xyz, vector_j_xyz, False)[0]
        beta_2 = geometry.get_angle(atom_j_xyz, atom_i_xyz, vector_i_xyz, False)[0]

        angles = np.array([beta_1, beta_2])
        scores = np.cos(angles)
        scores[angles >= np.pi / 2.0] = 0.0
        score = np.prod(scores)

        return score

    def distance_dependent_dielectric(self, r: Scalars) -> Scalars:
        """Return the Mehler & Solmajer distance-dependent dielectric.

        Args:
            r: 1D array of pairwise distances (Angstrom).

        Returns:
            The dielectric value(s) at ``r`` (same shape as ``r``).
        """
        lambda_B = -self.dielectric_lambda * self.dielectric_B
        return self.dielectric_A + self.dielectric_B / (
            1.0 + self.dielectric_k * np.exp(lambda_B * r)
        )

    def electrostatic(self, r: Scalars, qi: float, qj: float) -> float:
        """Compute the electrostatic interaction energy.

        Uses the distance-dependent dielectric when :attr:`dielectric` is
        negative, and a constant dielectric otherwise.

        Args:
            r: 1D array of pairwise distances (Angstrom).
            qi: Partial charge of atom ``i``.
            qj: Partial charge of atom ``j``.

        Returns:
            The summed electrostatic energy, or ``0`` beyond :attr:`elec_cutoff`.
        """
        if r <= self.elec_cutoff:
            if self.dielectric < 0:
                r_ddd = self.distance_dependent_dielectric(r)
                return self.elec_scale * np.sum(qi * qj / (r * r_ddd))
            else:
                return self.elec_scale * np.sum(qi * qj / (r * self.dielectric))
        else:
            return 0.0

    def desolvation(
        self,
        r: Scalars,
        qi: float,
        qj: float,
        ai: float,
        aj: float,
        vi: float,
        vj: float,
        sigma: float = 3.6,
    ) -> float:
        """Compute the Gaussian-weighted desolvation interaction energy.

        Args:
            r: 1D array of pairwise distances (Angstrom).
            qi: Partial charge of atom ``i``.
            qj: Partial charge of atom ``j``.
            ai: Atomic solvation parameter of atom ``i``.
            aj: Atomic solvation parameter of atom ``j``.
            vi: Atomic volume of atom ``i``.
            vj: Atomic volume of atom ``j``.
            sigma: Gaussian width.

        Returns:
            The summed desolvation energy.
        """
        desolv = (ai * vj) + (aj * vi)
        desolv += (self.desolvation_k * np.abs(qi) * vj) + (self.desolvation_k * np.abs(qj) * vi)
        desolv *= np.exp(-0.5 * (r**2 / sigma**2))
        return np.sum(desolv)

    def intermolecular_energy(
        self,
        atoms_i: Shaped[np.ndarray, "n"],  # structured ATOM_DTYPE record array
        atoms_j: Shaped[np.ndarray, "m"],  # structured ATOM_DTYPE record array
        hbs_i: pd.DataFrame | None = None,
        hbs_j: pd.DataFrame | None = None,
        details: bool = False,
    ) -> float | Float[np.ndarray, "6"]:
        """Compute the total AutoDock4 interaction energy between two atom sets.

        Each atom record must expose ``t`` (AutoDock type), ``xyz`` (coordinates)
        and ``q`` (partial charge); when ``hbs_i``/``hbs_j`` are supplied the
        records must additionally expose ``atom_i`` for matching against the
        hydrogen-bond tables (mirroring the upstream layout). Terms whose weight
        is zero are skipped.

        Args:
            atoms_i: Structured ``ATOM_DTYPE`` record array (1D, ``n`` atoms) for
                the first set (e.g. from
                :meth:`waterkit.core.molecule.Molecule.atom_informations`).
            atoms_j: Structured ``ATOM_DTYPE`` record array (1D, ``m`` atoms) for
                the second set.
            hbs_i: Optional hydrogen-bond table for ``atoms_i`` providing
                ``atom_i`` and ``vector_xyz`` columns; enables the directional
                hydrogen-bond term.
            hbs_j: Optional hydrogen-bond table for ``atoms_j``.
            details: If ``True`` return the per-term breakdown.

        Returns:
            The total interaction energy, or, when ``details`` is ``True``, the
            array ``[total, hb, vdw, elec, desolv, nbp]``.
        """
        desolv = 0.0
        elec = 0.0
        hb = 0.0
        total = 0.0
        vdw = 0.0
        nbp = 0.0

        for atom_i in atoms_i:
            # Get HB vectors from atom i
            if hbs_i is not None and hbs_j is not None:
                hb_i = hbs_i.loc[hbs_i["atom_i"] == atom_i["atom_i"]]

            for atom_j in atoms_j:
                pairwise = self.pairwise.loc[(str(atom_i["t"]), str(atom_j["t"]))]

                if pairwise["statut"] == "active":
                    r = geometry.get_euclidean_distance(
                        np.array(atom_i["xyz"]), np.array([atom_j["xyz"]])
                    )

                    # We do not want to calculate useless things
                    # if the weight is equal to zero.
                    if self.weights["hbond"] > 0 and pairwise["hb_rij"] > 0:
                        hb_distance = self.hydrogen_bond_distance(
                            r, pairwise["hb_rij"], pairwise["C"], pairwise["D"]
                        )

                        # Get HB vectors from atom j
                        # Calculate directional HB if HB vectors provided
                        if hbs_i is not None and hbs_j is not None:
                            hb_j = hbs_j.loc[hbs_j["atom_i"] == atom_j["atom_i"]]

                            hb_angles = []

                            for h_i in hb_i.iterrows():
                                for h_j in hb_j.iterrows():
                                    hb_angle = self.hydrogen_bond_angle(
                                        atom_i["xyz"],
                                        atom_j["xyz"],
                                        h_i["vector_xyz"],
                                        h_j["vector_xyz"],
                                    )
                                    hb_angles.append(hb_angle)

                            hb += np.sum(hb_distance * np.array(hb_angles))
                        else:
                            hb += hb_distance

                    if self.weights["vdW"] > 0:
                        vdw += self.van_der_waals(
                            r, pairwise["vdw_rij"], pairwise["A"], pairwise["B"]
                        )

                    if self.weights["estat"] > 0:
                        elec += self.electrostatic(r, atom_i["q"], atom_j["q"])

                    if self.weights["desolv"] > 0:
                        desolv += self.desolvation(
                            r,
                            atom_i["q"],
                            atom_j["q"],
                            self.atom_par.loc[atom_i["t"]]["solpar"],
                            self.atom_par.loc[atom_j["t"]]["solpar"],
                            self.atom_par.loc[atom_i["t"]]["vol"],
                            self.atom_par.loc[atom_j["t"]]["vol"],
                        )

                elif pairwise["statut"] == "nbp":
                    r = geometry.get_euclidean_distance(
                        np.array(atom_i["xyz"]), np.array([atom_j["xyz"]])
                    )
                    nbp += self.nbp_r_eps(
                        r,
                        pairwise["nbp_ij"],
                        pairwise["cn"],
                        pairwise["cm"],
                        pairwise["n"],
                        pairwise["m"],
                    )

        hb *= self.weights["hbond"]
        elec *= self.weights["estat"]
        vdw *= self.weights["vdW"]
        desolv *= self.weights["desolv"]
        total = hb + vdw + elec + desolv + nbp

        if details:
            return np.array([total, hb, vdw, elec, desolv, nbp])
        else:
            return total
