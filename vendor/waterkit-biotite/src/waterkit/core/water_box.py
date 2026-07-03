"""WaterBox: receptor + accumulated water shells with fast spatial lookup.

Faithful port of ``waterkit.water_box``. The :class:`WaterBox` owns the molecule
registry, the per-shell bookkeeping, the inter-molecule connection table, and the
``scipy.spatial.cKDTree`` used to find hydrogen-bond anchors near a candidate
water. :meth:`WaterBox.build_next_shell` drives one hydration layer by delegating
the placement / optimization to :class:`~waterkit.core.sampling.WaterSampler`.

This is a numerically identical port of the original implementation; the only
changes are cosmetic (the unused ``import openbabel`` was dropped, ``utils.X``
calls now resolve through :mod:`waterkit.utils.geometry`, and a couple of
column-name references in :meth:`closest_hydrogen_bond_anchor` were reconciled
against the perception DataFrames — see the module notes below).

Notes:
    The hydrogen-bond DataFrames produced by the perception layer (and by
    :class:`~waterkit.core.water.Water`) use the columns ``atom_i``,
    ``vector_xyz``, ``anchor_type`` and ``anchor_name``. The upstream
    ``closest_hydrogen_bond_anchor`` referred to ``hba.vectors`` / ``hba.type``,
    which never existed as columns; those references are mapped here to
    ``vector_xyz`` / ``anchor_type``. (This method is not used internally — only
    :meth:`closest_atoms` is — so the change has no effect on the sampling
    trajectory.)
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from string import ascii_uppercase

import numpy as np
import pandas as pd
from scipy import spatial

from ..utils import geometry
from ..utils.typing import Vec3
from .grid import Map
from .molecule import Molecule
from .sampling import WaterSampler
from .water import Water

logger = logging.getLogger("waterkit.water_box")


class WaterBox:
    """Holds the receptor and successive hydration shells of water molecules."""

    def __init__(
        self,
        receptor: Molecule,
        ad_map: Map,
        temperature: float = 300.0,
        water_model: str = "tip3p",
        spherical_water_map: str | None = None,
    ) -> None:
        """Initialize the water box around a receptor.

        Args:
            receptor: The receptor :class:`~waterkit.core.molecule.Molecule`.
            ad_map: The AutoDock affinity :class:`~waterkit.core.grid.Map`
                (copied, so the original is left untouched).
            temperature: Sampling temperature in Kelvin (default: ``300.``).
            water_model: Water model used to build explicit waters
                (default: ``"tip3p"``).
            spherical_water_map: Optional path to a spherical-water map used by
                the sampler (default: ``None``).
        """
        self.df: dict[str, pd.DataFrame] = {}
        self.molecules: dict[int, Molecule] = {}
        self.map: Map | None = None
        self._kdtree: spatial.cKDTree | None = None
        self._water_model = water_model

        # All the informations are stored into a dict of df
        columns = ["molecule_i", "atom_i", "molecule_j", "atom_j"]
        self.df["connections"] = pd.DataFrame(columns=columns)
        columns = ["shell_id"]
        self.df["shells"] = pd.DataFrame(columns=columns)
        columns = ["molecule_i", "atom_i"]
        self.df["kdtree_relations"] = pd.DataFrame(columns=columns)

        self.map = ad_map.copy()
        self._add_receptor(receptor)

        # Forcefields, forcefield parameters and water model
        self._temperature = temperature
        # Initialize the sampling method
        self._wopt = WaterSampler(
            self, temperature=self._temperature, spherical_water_map=spherical_water_map
        )

    def copy(self) -> WaterBox:
        """Return a deep copy of the water box (one per trajectory frame).

        Returns:
            An independent deep copy of this :class:`WaterBox`, including its
            molecules, DataFrames and KDTree.
        """
        return copy.deepcopy(self)

    def _add_receptor(self, receptor: Molecule) -> bool:
        """Add the receptor and the corresponding ad_map to the waterbox.

        Args:
            receptor: The receptor :class:`~waterkit.core.molecule.Molecule`.

        Returns:
            ``True`` if the receptor was added, ``False`` if it was already
            present.
        """
        if 0 not in self.molecules:
            # Add the receptor to the waterbox
            self._add_molecules(receptor)
            # Add informations about the receptor
            data = pd.DataFrame([[0]], columns=["shell_id"])
            self._add_informations(data, "shells")

            return True
        else:
            # The receptor was already added to the waterbox
            return False

    def _add_molecules(
        self,
        molecules: Molecule | list[Molecule] | tuple[Molecule, ...],
        connections: pd.DataFrame | None = None,
        add_KDTree: bool = True,
    ) -> None:
        """Add a new molecule to the waterbox.

        Args:
            molecules: A molecule or a list/tuple of molecules to register.
            connections: Optional connection table to append (default: ``None``).
            add_KDTree: Whether to add the new atoms to the KDTree
                (default: ``True``).
        """
        if not isinstance(molecules, (list, tuple)):
            molecules = [molecules]

        molecule_ids = list(self.molecules.keys())

        try:
            last_key = np.max(molecule_ids)
        except Exception:
            # We initliaze at -1, make first molecule at index 0
            last_key = -1

        # Add molecules to the dictionary
        new_keys = range(last_key + 1, len(molecules) + last_key + 1)
        d = {key: molecule for key, molecule in zip(new_keys, molecules)}
        self.molecules.update(d)

        if connections is not None:
            self._add_connections(connections)
        if add_KDTree:
            self._add_molecules_to_kdtree(molecules)

    def _add_connections(self, connections: pd.DataFrame) -> None:
        """Add connections between molecules.

        Args:
            connections: A DataFrame with columns ``molecule_i``, ``atom_i``,
                ``molecule_j`` and ``atom_j`` whose molecule indices are relative
                and shifted to absolute indices here.
        """
        try:
            last_connections = self.df["connections"].tail(1)
            last_molecule_i = last_connections["molecule_i"].values[0]
            last_molecule_j = last_connections["molecule_j"].values[0]
        except Exception:
            last_molecule_i = -1
            last_molecule_j = 0

        connections["molecule_i"] += last_molecule_i + 1
        connections["molecule_j"] += last_molecule_j + 1
        self._add_informations(connections, "connections")

    def _add_molecules_to_kdtree(
        self, molecules: Molecule | list[Molecule] | tuple[Molecule, ...]
    ) -> None:
        """Build or update the cKDTree of all the atoms in the water box.

        The tree allows quick nearest-neighbor lookup of atoms around a point.

        Args:
            molecules: A molecule or a list/tuple of molecules whose atoms are
                added to the KDTree.
        """
        if not isinstance(molecules, (list, tuple)):
            molecules = [molecules]

        try:
            last_kdtree_relations = self.df["kdtree_relations"].tail(1)
            last_molecule_i = last_kdtree_relations["molecule_i"].values[0]
        except Exception:
            # We initliaze at -1, make first molecule at index 0
            last_molecule_i = -1

        data = []
        relations = []

        for index, molecule in enumerate(molecules):
            coordinates = molecule.coordinates()
            mol_i = index + last_molecule_i + 1
            relations.append([[mol_i, i + 1] for i in range(coordinates.shape[0])])
            data.append(coordinates)

        # Update the KDTree relation database
        columns = ["molecule_i", "atom_i"]
        relations = np.vstack(relations)
        relations = pd.DataFrame(relations, columns=columns)
        self._add_informations(relations, "kdtree_relations")

        # Update the KDTree
        data = np.vstack(data)
        try:
            data = np.concatenate((self._kdtree.data, data))
        except Exception:
            pass
        self._kdtree = spatial.cKDTree(data)

    def _add_informations(self, data: pd.DataFrame, where: str) -> None:
        """Append a DataFrame to the existing information DataFrame.

        Args:
            data: The DataFrame to append.
            where: The key of the target DataFrame in :attr:`df`.
        """
        try:
            self.df[where] = pd.concat([self.df[where], data], ignore_index=True)
            self.df[where].reset_index(drop=True, inplace=True)
        except Exception:
            logger.error("Cannot add informations to %s dataframe.", where)

    def _update_informations_in_shell(self, data: object, shell_id: int, key: str) -> None:
        """Update shell information.

        Args:
            data: The value(s) to store for the matching shell(s). Assigned
                directly into the ``shells`` DataFrame, so it may be a scalar
                (broadcast to all matching rows) or an array-like of one value
                per matching row.
            shell_id: The id of the shell to update.
            key: The column to update in the ``shells`` DataFrame.
        """
        index = self.df["shells"]["shell_id"] == shell_id
        self.df["shells"].loc[index, key] = data

    def number_of_shells(self) -> int:
        """Total number of shells in the WaterBox.

        Returns:
            int: number of shells

        """
        # df["column"].max() faster than np.max(df["column"])
        return self.df["shells"]["shell_id"].max()

    def molecules_in_shell(
        self, shell_ids: int | list[int] | tuple[int, ...] | None = None
    ) -> list[Molecule]:
        """Get all the molecule in shell.

        Args:
            shell_ids (list): ids of the shell(s) (default: None)

        Returns:
            list: list of all the molecules in the selected shell(s)

        """
        if shell_ids is not None:
            if not isinstance(shell_ids, (list, tuple)):
                shell_ids = [shell_ids]
            df = self.df["shells"][self.df["shells"]["shell_id"].isin(shell_ids)]
        else:
            df = self.df["shells"]

        molecules = [self.molecules[i] for i in df.index.tolist()]

        return molecules

    def molecule_informations_in_shell(self, shell_id: int) -> pd.DataFrame:
        """Get information of shell.

        Args:
            shell_id (int): id the shell

        Returns:
            DataFrame: informations concerning the shell

        """
        df = self.df["shells"]
        # Return a copy to avoid a SettingWithCopyWarning flag
        return df.loc[df["shell_id"] == shell_id].copy()

    def closest_atoms(
        self, xyz: Vec3, radius: float, exclude: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Retrieve indices of the closest atoms around a point at a certain radius.

        Args:
            xyz: A single query point ``(x, y, z)`` of shape ``(3,)``.
            radius: Search radius in Angstrom.
            exclude: Molecules/atoms to exclude (columns: ``molecule_i`` and
                ``atom_i``). Default ``None``.

        Returns:
            A DataFrame (columns ``molecule_i``, ``atom_i``) of all the
            molecules/atoms within ``radius`` Angstrom of ``xyz``.
        """
        if self._kdtree is None:
            logger.warning("KDTree is empty.")
            return pd.DataFrame(columns=["molecule_i", "atom_i"])

        index = self._kdtree.query_ball_point(xyz, radius, p=2)
        df = self.df["kdtree_relations"].loc[index]

        if isinstance(exclude, pd.DataFrame):
            df = df.merge(exclude, indicator=True, how="outer")
            df = df[df["_merge"] == "left_only"]
            df.drop(columns="_merge", inplace=True)

        return df

    def closest_hydrogen_bond_anchor(
        self, xyz: Vec3, radius: float, exclude: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame | None, int | None]:
        """Find the closest hydrogen bond anchors.

        Args:
            xyz: A single query point ``(x, y, z)`` of shape ``(3,)``.
            radius: Maximum search radius in Angstrom.
            exclude: Molecules/atoms to exclude (columns: ``molecule_i`` and
                ``atom_i``). Default ``None``.

        Returns:
            A ``(best_hba, best_hbv_id)`` tuple where ``best_hba`` is the
            single-row hydrogen-bond DataFrame of the closest anchor (or
            ``None`` if none was found) and ``best_hbv_id`` is the index of the
            closest hydrogen-bond vector for that anchor (or ``None``).
        """
        best_hba = None
        best_hbv_id = None
        best_hbv_distance = np.inf

        df = self.closest_atoms(xyz, radius, exclude)

        if not df.empty:
            for index, row in df.iterrows():
                try:
                    hydrogen_bonds = self.molecules[row["molecule_i"]].hydrogen_bonds
                    hba = hydrogen_bonds.loc[hydrogen_bonds["atom_i"] == row["atom_i"]]
                    hba_xyz = self.molecules[row["molecule_i"]].coordinates(row["atom_i"])

                    hba_distance = geometry.get_euclidean_distance(xyz, hba_xyz)[0]
                    hbv_distances = geometry.get_euclidean_distance(xyz, hba.vector_xyz)
                    hbv_min_distance = np.min(hbv_distances)
                    hbv_min_id = np.argmin(hbv_distances)

                    # We add 1 A to interpolate the distance to the heavy atom
                    if hba.anchor_type == "donor":
                        hba_distance += 1.0

                    # Select the closest HBV and make sure that the heavy atom is close enough
                    if hbv_min_distance < best_hbv_distance and hba_distance <= radius:
                        best_hba = hba
                        best_hbv_id = hbv_min_id
                        best_hbv_distance = hbv_min_distance
                except KeyError:
                    continue

        return best_hba, best_hbv_id

    def _place_optimal_spherical_waters(
        self,
        molecules: list[Molecule],
        atom_type: str = "W",
        partial_charge: float = 0.0,
    ) -> tuple[list[Water], pd.DataFrame]:
        """Place spherical water molecules in the optimal position.

        Args:
            molecules (Molecule): molecules on which spherical water molecules will be placed
            atom_type (str): atom type of the spherical water molecules (default: OW)
            partial_charge (float): partial charge of the spherical waters (default: -0.834)

        Returns:
            list: list of water molecules
            DataFrame: contains connections between molecules

        """
        waters = []
        data = []

        for i, molecule in enumerate(molecules):
            for index, row in molecule.hydrogen_bonds.iterrows():
                # Add water molecule only if it's in the map
                anchor_xyz = molecule.coordinates(row.atom_i)[0]

                if self.map.is_in_map(anchor_xyz):
                    w = Water(
                        row.vector_xyz,
                        atom_type,
                        partial_charge,
                        anchor_xyz,
                        row.vector_xyz,
                        row.anchor_type,
                    )

                    waters.append(w)
                    data.append((i, row.atom_i, len(waters) - 1, None))

        # Convert list of tuples into dataframe
        columns = ["molecule_i", "atom_i", "molecule_j", "atom_j"]
        connections = pd.DataFrame(data, columns=columns)

        return (waters, connections)

    def build_next_shell(self) -> bool:
        """Build the next hydration shell.

        Returns:
            bool: True if water molecules were added or False otherwise

        """
        sw_type = "SW"
        partial_charge = 0.0
        shell_id = self.number_of_shells()
        molecules = self.molecules_in_shell(shell_id)

        # If multiple instances of WaterBox are running in parallel
        # we might have end up with the same result. So we force numpy to
        # generate different random numbers
        rand = int.from_bytes(os.urandom(4), sys.byteorder)
        np.random.seed(rand)

        waters, connections = self._place_optimal_spherical_waters(
            molecules, sw_type, partial_charge
        )

        # Only the receptor contains disordered hydrogens
        if shell_id == 0:
            waters, df = self._wopt.sample_grid(waters, connections, opt_disordered=True)
        else:
            # After the first hydration layer, we don't care anymore about
            # connections. It was only useful for the disordered hydrogen atoms.
            waters, df = self._wopt.sample_grid(waters, opt_disordered=False)

        if len(waters):
            self._add_molecules(waters, add_KDTree=False)
            # Add informations about the new shell
            if "shells" in df.keys():
                self._add_informations(df["shells"], "shells")

            return True
        else:
            return False

    def to_pdb(self, fname: str, water_model: str = "tip3p", include_receptor: bool = True) -> None:
        """Write all the content of the water box in a PDB file.

        We cannot use OpenBabel to write the PDB file of water molecules
        because it is always changing the atom types...

        Args:
            fname (str): name of the output file
            water_model (str): water model we want to use to export the water (default: tip3p)
            include_receptor (bool): include or exclude the receptor (default: True)

        Returns:
            None

        """
        i = 1
        j = 1
        output_str = ""
        pdb_str = "ATOM  %5d  %-4s%-3s%2s%4d    %8.3f%8.3f%8.3f  1.00  1.00          %2s\n"

        shell_id = self.number_of_shells()
        water_shells = [self.molecules_in_shell(i) for i in range(1, shell_id + 1)]

        if include_receptor:
            atoms = self.molecules[0].atoms
            for atom in atoms:
                x, y, z = atom["xyz"]
                output_str += pdb_str % (
                    atom["i"],
                    atom["name"],
                    atom["resname"],
                    "A",
                    atom["resnum"],
                    x,
                    y,
                    z,
                    atom["t"],
                )

            # Get the index of the nex atom and residue
            i = atom["i"] + 1
            j = atom["resnum"] + 1

        for water_shell, chain in zip(water_shells, ascii_uppercase[1:]):
            j = 1

            for water in water_shell:
                c = water.coordinates()

                output_str += pdb_str % (i, "O", "HOH", chain, j, c[0][0], c[0][1], c[0][2], "O")
                output_str += pdb_str % (
                    i + 1,
                    "H1",
                    "HOH",
                    chain,
                    j,
                    c[1][0],
                    c[1][1],
                    c[1][2],
                    "H",
                )
                output_str += pdb_str % (
                    i + 2,
                    "H2",
                    "HOH",
                    chain,
                    j,
                    c[2][0],
                    c[2][1],
                    c[2][2],
                    "H",
                )
                i += 2

                if water_model == "tip5p":
                    output_str += pdb_str % (
                        i + 3,
                        "L1",
                        "HOH",
                        chain,
                        j,
                        c[3][0],
                        c[3][1],
                        c[3][2],
                        "L",
                    )
                    output_str += pdb_str % (
                        i + 4,
                        "L2",
                        "HOH",
                        chain,
                        j,
                        c[4][0],
                        c[4][1],
                        c[4][2],
                        "L",
                    )
                    i += 2

                i += 1
                j += 1

        # ... but we add it again at the end
        output_str += "TER\n"

        with open(fname, "w") as w:
            w.write(output_str)
