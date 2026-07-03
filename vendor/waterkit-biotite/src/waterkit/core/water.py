"""Water molecule model (spherical, TIP3P, TIP5P).

Pure-geometry port of ``waterkit.water``. A :class:`Water` starts spherical (a
single oxygen pseudo-atom placed at a hydrogen-bond anchor) and is grown into an
explicit TIP3P or TIP5P geometry once its position is accepted by the sampler.

This is a faithful, numerically identical port of the original WaterKit
``Water`` class. The only behavioural change is :meth:`Water.from_file`, whose
OpenBabel-based reader has been replaced by the Biotite/PDBQT readers in
:mod:`waterkit.utils.io`; it extracts oxygen coordinates only (the original used
``OBAtom.IsOxygen``) and builds one spherical water per oxygen, exactly as
upstream. Atom records use :data:`waterkit.core.types.ATOM_DTYPE` and atom ids
are 1-based throughout.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd

from ..utils import geometry, io
from ..utils.typing import Indices, Vec3
from .molecule import Molecule
from .types import ATOM_DTYPE


class Water(Molecule):
    """A spherical or explicit (TIP3P/TIP5P) water molecule."""

    def __init__(
        self,
        xyz: Vec3,
        atom_type: str = "W",
        partial_charge: float = 0.0,
        hb_anchor: Vec3 | None = None,
        hb_vector: Vec3 | None = None,
        hb_type: str | None = None,
    ) -> None:
        """Initialize a Water object.

        Args:
            xyz: 3d coordinates of the oxygen water molecule, shape ``(3,)``.
            atom_type: Atom type of the spherical water molecule (default: ``W``).
            partial_charge: Partial charge of the spherical water molecule
                (default: ``0.``).
            hb_anchor: 3d coordinates of the HB anchor, shape ``(3,)``
                (default: ``[0, 0, 0]``).
            hb_vector: 3d coordinates of the HB vector, shape ``(3,)`` (default:
                the ``xyz`` argument, so the water is optimally placed on the HB
                anchor).
            hb_type: Type of the HB anchor (``acceptor`` or ``donor``).
        """
        # ``atoms`` is a structured ATOM_DTYPE record array, ``Shaped[np.ndarray,
        # "n"]`` (see core.types), or ``None`` until the oxygen atom is added.
        self.atoms = None
        self.hydrogen_bonds: pd.DataFrame | None = None
        self.rotatable_bonds: pd.DataFrame | None = None

        if hb_anchor is None:
            self.hb_anchor: Vec3 = np.array([0, 0, 0])
        else:
            self.hb_anchor = hb_anchor

        # If no hb vector is specified, the water molecule
        # is optimally placed on the hb anchor
        if hb_vector is None:
            self.hb_vector: Vec3 = xyz
        else:
            self.hb_vector = hb_vector

        if hb_type is None:
            self.hb_type: str = "acceptor"
        else:
            self.hb_type = hb_type

        # Add the oxygen atom
        self._add_atom(xyz, atom_type, partial_charge)

    @classmethod
    def from_file(
        cls, fname: str, atom_type: str = "W", partial_charge: float = 0.0
    ) -> list[Water]:
        """Create a list of spherical Water objects from a structure file.

        The water molecules created are spherical. Only the oxygen coordinates
        are needed, so this reader (replacing the original OpenBabel reader)
        extracts every oxygen atom and builds one spherical water per oxygen.
        PDBQT files are read with :func:`waterkit.utils.io.parse_pdbqt` (oxygen
        identified from the AutoDock atom type); other formats are read with
        :func:`waterkit.utils.io.load_structure` (oxygen identified from the
        Biotite element).

        Args:
            fname: Molecule filename.
            atom_type: Atom type of the spherical water molecules (default: ``W``).
            partial_charge: Partial charge of the spherical water molecules
                (default: ``0.``).

        Returns:
            List of :class:`Water` molecule objects.
        """
        waters = []

        # Get file extension
        file_extension = os.path.splitext(fname)[1].lower().lstrip(".")

        if file_extension == "pdbqt":
            rec = io.parse_pdbqt(fname)
            elements = [io.autodock_to_element(t) for t in rec["t"]]
            coordinates = rec["xyz"]
        else:
            atoms = io.load_structure(fname, add_hydrogens=False, perceive_bonds=False)
            elements = [str(e).strip().upper() for e in atoms.element]
            coordinates = atoms.coord

        for xyz, element in zip(coordinates, elements):
            if element == "O":
                waters.append(cls(np.array(xyz, dtype=float), atom_type, partial_charge))

        return waters

    def _add_atom(self, xyz: Vec3, atom_type: str, partial_charge: float) -> None:
        """Add an atom to the molecule.

        Args:
            xyz: 3d coordinates of the new atom, shape ``(3,)``.
            atom_type: AutoDock atom type of the new atom (the atom name is its
                first character).
            partial_charge: Partial charge of the new atom.
        """
        resname = "HOH"
        resnum = 1
        chain = ""

        if self.atoms is None:
            num_atoms = 0
        else:
            num_atoms = np.max(self.atoms["i"])

        new_atom = (
            num_atoms + 1,
            atom_type[0],
            resname,
            resnum,
            chain,
            xyz,
            partial_charge,
            atom_type,
        )
        new_atom = np.array(new_atom, dtype=ATOM_DTYPE)

        if self.atoms is not None:
            self.atoms = np.hstack((self.atoms, new_atom))
        else:
            self.atoms = new_atom

    def _delete_atoms(self, atom_ids: Indices) -> bool:
        """Delete atoms from the water.

        ``atom_ids`` are atom ids (1-based) and not array indices. Only the
        hydrogens and lone pairs are deleted, never the oxygen atom.

        Args:
            atom_ids: 1-based atom ids to delete, shape ``(n,)`` (any array-like,
                e.g. a list or ``range``, is accepted and converted internally).

        Returns:
            ``True`` if atoms were deleted, ``False`` otherwise.
        """
        if not isinstance(atom_ids, np.ndarray):
            atom_ids = np.array(atom_ids)

        # Make sure, we never delete the oxygen atom (1)
        atom_ids = np.delete(atom_ids, np.where(atom_ids == 1))

        if self.atoms.size > 1 and atom_ids.size > 0:
            # atoms_ids - 1, because the array is 0-based
            self.atoms = np.delete(self.atoms, atom_ids - 1)
            # From 1 to num_atom + 1, because atom ids are 1-based
            self.atoms["i"] = np.arange(1, self.atoms.shape[0] + 1)
            return True
        else:
            return False

    def is_water(self) -> bool:
        """Tell if it is a water or not.

        Returns:
            Always ``True`` for a :class:`Water`.
        """
        return True

    def is_spherical(self) -> bool:
        """Tell if water is spherical or not.

        Returns:
            ``True`` if the water has a single (oxygen) atom, ``False`` otherwise.
        """
        if self.atoms.size == 1:
            return True
        return False

    def is_tip3p(self) -> bool:
        """Tell if water is TIP3P or not.

        Returns:
            ``True`` if the water has 3 atoms (O + 2 H), ``False`` otherwise.
        """
        if self.atoms.size == 3:
            return True
        return False

    def is_tip5p(self) -> bool:
        """Tell if water is TIP5P or not.

        Returns:
            ``True`` if the water has 5 atoms (O + 2 H + 2 Lp), ``False``
            otherwise.
        """
        if self.atoms.size == 5:
            return True
        return False

    def tip3p(self) -> Water:
        """Return a TIP3P version of the Water (deepcopy).

        Returns:
            A TIP3P :class:`Water` when called on a TIP5P water, otherwise
            ``self``.
        """
        if self.is_tip5p():
            w = copy.deepcopy(self)
            # Atom ids are 1-based, and LP corresponds to atom 4 and 5
            w._delete_atoms([4, 5])
            # hbond df is 0-based, and so LP corresponds to index 2 and 3
            w.hydrogen_bonds.drop([2, 3], inplace=True)

            return w

        return self

    def build_explicit_water(self, water_model: str = "tip3p") -> bool:
        """Construct hydrogen atoms (H) and lone-pairs (Lp).

        TIP3P or TIP5P parameters: http://www1.lsbu.ac.uk/water/water_models.html

        Args:
            water_model: Water model (choice: ``tip3p`` or ``tip5p``)
                (default: ``tip3p``).

        Returns:
            ``True`` if successful, ``False`` otherwise.
        """
        models = {
            "tip3p": {"atom_types": ["OW", "HW", "HW"], "partial_charges": [-0.834, 0.417, 0.417]},
            "tip5p": {
                "atom_types": ["OT", "HT", "HT", "LP", "LP"],
                "partial_charges": [0.0, 0.241, 0.241, -0.241, -0.241],
            },
        }

        if water_model in models:
            atom_types = models[water_model]["atom_types"]
            partial_charges = models[water_model]["partial_charges"]
        else:
            print("Error: water model %s unknown." % water_model)
            return False

        if water_model == "tip3p":
            distances = [0.9572, 0.9572, 1.0]
            angles = [104.52, 127.74]
        elif water_model == "tip5p":
            distances = [0.9572, 0.9572, 0.7, 0.7]
            angles = [104.52, 109.47]

        # If donor, we started by building the lone-pairs first
        if self.hb_type == "donor":
            distances.reverse()
            angles.reverse()

        oxygen_xyz = self.coordinates(1)[0]

        # For both TIP3P and TIP5P
        # Vector between O and the Acceptor/Donor atom
        v = geometry.vector(oxygen_xyz, self.hb_anchor)
        v = geometry.normalize(v)
        # Compute a vector perpendicular to v
        p = oxygen_xyz + geometry.get_perpendicular_vector(v)
        # H/Lp between O and Acceptor/Donor atom
        a1_xyz = oxygen_xyz + (distances[0] * v)
        # Build the second H/Lp using the perpendicular vector p
        a2_xyz = geometry.rotate_point(a1_xyz, oxygen_xyz, p, np.radians(angles[0]))
        a2_xyz = geometry.resize_vector(a2_xyz, distances[1], oxygen_xyz)

        if water_model == "tip3p" and self.hb_type == "donor":
            # Build the second H/Lp using the perpendicular vector p
            a3_xyz = geometry.rotate_point(a2_xyz, oxygen_xyz, p, np.radians(angles[1]))
            a3_xyz = geometry.resize_vector(a3_xyz, distances[2], oxygen_xyz)
        elif water_model == "tip5p":
            # ... and rotate it to build the last H/Lp
            v = geometry.atom_to_move(oxygen_xyz, [a1_xyz, a2_xyz])
            r = oxygen_xyz + geometry.normalize(geometry.vector(a1_xyz, a2_xyz))
            a3_xyz = geometry.rotate_point(v, oxygen_xyz, r, np.radians(angles[1] / 2.0))
            a3_xyz = geometry.resize_vector(a3_xyz, distances[3], oxygen_xyz)
            a4_xyz = geometry.rotate_point(v, oxygen_xyz, r, -np.radians(angles[1] / 2.0))
            a4_xyz = geometry.resize_vector(a4_xyz, distances[3], oxygen_xyz)

        # Only now we do all the modifications to the atoms. We never know, we
        # might have an error before while calculating positions of the new
        # atoms. So no need to revert to the previous state if something wrong
        # is happening.

        # Change the type and partial charges of the oxygen atom
        # And remove any existing atoms (except oxygen of course)
        if self.is_spherical():
            self.atoms["q"] = partial_charges[0]
            self.atoms["t"] = atom_types[0]
        if not self.is_spherical():
            self.atoms[0]["q"] = partial_charges[0]
            self.atoms[0]["t"] = atom_types[0]
            self._delete_atoms(range(2, self.atoms.size + 1))

        # Select the right atom to add and their order
        if water_model == "tip3p":
            if self.hb_type == "acceptor":
                atoms = [a1_xyz, a2_xyz]
            else:
                atoms = [a2_xyz, a3_xyz]
        elif water_model == "tip5p":
            # Order them: H, H, Lp, Lp, we want hydrogen atoms first
            if self.hb_type == "acceptor":
                atoms = [a1_xyz, a2_xyz, a3_xyz, a4_xyz]
            else:
                atoms = [a3_xyz, a4_xyz, a1_xyz, a2_xyz]

        # ... and add the new ones
        for atom, atom_type, partial_charge in zip(atoms, atom_types[1:], partial_charges[1:]):
            self._add_atom(atom, atom_type, partial_charge)

        # Do the HB typing at the end
        self._guess_hydrogen_bonds()

        return True

    def _guess_hydrogen_bonds(self) -> None:
        """Guess all the hydrogen bond anchors in the water molecule.

        We do not need the waterfield here. The result is stored on
        ``self.hydrogen_bonds`` as a pandas DataFrame with columns ``atom_i``
        (1-based anchor id), ``vector_xyz`` (the idealized partner position, a
        :data:`Vec3` of shape ``(3,)``), ``anchor_type`` and ``anchor_name``.

        Returns:
            None. Updates ``self.hydrogen_bonds`` in place.
        """
        hb_length = 2.8
        angle_lp = 109.47
        columns = ["atom_i", "vector_xyz", "anchor_type", "anchor_name"]
        data = []

        oxygen_xyz = self.coordinates(1)[0]

        # For the lone pairs on the oxygen TIP3P
        if self.is_tip3p():
            h1_xyz = self.atoms[1]["xyz"]
            h2_xyz = self.atoms[2]["xyz"]

            # For the lone-pairs, we use the same recipe as for the TIP5P water
            v = geometry.atom_to_move(oxygen_xyz, [h1_xyz, h2_xyz])
            r = oxygen_xyz + geometry.normalize(geometry.vector(h1_xyz, h2_xyz))
            lp1_xyz = geometry.rotate_point(v, oxygen_xyz, r, np.radians(angle_lp / 2.0))
            lp1_xyz = geometry.resize_vector(lp1_xyz, hb_length, oxygen_xyz)
            lp2_xyz = geometry.rotate_point(v, oxygen_xyz, r, -np.radians(angle_lp / 2.0))
            lp2_xyz = geometry.resize_vector(lp2_xyz, hb_length, oxygen_xyz)

            data.append((1, lp1_xyz, "acceptor", "O_L_000"))
            data.append((1, lp2_xyz, "acceptor", "O_L_000"))

        # For the rest...
        for i, atom in enumerate(self.atoms[1:]):
            if atom["name"] == "H":
                hb_type = "donor"
                hb_name = "H_O_004"
            else:
                hb_type = "acceptor"
                hb_name = "O_L_000"

            vector_xyz = geometry.resize_vector(atom["xyz"], hb_length, oxygen_xyz)
            data.append((i + 1, vector_xyz, hb_type, hb_name))

        self.hydrogen_bonds = pd.DataFrame(data=data, columns=columns)

    def translate(self, vector: Vec3) -> None:
        """Translate the water molecule.

        Args:
            vector: 3d translation vector, shape ``(3,)``.

        Returns:
            None.
        """
        water_xyz = self.coordinates() + vector
        for atom_id, coord_xyz in enumerate(water_xyz):
            # +1, because atom ids are 1-based
            self.update_coordinates(coord_xyz, atom_id + 1)

        # We have also to translate the hydrogen bond vectors if present.
        # Adding ``vector`` (shape (3,)) to each per-anchor array in the object
        # column; written element-wise so it is robust across pandas versions
        # (a bare ``+= vector`` tries to broadcast the column against (3,)).
        if self.hydrogen_bonds is not None:
            self.hydrogen_bonds["vector_xyz"] = [
                vector_xyz + vector for vector_xyz in self.hydrogen_bonds["vector_xyz"]
            ]

    def rotate_around_axis(self, axis: str, angle: float) -> bool | None:
        """Rotate water molecule.

        The rotation is along the axis Oxygen and a chosen atom (H or Lp).

        Args:
            axis: Axis name (choices: ``o``, ``h1``, ``h2``, ``lp1``, ``lp2``).
            angle: Rotation angle in degrees.

        Returns:
            ``False`` on error, otherwise ``None``.
        """
        if self.is_spherical():
            print("Error: Cannot rotate a spherical water.")
            return False

        available_axes = ["o", "h1", "h2"]
        if self.is_tip5p():
            available_axes += ["lp1", "lp2"]

        axis = axis.lower()
        water_xyz = self.coordinates()
        oxygen_xyz = water_xyz[0]
        # From 2 to num_atom + 1, because atom ids are 1-based
        # And the oxygen (atom id = 1) is not moving of course...
        atom_ids = list(range(2, water_xyz.shape[0] + 1))

        try:
            ref_atom_id = available_axes.index(axis) + 1
        except ValueError:
            error_str = "Error: Axis %s not recognized. " % axis
            error_str += "Availvable axes for this water molecule: %s" % " ".join(available_axes)
            print(error_str)
            return False

        # If we want to rotate around the oxygen axis, the rotation axis
        # will be the vector between the oxygen and the position in
        # between the two hydrogen bonds
        if ref_atom_id == 1:
            ref_xyz = geometry.atom_to_move(oxygen_xyz, water_xyz[[1, 2]])
        else:
            # -1, because array is 0-based
            ref_xyz = water_xyz[ref_atom_id - 1]
            # Of course the reference atom is not moving because
            # it is part of the rotation axis, so we remove it
            # from the list of atom to rotate
            atom_ids.remove(ref_atom_id)

        r = oxygen_xyz + geometry.normalize(geometry.vector(ref_xyz, oxygen_xyz))

        for atom_id in atom_ids:
            # -1, because array is 0-based
            coord_xyz = geometry.rotate_point(
                water_xyz[atom_id - 1], oxygen_xyz, r, np.radians(angle)
            )
            self.update_coordinates(coord_xyz, atom_id)

        # We have also to rotate the hydrogen bond vectors if present
        if self.hydrogen_bonds is not None:
            for index, vector in self.hydrogen_bonds.iterrows():
                vector_xyz = geometry.rotate_point(
                    vector["vector_xyz"], oxygen_xyz, r, np.radians(angle)
                )
                self.hydrogen_bonds.at[index, "vector_xyz"] = vector_xyz

        return None
