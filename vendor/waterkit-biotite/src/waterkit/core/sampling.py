"""WaterSampler: the algorithmic heart of waterkit.

Faithful port of ``waterkit.sampling``. For one hydration shell it:

1. optimizes disordered (rotatable-hydrogen) groups by Boltzmann choice
   (:meth:`WaterSampler._optimize_disordered_waters`);
2. ranks candidate spherical waters by the best energy around their anchor and
   picks a placement order (Boltzmann + Metropolis filter,
   :meth:`WaterSampler._optimize_placement_order_grid`);
3. for each accepted water, optimizes its position in a spherical shell about the
   anchor (angle + distance constrained,
   :meth:`WaterSampler._optimize_position_grid`), builds the explicit water,
   optimizes its orientation against the OW/HW (TIP3P) or OT/HT/LP (TIP5P) maps
   (:meth:`WaterSampler._optimize_orientation_grid`), and
4. updates the receptor grids in place by adding the placed water's own
   interaction field so subsequent waters "see" it
   (:meth:`WaterSampler._update_maps`).

The water self-interaction map
------------------------------
Upstream WaterKit shipped *precomputed* AutoGrid maps for the field of a single
reference water (``water_SW.map``, ``water_OW.map``/``water_HW.map`` for TIP3P,
``water_OT/HT/LP.map`` for TIP5P) and loaded them in ``__init__`` as one
:class:`~waterkit.core.grid.Map`. Those files have been removed, so this port
regenerates that map in pure Python at construction time, reproducing exactly how
the originals were made:

1. the reference water is read from ``data/water/{model}/water.pdbqt`` (its
   oxygen sits at the origin, matching the original ``gridcenter 0 0 0``);
2. :func:`waterkit.core.grid.generate_maps` evaluates the AutoDock4 force field
   of that reference water on a box centered on its oxygen, producing the
   explicit-water affinity maps -- ``OW``/``HW`` (TIP3P) or ``OT``/``HT``/``LP``
   (TIP5P) -- at the receptor map's spacing (the AutoGrid replacement);
3. :class:`waterkit.core.grid.SphericalWaterMap` adds the Boltzmann
   orientation-averaged ``SW`` map on top, exactly as the original spherical map
   was built.

Box size (deviation from the literal ``box_size=[8, 8, 8]``)
-----------------------------------------------------------
:meth:`_update_maps` builds an 8 Angstrom update cube around the placed water and
*rotates* it before interpolating the water map. The corners of that rotated cube
reach ``~4 * sqrt(3) ~= 6.9`` Angstrom from the oxygen, so an 8 Angstrom (i.e.
+/-4 Angstrom) water map would fall out of bounds there and the interpolator
(``fill_value=np.inf``) would return ``inf``, which :meth:`_update_maps` adds
directly onto the receptor grid -- corrupting it. The original precomputed maps
avoided this by being 15 Angstrom across (tip3p: 40 npts @ 0.375; tip5p: 150 @
0.1; both 15 Angstrom). This port therefore generates a 15 Angstrom map
(:data:`_WATER_MAP_BOX_SIZE`), which fully encloses the rotated update region and
keeps :meth:`_update_maps` numerically faithful.

Caching
-------
Because the ``SW`` map averages ~10k pre-generated orientations over the whole
box, generation is expensive. The resulting arrays (``SW`` plus the explicit
maps, with the box geometry) are cached to ``data/water/{model}/`` as a
compressed ``.npz`` keyed by water model, grid spacing, temperature and box size;
a present, matching cache is loaded instead of regenerating. Cache write/read
failures degrade gracefully to regeneration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from ..utils.boltzmann import boltzmann_acceptance_rejection, boltzmann_choices
from ..utils.geometry import (
    dihedral,
    get_angle,
    normalize,
    quaternion_rotate,
    rotate_point,
    vector,
)
from ..utils.misc import data_path
from ..utils.typing import Coords, Indices, Scalars
from .grid import Map, SphericalWaterMap, generate_maps
from .molecule import Molecule

if TYPE_CHECKING:
    from pathlib import Path

    from .water_box import WaterBox

logger = logging.getLogger("waterkit.sampling")

# Side length (Angstrom) of the regenerated single-water self-interaction map.
# Large enough to enclose the rotated 8 A update cube of :meth:`_update_maps`
# (corners reach ~6.9 A) and matching the extent of the original precomputed
# water maps; see the module docstring.
_WATER_MAP_BOX_SIZE: tuple[float, float, float] = (15.0, 15.0, 15.0)


class WaterSampler:
    """Places and optimizes water molecules against the AutoDock grids."""

    def __init__(
        self,
        water_box: WaterBox,
        temperature: float = 300.0,
        spherical_water_map: str | None = None,
        min_distance: float = 2.5,
        max_distance: float = 3.6,
        angle: float = 90,
        energy_cutoff: float = 0,
    ) -> None:
        """Initialize the water sampler from a :class:`WaterBox`.

        Args:
            water_box: The :class:`~waterkit.core.water_box.WaterBox` providing
                the water model, the receptor map (``water_box.map``) and the
                receptor molecule (``water_box.molecules_in_shell(0)[0]``).
            temperature: Sampling temperature in Kelvin.
            spherical_water_map: Kept for signature compatibility with upstream.
                The ``SW`` map is now generated rather than loaded, so a path
                given here is ignored (a debug message is logged).
            min_distance: Minimum water-anchor distance (Angstrom).
            max_distance: Maximum water-anchor distance (Angstrom).
            angle: Minimum acceptance angle around the anchor (degrees).
            energy_cutoff: Reference energy for the Metropolis accept/reject.
        """
        self._water_box = water_box
        self._water_model = water_box._water_model
        self._ad_map = water_box.map
        self._receptor = water_box.molecules_in_shell(0)[0]

        self._min_distance = min_distance
        self._max_distance = max_distance
        self._angle = angle
        self._temperature = temperature
        self._energy_cutoff = energy_cutoff
        # Boltzmann constant (kcal/mol)
        self._kb = 0.0019872041

        if spherical_water_map is not None:
            logger.debug(
                "spherical_water_map=%s is ignored; the SW map is now generated.",
                spherical_water_map,
            )

        # Water orientations set
        # Load pre-generated water molecules (only the satellite atoms). We use
        # those to sample the orientation.
        n_atoms = 2
        usecols = [0, 1, 2, 3, 4, 5]
        if self._water_model == "tip5p":
            usecols += [6, 7, 8, 9, 10, 11]
            n_atoms += 2

        w_orientation_file = data_path("water_orientations.txt")
        water_orientations = np.loadtxt(w_orientation_file, usecols=usecols)
        shape = (water_orientations.shape[0], n_atoms, 3)
        self._water_orientations = water_orientations.reshape(shape)

        # Reference water and explicit-water atom types
        if self._water_model == "tip3p":
            self._explicit_atom_types = ["OW", "HW"]
            water_ref_file = data_path("water", "tip3p", "water.pdbqt")
        elif self._water_model == "tip5p":
            self._explicit_atom_types = ["OT", "HT", "LP"]
            water_ref_file = data_path("water", "tip5p", "water.pdbqt")
        else:
            raise ValueError("water model %s unknown." % self._water_model)

        self._water_ref = Molecule.from_file(
            water_ref_file, guess_hydrogen_bonds=False, guess_disordered_hydrogens=False
        )

        # Regenerate (or load from cache) the single-water self-interaction map
        # that upstream shipped as precomputed AutoGrid maps.
        self._water_map = self._build_water_map()

    # ------------------------------------------------------------------ #
    # Single-water self-interaction map (regenerated AutoGrid replacement)
    # ------------------------------------------------------------------ #
    def _build_water_map(self) -> Map:
        """Generate (or load from cache) the reference-water self-interaction map.

        Returns:
            A :class:`~waterkit.core.grid.Map` with the ``SW`` map plus the
            explicit-water maps (``OW``/``HW`` for TIP3P, ``OT``/``HT``/``LP``
            for TIP5P), built on the receptor map's spacing.
        """
        spacing = self._ad_map._spacing

        cached = self._load_cached_water_map(spacing)
        if cached is not None:
            logger.debug("Loaded cached water self-interaction map.")
            return cached

        # The box is centered on the reference water oxygen (at the origin).
        box_center = self._water_ref.coordinates(1)[0]

        logger.info(
            "Generating the single-water self-interaction map (%s, spacing=%.3f).",
            self._water_model,
            spacing,
        )
        water_map = generate_maps(
            self._water_ref,
            self._explicit_atom_types,
            box_center=box_center,
            box_size=list(_WATER_MAP_BOX_SIZE),
            spacing=spacing,
        )

        # Boltzmann orientation-averaged spherical-water map on top.
        SphericalWaterMap(water_model=self._water_model, temperature=self._temperature).run(
            water_map, name="SW"
        )

        self._save_cached_water_map(water_map, spacing)

        return water_map

    def _water_map_cache_path(self, spacing: float) -> Path:
        """Return the cache path for the water map at the given spacing.

        Args:
            spacing: Grid spacing (Angstrom) the cache is keyed on.

        Returns:
            The ``.npz`` cache path for this water model, spacing, temperature
            and box size.
        """
        box = _WATER_MAP_BOX_SIZE[0]
        fname = "water_map_cache_s%g_t%g_b%g.npz" % (spacing, self._temperature, box)
        return data_path("water", self._water_model, fname)

    def _load_cached_water_map(self, spacing: float) -> Map | None:
        """Load the cached water map if a matching cache exists, else ``None``.

        Args:
            spacing: Grid spacing (Angstrom) the cache is keyed on.

        Returns:
            A reconstructed :class:`~waterkit.core.grid.Map` (the ``SW`` map plus
            the explicit-water maps) if a readable matching cache exists,
            otherwise ``None``.
        """
        cache_path = self._water_map_cache_path(spacing)

        if not cache_path.exists():
            return None

        try:
            with np.load(cache_path) as data:
                center = data["_center_"]
                npts = data["_npts_"]
                spacing_cached = float(data["_spacing_"])
                names = ["SW", *self._explicit_atom_types]
                arrays = {name: data[name] for name in names}
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("Ignoring unreadable water-map cache %s: %s", cache_path, exc)
            return None

        water_map = Map._create_empty(npts, spacing_cached, center)
        for name, affinity_map in arrays.items():
            water_map.add_map(name, affinity_map)

        return water_map

    def _save_cached_water_map(self, water_map: Map, spacing: float) -> None:
        """Cache the water map's arrays and geometry (best effort).

        Args:
            water_map: The :class:`~waterkit.core.grid.Map` to cache (its ``SW``
                and explicit-water arrays plus center/npts/spacing geometry).
            spacing: Grid spacing (Angstrom) the cache is keyed on.

        Returns:
            None. Write failures are logged and ignored (degrade to
            regeneration on the next run).
        """
        cache_path = self._water_map_cache_path(spacing)

        names = ["SW", *self._explicit_atom_types]
        arrays = {name: water_map._maps[name] for name in names}
        arrays["_center_"] = water_map._center
        arrays["_npts_"] = water_map._npts
        arrays["_spacing_"] = np.asarray(water_map._spacing)

        try:
            np.savez_compressed(cache_path, **arrays)
            logger.debug("Cached water self-interaction map to %s.", cache_path)
        except OSError as exc:
            logger.warning("Could not cache water map to %s: %s", cache_path, exc)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _optimize_disordered_waters(
        self, waters: list[Molecule], connections: pd.DataFrame
    ) -> list[float]:
        """Optimize water molecules on rotatable bonds.

        Args:
            waters: The candidate water molecules for this shell.
            connections: The receptor-water connection table (a pandas
                ``DataFrame`` with ``atom_i``/``molecule_j`` columns).

        Returns:
            A list with the chosen energy (kcal/mol) of each optimized disordered
            group.
        """
        ad_map = self._ad_map
        receptor = self._receptor
        disordered_energies = []
        self._rotation = 10

        # Number of rotation necessary to do a full spin
        n_rotation = int(np.floor(360 / self._rotation)) - 1
        rotation = np.radians(self._rotation)

        # Iterate through every disordered bonds
        for index, row in receptor.rotatable_bonds.iterrows():
            energies = []
            angles = []
            rot_waters = []

            # Get index of all the waters attached
            # to a disordered group by looking at the connections
            tmp = connections["atom_i"].isin(row[["atom_i", "atom_j"]])
            molecule_j = connections.loc[tmp]["molecule_j"].values
            rot_waters.extend([waters[j] for j in molecule_j])

            if rot_waters:
                # Get energy of the favorable disordered waters
                energy_waters = np.array(
                    [
                        ad_map.energy(
                            w.atom_informations(),
                            ignore_electrostatic=True,
                            ignore_desolvation=True,
                        )
                        for w in rot_waters
                    ]
                )
                energy_waters[energy_waters > 0] = 0
                energies.append(np.sum(energy_waters))
                # Current angle of the disordered group
                current_angle = dihedral(
                    row[["atom_i_xyz", "atom_j_xyz", "atom_k_xyz", "atom_l_xyz"]].values
                )
                angles.append(current_angle)

                # Find all the atoms that depends on these atoms. This
                # will be useful when we will want to rotate a whole sidechain.
                # Atom children has to be initialized before
                # molecule._OBMol.FindChildren(atom_children, match[2], match[3])

                # Atoms 3 and 2 define the rotation axis
                p1 = row["atom_k_xyz"]
                p2 = row["atom_j_xyz"]

                # Scan all the angles
                for i in range(n_rotation):
                    # TODO: Performance wise, we should not update water
                    # coordinates everytime. Coordinates should be extracted
                    # before doing the optimization and only at the end
                    # we update the coordinates of the water molecules.
                    for rot_water in rot_waters:
                        p0 = rot_water.coordinates(1)[0]
                        p_new = rotate_point(p0, p1, p2, rotation)
                        rot_water.update_coordinates(p_new, atom_id=1)

                    # Get energy and update the current angle (increment rotation)
                    energy_waters = np.array(
                        [
                            ad_map.energy(
                                w.atom_informations(),
                                ignore_electrostatic=True,
                                ignore_desolvation=True,
                            )
                            for w in rot_waters
                        ]
                    )

                    energy_waters[energy_waters > 0] = 0
                    energies.append(np.nansum(energy_waters))
                    current_angle += rotation
                    angles.append(current_angle)

                # Pick state based on Boltzmann choices
                i = boltzmann_choices(energies, self._temperature)[0]

                disordered_energies.append(energies[i])

                # Calculate the best angle, based on how much we rotated
                best_angle = np.radians((360.0 - np.degrees(current_angle)) + np.degrees(angles[i]))
                # Update coordinates to the choosen state
                for rot_water in rot_waters:
                    p0 = rot_water.coordinates(1)[0]
                    p_new = rotate_point(p0, p1, p2, best_angle)
                    rot_water.update_coordinates(p_new, atom_id=1)
                    # Update also the anchor point
                    rot_water.hb_anchor = rotate_point(rot_water.hb_anchor, p1, p2, best_angle)
                    rot_water.hb_vector = rotate_point(rot_water.hb_vector, p1, p2, best_angle)

        return disordered_energies

    def _neighbor_points_grid(
        self, water: Molecule, from_edges: float | None = None
    ) -> tuple[Coords, Scalars]:
        """Return the allowed positions (and energies) around a water's anchor.

        The allowed positions are selected as follows:

        1. Get all the grid point coordinates around the anchor (a sphere). If
           the anchor type is donor, reduce the radius by 1 Angstrom, because the
           hydrogen atom is closer to the water than the heavy atom.
        2. Compute the angles between all the coordinates and the anchor.
        3. Select coordinates with an angle superior or equal to the chosen one.
        4. Get their energy.

        Args:
            water: The (spherical) water molecule being placed.
            from_edges: If set, discard grid points within this distance
                (Angstrom) of the box edges.

        Returns:
            A tuple ``(coord_sphere, energy_sphere)`` where ``coord_sphere`` has
            shape ``(n, 3)`` (the accepted grid-point coordinates) and
            ``energy_sphere`` has shape ``(n,)`` (their oxygen energies,
            kcal/mol).
        """
        ad_map = self._ad_map
        oxygen_type = water.atom_types(0)

        if water.hb_type == "donor":
            coord_sphere = ad_map.neighbor_points(
                water.hb_anchor, self._max_distance - 1.0, self._min_distance - 1.0
            )
        else:
            coord_sphere = ad_map.neighbor_points(
                water.hb_anchor, self._max_distance, self._min_distance
            )

        if from_edges is not None:
            is_close = ad_map.is_close_to_edge(coord_sphere, from_edges)
            coord_sphere = coord_sphere[~is_close]

        # It is mandatory to normalize the hb_vector, otherwise you don't get the angle right
        hb_vector = water.hb_anchor + normalize(vector(water.hb_vector, water.hb_anchor))
        angle_sphere = get_angle(coord_sphere, water.hb_anchor, hb_vector)

        coord_sphere = coord_sphere[angle_sphere >= self._angle]
        energy_sphere = ad_map.energy_coordinates(coord_sphere, atom_type=oxygen_type)

        return coord_sphere, energy_sphere

    def _optimize_placement_order_grid(
        self, waters: list[Molecule], from_edges: float | None = None
    ) -> Indices:
        """Return a Boltzmann-sampled, Metropolis-filtered water placement order.

        Args:
            waters: The candidate water molecules for this shell.
            from_edges: If set, discard anchor grid points within this distance
                of the box edges.

        Returns:
            A 1D integer array of water indices (the placement order), possibly
            empty.
        """
        energies = []

        for water in waters:
            _, energy_sphere = self._neighbor_points_grid(water, from_edges)

            if energy_sphere.size:
                energy_sphere[energy_sphere == 0.0] = np.inf
                energies.append(np.min(energy_sphere))
            else:
                energies.append(np.inf)

        energies = np.array(energies)

        # Pick order based on Boltzmann choices
        order = boltzmann_choices(energies, self._temperature, len(energies))

        if order.size > 0:
            decisions = boltzmann_acceptance_rejection(
                energies[order], self._energy_cutoff, self._temperature
            )
            order = order[decisions]

            return order
        else:
            return np.array([])

    def _optimize_position_grid(
        self, water: Molecule, add_noise: bool = False, from_edges: float | None = None
    ) -> float:
        """Optimize the position of the spherical water molecule.

        The movement of the water is constrained by the distance and the angle
        with the anchor.

        Args:
            water: The spherical water molecule to place.
            add_noise: If True, jitter the chosen position by up to half the grid
                spacing.
            from_edges: If set, discard anchor grid points within this distance
                of the box edges.

        Returns:
            The energy (kcal/mol) of the chosen (or current) position.
        """
        ad_map = self._ad_map
        oxygen_type = water.atom_types([0])

        coord_sphere, energy_sphere = self._neighbor_points_grid(water, from_edges)

        if energy_sphere.size:
            # Pick position based on Boltzmann choices
            idx = boltzmann_choices(energy_sphere, self._temperature)

            if idx.size > 0:
                new_coord = coord_sphere[idx[0]]

                if add_noise:
                    limit = ad_map._spacing / 2.0
                    new_coord += np.random.uniform(-limit, limit, new_coord.shape[0])

                # Update the coordinates
                water.translate(vector(water.coordinates(1), new_coord))
                return energy_sphere[idx[0]]

        # If we do not find anything, at least we return the energy
        # of the current water molecule.
        return ad_map.energy_coordinates(water.coordinates(1), atom_type=oxygen_type)[0]

    def _optimize_orientation_grid(self, water: Molecule) -> float:
        """Optimize the orientation of the water molecule using the grid.

        Args:
            water: The explicit water molecule whose orientation is optimized.

        Returns:
            The energy (kcal/mol) of the chosen (or current) orientation.
        """
        ad_map = self._ad_map
        oxygen_xyz = water.coordinates(1)
        water_info = water.atom_informations()
        energies = np.zeros(self._water_orientations.shape[0])

        # Translate the coordinates
        water_orientations = self._water_orientations + oxygen_xyz
        # Get the energies for each atom
        # Oxygen first
        energies += ad_map.energy_coordinates(oxygen_xyz, water_info["t"][0])
        # ... and then hydrogens/lone-pairs
        for i, atom_type in enumerate(water_info["t"][1:]):
            energies += ad_map.energy_coordinates(water_orientations[:, i], atom_type)

        # Pick orientation based on Boltzmann choices
        idx = boltzmann_choices(energies, self._temperature)

        if idx.size > 0:
            # Update the coordinates with the selected orientation, except oxygen (i + 2)
            new_orientation = water_orientations[idx[0]]
            for i, xyz in enumerate(new_orientation):
                water.update_coordinates(xyz, i + 2)

            return energies[idx[0]]

        # If we do not find anything, at least we return the energy
        # of the current water molecule.
        current_energy = ad_map.energy(
            water.atom_informations(), ignore_electrostatic=True, ignore_desolvation=True
        )

        return current_energy

    def _update_maps(self, water: Molecule) -> None:
        """Add a placed water's own interaction field onto the receptor maps.

        If we choose the closest point in the grid and not the coordinates of the
        oxygen as the center of the grid, it is because we want to avoid any edge
        effect when we combine the small box with the bigger box, and also the
        energy is not interpolated but it comes from the grid directly.

        Args:
            water: The explicit, placed water molecule.

        Returns:
            None. The receptor grids (``self._ad_map``) are mutated in place.
        """
        spacing = self._ad_map._spacing
        ad_map = self._ad_map
        boxsize = np.array([8, 8, 8])
        npts = np.round(boxsize / spacing).astype(int) // 2 * 2 + 1
        water_xyz = water.coordinates()

        # The center is the closest grid point from the water molecule
        center_xyz = self._ad_map.neighbor_points(water.coordinates(1)[0], spacing)[0]
        center_index = self._ad_map._cartesian_to_index(center_xyz)

        # Get grid indices in the receptor box
        # Necessary, in order to add the water map to the receptor map
        size_half_box = (npts - 1) // 2
        i_min = np.clip(center_index - size_half_box, [0, 0, 0], ad_map._npts)
        i_max = np.clip(center_index + size_half_box + 1, [0, 0, 0], ad_map._npts)
        indices = np.index_exp[i_min[0] : i_max[0], i_min[1] : i_max[1], i_min[2] : i_max[2]]

        # Convert the indices in xyz coordinates, and translate it in order
        # that the water molecule is at the origin
        x = np.arange(i_min[0], i_max[0])
        y = np.arange(i_min[1], i_max[1])
        z = np.arange(i_min[2], i_max[2])
        X, Y, Z = np.meshgrid(x, y, z)
        grid_index = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        grid_xyz = ad_map._index_to_cartesian(grid_index)
        grid_xyz -= water_xyz[0]

        # Get the quaternion between the current water and the water reference
        # The water is translating to the origin before
        q = quaternion_rotate(water_xyz - water_xyz[0], self._water_ref.coordinates())

        # Rotate the grid!
        r = Rotation.from_quat(q)
        rotated_grid_xyz = r.apply(grid_xyz)

        atom_types = ["SW"]
        if self._water_model == "tip3p":
            atom_types += ["OW", "HW"]
        elif self._water_model == "tip5p":
            atom_types += ["OT", "HT", "LP"]

        # Interpolation baby!
        for atom_type in atom_types:
            energy = self._water_map.energy_coordinates(rotated_grid_xyz, atom_type)
            energy = np.swapaxes(energy.reshape((y.shape[0], x.shape[0], z.shape[0])), 0, 1)

            ad_map._maps[atom_type][indices] += energy
            ad_map._maps_interpn[atom_type] = ad_map._generate_affinity_map_interpn(
                ad_map._maps[atom_type]
            )

    def sample_grid(
        self,
        waters: list[Molecule],
        connections: pd.DataFrame | None = None,
        tmp_dir: str = ".",
        opt_disordered: bool = True,
    ) -> tuple[list[Molecule], dict[str, pd.DataFrame]]:
        """Optimize position of water molecules.

        Args:
            waters: The candidate water molecules for this shell.
            connections: The receptor-water connection table (pandas
                ``DataFrame``), or ``None``.
            tmp_dir: Kept for signature compatibility (unused).
            opt_disordered: Whether to optimize disordered (rotatable) groups
                first.

        Returns:
            A tuple ``(waters, df)`` with the accepted water molecules and a dict
            holding the ``shells`` (and ``connections``) ``DataFrame``s.
        """
        df = {}
        data = []
        unfavorable_water_indices = []
        shell_id = self._water_box.number_of_shells()
        add_noise = True

        if opt_disordered and connections is not None:
            self._optimize_disordered_waters(waters, connections)

        # The placement order is based on the best energy around each hydrogen
        # anchor point. But if it returns [], it means the most favorable spots
        # for placing water molecules are definitively not that favorable,
        # likely they are all outside the box.
        water_orders = self._optimize_placement_order_grid(waters, from_edges=1.0)
        unfavorable_water_indices.extend(set(np.arange(len(waters))) - set(water_orders))

        # And now we sample the position of all the water individually. The
        # acceptance of the new position/orientation is based on the metropolis
        # acceptance-rejection criteria. But per default, everything is accepted
        # with a favorable energy (< 0 kcal/mol).
        for i in water_orders:
            water = waters[i]

            energy_position = self._optimize_position_grid(water, add_noise, from_edges=1.0)

            # The first great filter
            if boltzmann_acceptance_rejection(
                energy_position, self._energy_cutoff, self._temperature
            )[0]:
                # Build the explicit water
                water.build_explicit_water(self._water_model)

                # Optimize the orientation
                energy_orientation = self._optimize_orientation_grid(water)

                # The last great energy filter
                if boltzmann_acceptance_rejection(
                    energy_orientation, self._energy_cutoff, self._temperature
                )[0]:
                    data.append((shell_id + 1, energy_position, energy_orientation))
                    self._update_maps(water)
                else:
                    unfavorable_water_indices.append(i)
            else:
                unfavorable_water_indices.append(i)

        # Keep only the good waters
        waters = [waters[i] for i in water_orders if i not in unfavorable_water_indices]
        # Keep connections of the good waters
        if connections is not None:
            index = connections.loc[connections["molecule_j"].isin(unfavorable_water_indices)].index
            connections.drop(index, inplace=True)
            # Renumber the water molecules
            connections["molecule_j"] = range(0, len(waters))
            df["connections"] = connections

        # Add water shell informations
        columns = ["shell_id", "energy_position", "energy_orientation"]
        df_shell = pd.DataFrame(data, columns=columns)
        df["shells"] = df_shell

        return (waters, df)
