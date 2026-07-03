"""AutoDock affinity grids: reading, interpolation, and pure-Python generation.

This module combines three dependency-light pieces ported from upstream
WaterKit:

* :class:`Map` -- a faithful port of ``waterkit.autodock_map``. It holds one 3D
  affinity grid per atom type, wraps each grid in a
  :class:`scipy.interpolate.RegularGridInterpolator` for off-grid energies, and
  builds a :class:`scipy.spatial.cKDTree` for neighbour queries. It reads and
  writes AutoGrid ``.map`` / ``.fld`` files.
* :func:`generate_maps` -- a pure-Python replacement for the AutoGrid binary. It
  mirrors the role of ``waterkit.autogrid.AutoGrid.run`` but evaluates the
  AutoDock4 force field directly on every grid point (vectorised), producing one
  affinity map per ligand atom type plus an ``Electrostatics`` and a
  ``Desolvation`` map. The result is structurally identical to what
  :meth:`Map.from_fld` produces.
* :class:`SphericalWaterMap` -- a faithful port of
  ``waterkit.spherical_model_map``. It builds the Boltzmann-orientation-averaged
  ``SW`` map from the explicit water maps.

The energy model is a near-verbatim port of ``waterkit.forcefield``. Because
``waterkit.core.forcefield.AutoDockForceField`` is not yet implemented (and its
public ``intermolecular_energy`` interface returns only a *combined* energy,
which cannot be decomposed into the separate vdW/electrostatic/desolvation maps
AutoGrid emits), :func:`generate_maps` uses a self-contained evaluator,
:class:`_AutoDockGridForceField`, that ports the per-term force-field math. It
can be swapped for the package force field once that is fleshed out.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import spatial
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

from ..utils import boltzmann_probabilities, data_path
from ..utils.exceptions import MapError
from ..utils.typing import (
    BoolMask,
    Coords,
    Float,
    Grid3D,
    Indices,
    Int,
    Point3,
    Scalars,
    Shaped,
)

if TYPE_CHECKING:
    from .molecule import Molecule

logger = logging.getLogger("waterkit.grid")

# AutoGrid clamps blown-up interaction energies (clashes) before writing maps.
ENERGY_CLAMP = 100000.0
# Minimum pairwise distance used in reciprocals to avoid division by zero.
_DISTANCE_FLOOR = 1e-2
# Gaussian width of the desolvation term (Angstrom).
_DESOLV_SIGMA = 3.6


class _FastGridInterpolator:
    """Fast multilinear interpolation on a uniform grid via ``map_coordinates``.

    A drop-in replacement for
    ``RegularGridInterpolator(edges, grid, bounds_error=False, fill_value=inf)``
    for ``method="linear"`` (the default everywhere in waterkit). Because the
    AutoDock grid is uniform, :func:`scipy.ndimage.map_coordinates` (order 1)
    evaluates large batches of query points far faster than
    ``RegularGridInterpolator`` (which dominates the spherical-water map build).
    Any other interpolation method falls back to a lazily-built
    ``RegularGridInterpolator``.

    Points outside the box return ``inf`` (matching ``fill_value=inf``), which the
    spherical-water calculation relies on to drop orientations that poke out of
    the box. The object holds only numpy arrays, so it pickles cleanly for the
    multiprocessing workers in :meth:`SphericalWaterMap.run`.
    """

    def __init__(
        self,
        edges: tuple[Float[np.ndarray, "nx"], Float[np.ndarray, "ny"], Float[np.ndarray, "nz"]],
        grid: Grid3D,
    ) -> None:
        """Bind the interpolator to a grid and its per-axis edge coordinates.

        Args:
            edges: Per-axis edge coordinates ``(x, y, z)`` (1D arrays).
            grid: The 3D affinity grid indexed ``[ix, iy, iz]``.
        """
        self._edges = edges
        self._grid = np.ascontiguousarray(grid, dtype=float)
        self._lower = np.array([float(e[0]) for e in edges])
        self._upper = np.array([float(e[-1]) for e in edges])
        # Fractional index per Angstrom along each axis (uniform grid).
        self._scale = np.array([(len(e) - 1) / (e[-1] - e[0]) for e in edges])
        self._rgi: RegularGridInterpolator | None = None

    def __call__(self, xyz: Coords, method: str = "linear") -> Scalars:
        """Interpolate the grid at ``xyz`` (linear) returning ``inf`` outside the box.

        Args:
            xyz: Query coordinates ``(n, 3)`` (a single ``(3,)`` point is also
                accepted, returning a scalar).
            method: Interpolation method; ``"linear"`` uses ``map_coordinates``,
                anything else falls back to ``RegularGridInterpolator``.

        Returns:
            The interpolated energies, one per query point.
        """
        xyz = np.asarray(xyz, dtype=float)
        single = xyz.ndim == 1
        pts = xyz[None, :] if single else xyz

        if method != "linear":
            if self._rgi is None:
                self._rgi = RegularGridInterpolator(
                    self._edges, self._grid, bounds_error=False, fill_value=np.inf
                )
            out = self._rgi(pts, method=method)
            return out[0] if single else out

        # Continuous (fractional) grid indices, then batched linear sampling.
        frac = (pts - self._lower) * self._scale
        out = map_coordinates(self._grid, frac.T, order=1, mode="nearest")
        # Any point outside the box -> inf (matches RegularGridInterpolator).
        oob = np.any((pts < self._lower) | (pts > self._upper), axis=1)
        out[oob] = np.inf
        return out[0] if single else out


class Map:
    """A set of AutoDock affinity grids sharing one box, with interpolation.

    Each entry of :attr:`_maps` is a 3D ``numpy`` array indexed ``[ix, iy, iz]``
    over the grid box, and :attr:`_maps_interpn` holds the matching
    :class:`~scipy.interpolate.RegularGridInterpolator`. The box is described by
    :attr:`_center`, :attr:`_spacing` and :attr:`_npts` (the number of grid
    *points* along each axis, always odd). :attr:`_kdtree` indexes every grid
    point for neighbour queries and :attr:`_edges` are the per-axis coordinates.
    """

    def __init__(
        self,
        map_files: str | Sequence[str] | None = None,
        labels: str | Sequence[str] | None = None,
    ) -> None:
        """Initialize a Map by reading AutoDock ``.map`` files.

        Args:
            map_files: One map file or a list of map files.
            labels: The atom type label for each map (same length as
                ``map_files``).
        """
        self._maps: dict[str, Grid3D] = {}
        self._maps_interpn: dict[str, _FastGridInterpolator] = {}

        if map_files is None or labels is None:
            return

        if not isinstance(map_files, (list, tuple)):
            map_files = [map_files]
        if not isinstance(labels, (list, tuple)):
            labels = [labels]

        if len(map_files) != len(labels):
            raise MapError("map files and labels must have the same number of elements")

        grid_information = None
        for map_file, label in zip(map_files, labels):
            current = self._grid_information_from_map(map_file)
            if grid_information is not None and not self._grids_match(grid_information, current):
                raise MapError(f"grid {label} is different from the previous one.")
            grid_information = current

        self._setup_geometry(
            grid_information["spacing"],
            grid_information["nelements"],
            grid_information["center"],
        )

        for label, map_file in zip(labels, map_files):
            affinity_map = self._read_affinity_map(map_file)
            self.add_map(label, affinity_map)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _setup_geometry(
        self, spacing: float, npts: Int[np.ndarray, "3"], center: Float[np.ndarray, "3"]
    ) -> None:
        """Configure the box geometry (edges, bounds, kdtree) for empty maps.

        Args:
            spacing: Grid spacing in Angstrom.
            npts: Number of grid points along each axis ``(nx, ny, nz)``.
            center: Cartesian center of the box ``(x, y, z)``.
        """
        self._spacing = float(spacing)
        self._npts = np.asarray(npts, dtype=int)
        self._center = np.asarray(center, dtype=float)

        # Half of each side of the box.
        half = (self._spacing * (self._npts - 1)) / 2.0
        self._xmin, self._ymin, self._zmin = self._center - half
        self._xmax, self._ymax, self._zmax = self._center + half

        self._kdtree, self._edges = self._build_kdtree_from_grid()

    @classmethod
    def _create_empty(
        cls, npts: Int[np.ndarray, "3"], spacing: float, center: Float[np.ndarray, "3"]
    ) -> Map:
        """Return a Map with the given geometry and no affinity maps yet.

        Args:
            npts: Number of grid points along each axis ``(nx, ny, nz)``.
            spacing: Grid spacing in Angstrom.
            center: Cartesian center of the box ``(x, y, z)``.

        Returns:
            An empty :class:`Map` with the requested box geometry.
        """
        obj = cls()
        obj._setup_geometry(spacing, npts, center)
        return obj

    @staticmethod
    def _grids_match(a: dict, b: dict) -> bool:
        """Return True if two grid-information dicts describe the same box."""
        return (
            a["spacing"] == b["spacing"]
            and np.array_equal(a["nelements"], b["nelements"])
            and np.array_equal(a["center"], b["center"])
        )

    def __str__(self) -> str:
        """Return a short human-readable description of the box and maps."""
        info = "Box center   : %s\n" % " ".join(self._center.astype(str))
        info += "Box size     : %s\n" % " ".join(self._npts.astype(str))
        info += "Box spacing  : %s\n" % self._spacing
        info += "Affinity maps: %s\n" % " ".join(self._maps.keys())
        return info

    def info(self) -> dict:
        """Return information about the affinity maps.

        Returns:
            A dict with keys ``box_center`` (ndarray), ``box_size`` (ndarray),
            ``box_spacing`` (float) and ``maps`` (the map labels).
        """
        return {
            "box_center": self._center,
            "box_size": self._npts,
            "box_spacing": self._spacing,
            "maps": self._maps.keys(),
        }

    def copy(self) -> Map:
        """Return a deep copy of this Map."""
        return copy.deepcopy(self)

    @classmethod
    def from_fld(cls, fld_file: str | Path) -> Map | None:
        """Read a map set from an AutoGrid ``.fld`` header and its ``.map`` files.

        Args:
            fld_file: Path to the AutoGrid ``.fld`` file.

        Returns:
            The constructed :class:`Map`, or ``None`` if no maps were found.
        """
        fld_file = str(fld_file)
        map_files: list[str] = []
        labels: list[str] = []

        path = os.path.dirname(fld_file)
        if path == "":
            path = "."

        with open(fld_file) as f:
            for line in f:
                if re.search("^label=", line):
                    labels.append(line.split("=")[1].split("#")[0].split("-")[0].strip())
                elif re.search("^variable", line):
                    map_files.append(
                        path + os.sep + line.split(" ")[2].split("=")[1].split("/")[-1]
                    )

        if map_files and labels:
            return cls(map_files, labels)
        return None

    def _grid_information_from_map(self, map_file: str) -> dict:
        """Read spacing/nelements/center from the header of a ``.map`` file."""
        grid_information: dict = {"spacing": None, "nelements": None, "center": None}

        with open(map_file) as f:
            for line in f:
                if re.search("^SPACING", line):
                    grid_information["spacing"] = float(line.split(" ")[1])
                elif re.search("^NELEMENTS", line):
                    nelements = np.array(line.split(" ")[1:4], dtype=int)
                    # Transform even numbers to the nearest odd integer (n points).
                    nelements = nelements // 2 * 2 + 1
                    grid_information["nelements"] = nelements
                elif re.search("CENTER", line):
                    grid_information["center"] = np.array(line.split(" ")[1:4], dtype=float)
                elif re.search("^[0-9]", line):
                    # The header is finished as soon as a numeric value appears.
                    break

        return grid_information

    def _read_affinity_map(self, map_file: str) -> Grid3D:
        """Read the affinity values of a ``.map`` file into a 3D array.

        Args:
            map_file: Path to the AutoDock ``.map`` file.

        Returns:
            The affinity grid indexed ``[ix, iy, iz]``.
        """
        with open(map_file) as f:
            lines = f.readlines()

            npts = np.array(lines[4].split(" ")[1:4], dtype=int) + 1

            affinity = [float(line) for line in lines[6:]]
            # AutoDock stores z fastest; swap x and z axes to index as [x, y, z].
            affinity = np.swapaxes(np.reshape(affinity, npts[::-1]), 0, 2)

        return affinity

    def _generate_affinity_map_interpn(self, affinity_map: Grid3D) -> _FastGridInterpolator:
        """Return an interpolator over the grid edges for ``affinity_map``.

        Args:
            affinity_map: The 3D affinity grid to wrap.

        Returns:
            A :class:`_FastGridInterpolator` bound to the box edges, returning
            ``inf`` outside the box. It uses :func:`scipy.ndimage.map_coordinates`
            for the (default) linear method and is numerically equivalent to a
            ``RegularGridInterpolator`` with ``fill_value=inf``, but much faster
            for the large batched queries in the spherical-water map build.
        """
        return _FastGridInterpolator(self._edges, affinity_map)

    def _build_kdtree_from_grid(
        self,
    ) -> tuple[
        spatial.cKDTree,
        tuple[Float[np.ndarray, "nx"], Float[np.ndarray, "ny"], Float[np.ndarray, "nz"]],
    ]:
        """Return the kdtree and per-axis edge coordinates of the grid.

        Returns:
            A 2-tuple of the :class:`~scipy.spatial.cKDTree` over all grid points
            and the per-axis edge coordinates ``(x, y, z)`` (1D arrays of length
            ``nx``, ``ny`` and ``nz`` respectively).
        """
        x = np.linspace(self._xmin, self._xmax, self._npts[0])
        y = np.linspace(self._ymin, self._ymax, self._npts[1])
        z = np.linspace(self._zmin, self._zmax, self._npts[2])

        # Keep a tuple of arrays so non-cubic grids interpolate correctly.
        edges = (x, y, z)

        X, Y, Z = np.meshgrid(x, y, z)
        xyz = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        kdtree = spatial.cKDTree(xyz)

        return kdtree, edges

    def _index_to_cartesian(
        self, idx: Int[np.ndarray, "*batch 3"]
    ) -> Float[np.ndarray, "*batch 3"]:
        """Return the cartesian coordinates of grid index ``idx``.

        Shape-polymorphic: accepts a single ``(3,)`` index or a batch ``(n, 3)``.

        Args:
            idx: Grid index/indices ``(ix, iy, iz)``.

        Returns:
            The corresponding cartesian coordinate(s).
        """
        return self._kdtree.mins + idx * self._spacing

    def _cartesian_to_index(
        self, xyz: Float[np.ndarray, "*batch 3"]
    ) -> Int[np.ndarray, "*batch 3"]:
        """Return the closest grid index of cartesian coordinates ``xyz``.

        Shape-polymorphic: accepts a single ``(3,)`` point or a batch ``(n, 3)``.

        Args:
            xyz: Cartesian coordinate(s) ``(x, y, z)``.

        Returns:
            The nearest grid index/indices, clipped to the valid range.
        """
        idx = np.rint((xyz - self._kdtree.mins) / self._spacing).astype(int)
        # Clip indices that fall outside the grid to the nearest valid index.
        np.clip(idx, [0, 0, 0], self._npts, idx)
        return idx

    def size(self) -> int:
        """Return the total number of grid points in the box."""
        return int(self._npts.prod())

    def create_map(self, name: str, fill_value: float | None = None) -> bool:
        """Create a new (zero- or constant-filled) map.

        Args:
            name: Name of the new map.
            fill_value: Constant value to fill the map with (default: zeros).

        Returns:
            True if the map was created, False if it already existed.
        """
        if name not in self._maps:
            if fill_value is None:
                new_map = np.zeros(self._npts)
            else:
                new_map = np.full(self._npts, fill_value)

            self._maps[name] = new_map
            self._maps_interpn[name] = self._generate_affinity_map_interpn(new_map)
            return True

        logger.error("map %s already exists.", name)
        return False

    def add_map(self, name: str, new_map: Grid3D) -> None:
        """Add a precomputed affinity map.

        Args:
            name: Name of the new map.
            new_map: A 3D array ``(nx, ny, nz)`` with the same shape as the grid.

        Raises:
            MapError: If the shape does not match the grid or the name exists.
        """
        if not isinstance(new_map, np.ndarray):
            new_map = np.array(new_map)

        if not np.array_equal(new_map.shape, self._npts):
            raise MapError(
                f"New grid does not have the same dimension ({new_map.shape} != {self._npts})"
            )

        if name not in self._maps:
            self._maps[name] = new_map
            self._maps_interpn[name] = self._generate_affinity_map_interpn(new_map)
        else:
            raise MapError(f"Map {name} already exists.")

    def delete_map(self, name: str) -> None:
        """Delete a map (warns, but does not raise, if absent)."""
        try:
            del self._maps[name]
            del self._maps_interpn[name]
        except KeyError:
            warnings.warn(f"Could not delete map {name}.", RuntimeWarning, stacklevel=2)

    def copy_map(self, new_name: str, name: str) -> None:
        """Copy an existing map under a new name.

        Raises:
            MapError: If the source map does not exist.
        """
        if name in self._maps:
            map_copy = self._maps[name].copy()
            self._maps[new_name] = map_copy
            self._maps_interpn[new_name] = self._generate_affinity_map_interpn(map_copy)
        else:
            raise MapError(f"Map {name} does not exist. Cannot copy.")

    def atoms_in_map(self, molecule: Molecule) -> Indices:
        """Return the indices of ``molecule`` atoms whose coordinates lie in the box.

        Args:
            molecule: The molecule whose atom coordinates are tested.

        Returns:
            The integer indices of the atoms inside the box.
        """
        xyz = molecule.atoms["xyz"]
        in_out = self.is_in_map(xyz)
        idx = np.arange(0, xyz.size)[in_out]
        return idx

    def is_in_map(self, xyz: Coords) -> BoolMask:
        """Return a boolean array marking which of ``xyz`` lie inside the box.

        Args:
            xyz: Coordinates ``(n, 3)`` (a single ``(3,)`` point is accepted).

        Returns:
            A boolean mask, ``True`` where the point lies inside the box.
        """
        xyz = np.atleast_2d(xyz)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        x_in = np.logical_and(self._xmin <= x, x <= self._xmax)
        y_in = np.logical_and(self._ymin <= y, y <= self._ymax)
        z_in = np.logical_and(self._zmin <= z, z <= self._zmax)
        all_in = np.all((x_in, y_in, z_in), axis=0)

        return all_in

    def is_close_to_edge(self, xyz: Coords, distance: float) -> BoolMask:
        """Return a boolean array marking points within ``distance`` of any edge.

        Args:
            xyz: Coordinates ``(n, 3)`` (a single ``(3,)`` point is accepted).
            distance: Edge proximity threshold in Angstrom.

        Returns:
            A boolean mask, ``True`` where the point is within ``distance`` of
            any box edge.
        """
        xyz = np.atleast_2d(xyz)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        x_close = np.logical_or(
            np.abs(self._xmin - x) <= distance, np.abs(self._xmax - x) <= distance
        )
        y_close = np.logical_or(
            np.abs(self._ymin - y) <= distance, np.abs(self._ymax - y) <= distance
        )
        z_close = np.logical_or(
            np.abs(self._zmin - z) <= distance, np.abs(self._zmax - z) <= distance
        )
        close_to = np.any((x_close, y_close, z_close), axis=0)

        return close_to

    def energy_coordinates(self, xyz: Coords, atom_type: str, method: str = "linear") -> Scalars:
        """Interpolate the affinity of ``atom_type`` at arbitrary coordinates.

        Args:
            xyz: A 2D array ``(n, 3)`` of 3D coordinates.
            atom_type: The map (atom type) to interpolate.
            method: Interpolation method (default: ``"linear"``).

        Returns:
            The interpolated energies, one per coordinate ``(n,)``.
        """
        return self._maps_interpn[atom_type](xyz, method=method)

    def energy(
        self,
        nd: Shaped[np.ndarray, "n"],
        ignore_atom_types: str | Sequence[str] | None = None,
        ignore_vdw_hb: bool = False,
        ignore_electrostatic: bool = False,
        ignore_desolvation: bool = False,
        method: str = "linear",
        sum_energies: bool = True,
    ) -> float | Scalars:
        """Return the grid interaction energy of a molecule.

        Args:
            nd: A 1D structured array ``(n,)`` with fields ``i``, ``xyz``, ``q``
                and ``t`` (one record per atom).
            ignore_atom_types: Atom type(s) to ignore.
            ignore_vdw_hb: Ignore the van der Waals / H-bond term.
            ignore_electrostatic: Ignore the electrostatic term.
            ignore_desolvation: Ignore the desolvation term.
            method: Interpolation method (default: ``"linear"``).
            sum_energies: If True, return the summed energy; otherwise an array.

        Returns:
            The total grid energy (float) or per-atom energies (array).
        """
        energies: list = []
        vdw_hb: float | Scalars = 0.0
        elec: float | Scalars = 0.0
        desolv: float | Scalars = 0.0

        if ignore_atom_types is None:
            ignore_atom_types = []
        if not isinstance(ignore_atom_types, (list, tuple)):
            ignore_atom_types = [ignore_atom_types]

        atom_types = np.unique(nd["t"])
        atom_types = set(atom_types).difference(ignore_atom_types)

        for atom_type in atom_types:
            if nd.size > 1:
                index = np.where(nd["t"] == atom_type)[0]
                xyz = nd[index]["xyz"]
            else:
                xyz = nd["xyz"]

            if not ignore_vdw_hb:
                vdw_hb = self._maps_interpn[atom_type](xyz, method=method)

            if not ignore_electrostatic:
                q = nd[index]["q"] if nd.size > 1 else nd["q"]
                elec = self._maps_interpn["Electrostatics"](xyz, method=method) * q

            if not ignore_desolvation:
                desolv = self._maps_interpn["Desolvation"](xyz, method=method)

            energies.extend(vdw_hb + elec + desolv)

        if sum_energies:
            return np.sum(energies)
        return np.array(energies)

    def neighbor_points(self, xyz: Point3, radius: float, min_radius: float = 0.0) -> Coords:
        """Return grid points within a spherical shell around ``xyz``.

        Args:
            xyz: The 3D coordinate ``(3,)`` of the query point.
            radius: Maximum radius of the shell.
            min_radius: Minimum radius of the shell (default: 0).

        Returns:
            A 2D array ``(n, 3)`` of grid-point coordinates.
        """
        coordinates = self._kdtree.data[self._kdtree.query_ball_point(xyz, radius)]

        if min_radius > 0:
            distances = spatial.distance.cdist([xyz], coordinates, "euclidean")[0]
            coordinates = coordinates[distances >= min_radius]

        return coordinates

    def apply_operation_on_maps(
        self,
        names: str | Sequence[str],
        atom_types: str | Sequence[str],
        expression: str,
    ) -> None:
        """Apply a maths expression (in ``x``) to one or more maps.

        Args:
            names: Name(s) of the new or existing maps to write.
            atom_types: Source map(s) whose values become ``x``.
            expression: A Python expression in ``x`` (the source grid values).
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        if not isinstance(atom_types, (list, tuple)):
            atom_types = [atom_types]

        if len(atom_types) != len(names):
            raise MapError("Names and atom_types lengths are not matching.")

        if "x" not in expression:
            logger.error("operation cannot be applied, x is not defined.")
            return

        for name, atom_type in zip(names, atom_types):
            try:
                x = self._maps[atom_type]
                # eval creates a new array, leaving the source untouched.
                x = eval(expression, {"np": np}, {"x": x})

                self._maps[name] = x
                self._maps_interpn[name] = self._generate_affinity_map_interpn(x)
            except Exception:
                logger.warning("This map %s does not exist.", atom_type)
                continue

    def add_bias(self, name: str, coordinates: Coords, bias_value: float, radius: float) -> None:
        """Add a Gaussian energy bias around ``coordinates`` to map ``name``.

        Args:
            name: Name of the existing map to modify.
            coordinates: Bias center(s) ``(n, 3)`` (a single ``(3,)`` point is
                accepted).
            bias_value: Peak energy added at each center.
            radius: Gaussian radius of the bias in Angstrom.

        Raises:
            MapError: If the map ``name`` does not exist.
        """
        coordinates = np.atleast_2d(coordinates)

        try:
            new_map = self._maps[name]
        except KeyError as exc:
            raise MapError(f"Map {name} does not exist. Cannot add energy bias.") from exc

        for coordinate in coordinates:
            sphere_xyz = self.neighbor_points(coordinate, radius)
            indexes = self._cartesian_to_index(sphere_xyz)

            distances = spatial.distance.cdist([coordinate], sphere_xyz, "euclidean")[0]
            bias_energy = bias_value * np.exp(-1.0 * (distances**2) / (radius**2))

            new_map[indexes[:, 0], indexes[:, 1], indexes[:, 2]] += bias_energy

        self._maps[name] = new_map
        self._maps_interpn[name] = self._generate_affinity_map_interpn(new_map)

    def add_mask(
        self, name: str, coordinates: Coords, radius: float, mask_value: float = 999.0
    ) -> None:
        """Mask everything except spheres of ``radius`` around ``coordinates``.

        Args:
            name: Name of the existing map to mask.
            coordinates: Center(s) ``(n, 3)`` of the kept spheres (a single
                ``(3,)`` point is accepted).
            radius: Radius of the kept spheres in Angstrom.
            mask_value: Energy assigned everywhere outside the spheres.

        Raises:
            MapError: If the map ``name`` does not exist.
        """
        coordinates = np.atleast_2d(coordinates)

        new_map = np.zeros(self._npts) + mask_value

        try:
            current_map = self._maps[name]
        except KeyError as exc:
            raise MapError(f"Map {name} does not exist. Cannot add energy mask.") from exc

        for coordinate in coordinates:
            sphere_xyz = self.neighbor_points(coordinate, radius)
            indexes = self._cartesian_to_index(sphere_xyz)

            new_map[indexes[:, 0], indexes[:, 1], indexes[:, 2]] = current_map[
                indexes[:, 0], indexes[:, 1], indexes[:, 2]
            ]

        self._maps[name] = new_map
        self._maps_interpn[name] = self._generate_affinity_map_interpn(new_map)

    def combine(
        self,
        name: str,
        atom_types: str | Sequence[str],
        how: str = "best",
        ad_map: Map | None = None,
    ) -> bool:
        """Combine several maps into one.

        Args:
            name: Name of the new or existing map to write.
            atom_types: Maps to combine.
            how: ``"best"`` (min), ``"add"`` (sum) or ``"replace"``.
            ad_map: Another Map whose values are pulled in (default: None).

        Returns:
            True on success, False if no maps could be combined.
        """
        selected_maps: list[Grid3D] = []
        same_grid = True
        indices = np.index_exp[:, :, :]

        if not isinstance(atom_types, (list, tuple)):
            atom_types = [atom_types]

        if how == "replace":
            if ad_map is None:
                raise MapError("Another map has to be specified for the replace mode.")
            if len(atom_types) != 1:
                raise MapError("Multiple atom types cannot replace the same atom type.")

        if name not in self._maps:
            self._maps[name] = np.zeros(self._npts)

        selected_types = set(self._maps.keys()) & set(atom_types)
        if ad_map is not None:
            selected_types = set(selected_types) & set(ad_map._maps.keys())
        unselected_types = set(selected_types) - set(atom_types)

        if not selected_types:
            logger.warning("no maps were selected from %s list.", atom_types)
            return False
        if unselected_types:
            logger.warning("%s maps can not be combined.", " ".join(unselected_types))

        if ad_map is not None:
            same_grid = all(np.array_equal(x, y) for x, y in zip(self._edges, ad_map._edges))

        if ad_map is not None and same_grid is False:
            # Map the other grid's bounds onto this (reference) grid's indices.
            ix_min, iy_min, iz_min = self._cartesian_to_index(
                [ad_map._xmin, ad_map._ymin, ad_map._zmin]
            )
            ix_max, iy_max, iz_max = self._cartesian_to_index(
                [ad_map._xmax, ad_map._ymax, ad_map._zmax]
            )

            x = self._edges[0][ix_min:ix_max]
            y = self._edges[1][iy_min:iy_max]
            z = self._edges[2][iz_min:iz_max]

            X, Y, Z = np.meshgrid(x, y, z)
            grid = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

            np.clip(
                grid,
                [ad_map._xmin, ad_map._ymin, ad_map._zmin],
                [ad_map._xmax, ad_map._ymax, ad_map._zmax],
                grid,
            )

            indices = np.index_exp[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max]

        for selected_type in selected_types:
            if how != "replace":
                selected_maps.append(self._maps[selected_type][indices])

            if ad_map is not None:
                if same_grid:
                    selected_maps.append(ad_map._maps[selected_type])
                else:
                    energy = ad_map.energy_coordinates(grid, selected_type)
                    energy = np.reshape(energy, (y.shape[0], x.shape[0], z.shape[0]))
                    energy = np.swapaxes(energy, 0, 1)
                    selected_maps.append(energy)

        if selected_types:
            if how == "best":
                self._maps[name][indices] = np.nanmin(selected_maps, axis=0)
            elif how == "add":
                self._maps[name][indices] = np.nansum(selected_maps, axis=0)
            elif how == "replace":
                self._maps[name][indices] = selected_maps[0]

            self._maps_interpn[name] = self._generate_affinity_map_interpn(self._maps[name])

        return True

    def to_pdb(self, fname: str, map_type: str, max_energy: float | None = None) -> None:
        """Export a map as a PDB of pseudo-atoms below ``max_energy``.

        Args:
            fname: Output PDB filename.
            map_type: The map (atom type) to export.
            max_energy: Only grid points with an energy ``<= max_energy`` are
                written.
        """
        idx = np.array(np.where(self._maps[map_type] <= max_energy)).T

        i = 0
        line = "ATOM  %5d  D   DUM Z%4d    %8.3f%8.3f%8.3f  1.00%6.2f           D\n"

        with open(fname, "w") as w:
            for j in range(idx.shape[0]):
                v = self._maps[map_type][idx[j][0], idx[j][1], idx[j][2]]
                v = min(v, 999.99)

                w.write(
                    line
                    % (
                        i,
                        i,
                        self._edges[0][idx[j][0]],
                        self._edges[1][idx[j][1]],
                        self._edges[2][idx[j][2]],
                        v,
                    )
                )
                i += 1

    def to_map(
        self,
        map_types: str | Sequence[str] | None = None,
        prefix: str | None = None,
        grid_parameter_file: str = "grid.gpf",
        grid_data_file: str = "maps.fld",
        macromolecule: str = "molecule.pdbqt",
    ) -> None:
        """Export maps as AutoDock ``.map`` files.

        Args:
            map_types: Map(s) to export (default: all maps).
            prefix: Optional filename prefix for each exported map.
            grid_parameter_file: Value written to the ``GRID_PARAMETER_FILE``
                header field.
            grid_data_file: Value written to the ``GRID_DATA_FILE`` header field.
            macromolecule: Value written to the ``MACROMOLECULE`` header field.
        """
        if map_types is None:
            map_types = list(self._maps.keys())
        elif not isinstance(map_types, (list, tuple)):
            map_types = [map_types]

        for map_type in map_types:
            if map_type in self._maps:
                filename = "%s.map" % map_type
                if prefix is not None:
                    filename = "%s.%s" % (prefix, filename)

                with open(filename, "w") as w:
                    npts = np.array([n if not n % 2 else n - 1 for n in self._npts])

                    w.write("GRID_PARAMETER_FILE %s\n" % grid_parameter_file)
                    w.write("GRID_DATA_FILE %s\n" % grid_data_file)
                    w.write("MACROMOLECULE %s\n" % macromolecule)
                    w.write("SPACING %s\n" % self._spacing)
                    w.write("NELEMENTS %s\n" % " ".join(npts.astype(str)))
                    w.write("CENTER %s\n" % " ".join(self._center.astype(str)))
                    m = np.swapaxes(self._maps[map_type], 0, 2).flatten()
                    w.write("\n".join(m.astype(str)))
            else:
                logger.error("Map %s does not exist.", map_type)


# ---------------------------------------------------------------------------- #
# Pure-Python AutoGrid replacement
# ---------------------------------------------------------------------------- #
class _AutoDockGridForceField:
    """Vectorised AutoDock4 force field used to build affinity grids.

    This is a near-verbatim port of ``waterkit.forcefield.AutoDockForceField``,
    rewritten to evaluate one *probe* atom type against many receptor atoms at
    many grid points at once. It exposes the per-term contributions
    (van der Waals + H-bond, electrostatics, desolvation) needed to assemble the
    separate maps AutoGrid emits.
    """

    def __init__(
        self,
        parameter_file: str | Path,
        smooth: float = 0.5,
        dielectric: float = -0.1465,
        nb_cutoff: float = 12.0,
        hb_cutoff: float = 8.0,
        elec_cutoff: float = 20.0,
        weighted: bool = True,
    ) -> None:
        """Load free-energy weights and atom parameters from ``parameter_file``.

        Args:
            parameter_file: Path to an AutoDock4 parameter file (``AD4_parameters.dat``).
            smooth: AutoDock distance-smoothing window (Angstrom).
            dielectric: Dielectric constant; negative selects the
                distance-dependent (Mehler & Solmajer) dielectric.
            nb_cutoff: Non-bonded cutoff (Angstrom) for the steep van der Waals /
                H-bond terms. Beyond it those terms are negligible (``< 1e-4``
                kcal/mol per pair), so only receptor atoms within ``nb_cutoff`` of
                a grid point are summed -- this matches AutoGrid's ``NBCUTOFF`` and
                is what makes map generation fast. The smooth electrostatic and
                desolvation terms are unaffected (still summed over all atoms).
            hb_cutoff: Distance cutoff for the H-bond term (Angstrom).
            elec_cutoff: Distance cutoff for the electrostatic term (Angstrom).
            weighted: If ``True`` use the file's free-energy coefficients,
                otherwise use unit weights.
        """
        self.smooth = smooth
        self.dielectric = dielectric
        self.nb_cutoff = nb_cutoff
        self.hb_cutoff = hb_cutoff
        self.elec_cutoff = elec_cutoff

        self.weights = self._read_weights(parameter_file, weighted)
        self.atom_par = self._read_atom_parameters(parameter_file)

        # Desolvation constant.
        self.desolvation_k = 0.01097

        # Distance-dependent dielectric (Mehler & Solmajer).
        self.dielectric_epsilon = 78.4
        self.dielectric_A = -8.5525
        self.dielectric_B = self.dielectric_epsilon - self.dielectric_A
        self.dielectric_lambda = 0.003627
        self.dielectric_k = 7.7839
        self.elec_scale = 332.06363

    @staticmethod
    def _read_weights(parameter_file: str | Path, weighted: bool) -> dict[str, float]:
        """Read the ``FE_coeff_*`` free-energy weights.

        Args:
            parameter_file: Path to the AutoDock4 parameter file.
            weighted: If ``True`` keep the file coefficients, else use ``1.0``.

        Returns:
            A dict mapping each weight name to its coefficient.
        """
        weights: dict[str, float] = {}
        with open(parameter_file) as f:
            for line in f.readlines():
                if re.search("^FE_coeff", line):
                    fields = line.split()
                    weight_name = fields[0].split("_")[2]
                    weights[weight_name] = float(fields[-1]) if weighted else 1.0
        return weights

    @staticmethod
    def _read_atom_parameters(parameter_file: str | Path) -> dict[str, dict[str, float]]:
        """Read the ``atom_par`` table keyed by atom type.

        Args:
            parameter_file: Path to the AutoDock4 parameter file.

        Returns:
            A dict mapping each atom type to its per-type parameter dict
            (``rii``, ``epsii``, ``vol``, ``solpar``, ``rij_hb``, ``epsij_hb``,
            ``hbond``).
        """
        atom_par: dict[str, dict[str, float]] = {}
        with open(parameter_file) as f:
            for line in f.readlines():
                if re.search("^atom_par", line):
                    fields = line.split()
                    atom_par[fields[1]] = {
                        "rii": float(fields[2]),
                        "epsii": float(fields[3]),
                        "vol": float(fields[4]),
                        "solpar": float(fields[5]),
                        "rij_hb": float(fields[6]),
                        "epsij_hb": float(fields[7]),
                        "hbond": int(fields[8]),
                    }
        return atom_par

    @staticmethod
    def _coefficient(
        epsilon: Float[np.ndarray, "n_receptor"],
        reqm: Float[np.ndarray, "n_receptor"],
        a: float,
        b: float,
    ) -> Float[np.ndarray, "n_receptor"]:
        """Return the AutoDock n-m coefficient ``(a/|a-b|) * eps * reqm**b``.

        Args:
            epsilon: Per-receptor-atom well depth(s).
            reqm: Per-receptor-atom equilibrium distance(s).
            a: The ``n`` exponent of the n-m potential.
            b: The ``m`` exponent of the n-m potential.

        Returns:
            The per-receptor-atom coefficient.
        """
        return (a / np.abs(a - b)) * epsilon * reqm**b

    def _smooth_distance(
        self, r: Float[np.ndarray, "n_points n_receptor"], reqm: Float[np.ndarray, "n_receptor"]
    ) -> Float[np.ndarray, "n_points n_receptor"]:
        """Apply AutoDock distance smoothing element-wise (per-pair ``reqm``).

        Args:
            r: Grid-point-to-receptor distance matrix ``(n_points, n_receptor)``.
            reqm: Per-receptor-atom equilibrium distance(s).

        Returns:
            The smoothed distance matrix (a copy; ``r`` is not modified).
        """
        if self.smooth <= 0:
            return r

        r = r.copy()
        sf = 0.5 * self.smooth
        reqm_b = np.broadcast_to(np.asarray(reqm)[None, :], r.shape)

        inside = (reqm_b - sf < r) & (r < reqm_b + sf)
        r[inside] = reqm_b[inside]
        r[r >= reqm_b + sf] -= sf
        r[r <= reqm_b - sf] += sf
        return r

    def _smooth_distance_1d(
        self, r: Float[np.ndarray, "n_pairs"], reqm: Float[np.ndarray, "n_pairs"]
    ) -> Float[np.ndarray, "n_pairs"]:
        """Apply AutoDock distance smoothing to a flat 1D array of pair distances.

        Identical maths to :meth:`_smooth_distance` but for the sparse
        (within-cutoff) pair list, where ``r`` and ``reqm`` are aligned 1D arrays.

        Args:
            r: Per-pair distances ``(n_pairs,)``.
            reqm: Per-pair equilibrium distances ``(n_pairs,)``.

        Returns:
            The smoothed distances (a copy; ``r`` is not modified).
        """
        if self.smooth <= 0:
            return r

        r = r.copy()
        sf = 0.5 * self.smooth
        inside = (reqm - sf < r) & (r < reqm + sf)
        r[inside] = reqm[inside]
        r[r >= reqm + sf] -= sf
        r[r <= reqm - sf] += sf
        return r

    def _distance_dependent_dielectric(
        self, r: Float[np.ndarray, "n_points n_receptor"]
    ) -> Float[np.ndarray, "n_points n_receptor"]:
        """Distance-dependent dielectric (Mehler & Solmajer).

        Applied element-wise to the distance matrix.

        Args:
            r: Grid-point-to-receptor distance matrix ``(n_points, n_receptor)``.

        Returns:
            The dielectric at each distance, same shape as ``r``.
        """
        lambda_b = -self.dielectric_lambda * self.dielectric_B
        return self.dielectric_A + self.dielectric_B / (
            1.0 + self.dielectric_k * np.exp(lambda_b * r)
        )

    def receptor_parameters(
        self, types: Sequence[str]
    ) -> dict[str, Float[np.ndarray, "n_receptor"]]:
        """Return per-atom parameter arrays for a list of receptor atom types.

        Args:
            types: The receptor atom types, one per receptor atom.

        Returns:
            A dict mapping each parameter name (``rii``, ``epsii``, ``vol``,
            ``solpar``, ``rij_hb``, ``epsij_hb``, ``hbond``) to a
            ``(n_receptor,)`` array.

        Raises:
            MapError: If a type is not present in the parameter file.
        """
        missing = sorted({t for t in types if t not in self.atom_par})
        if missing:
            raise MapError(f"Atom type(s) {missing} not found in the parameter file.")

        keys = ("rii", "epsii", "vol", "solpar", "rij_hb", "epsij_hb", "hbond")
        return {k: np.array([self.atom_par[t][k] for t in types], dtype=float) for k in keys}

    def atom_map_term(
        self,
        probe_type: str,
        distances: Float[np.ndarray, "n_points n_receptor"],
        rec_params: dict[str, Float[np.ndarray, "n_receptor"]],
        rec_q: Float[np.ndarray, "n_receptor"],
        near: tuple[Indices, Indices, Scalars] | None = None,
    ) -> Float[np.ndarray, "n_points"]:
        """Return the vdW + H-bond + (type-dependent) desolvation contribution.

        Args:
            probe_type: The ligand probe atom type.
            distances: ``(n_points, n_receptor)`` distance matrix.
            rec_params: Per-receptor-atom parameter arrays.
            rec_q: Per-receptor-atom partial charges ``(n_receptor,)``.
            near: Optional precomputed within-:attr:`nb_cutoff` pairs as
                ``(rows, cols, dists)`` flat arrays (the indices into the
                ``distances`` matrix and their distances). Passing this avoids
                recomputing the ``distances <= nb_cutoff`` scan for every probe
                type when several maps share one distance block.

        Returns:
            The summed contribution for each grid point ``(n_points,)``.

        Note:
            The steep van der Waals (12-6) and H-bond (12-10) terms are summed
            only over receptor atoms within :attr:`nb_cutoff` of each grid point
            (their contribution is negligible beyond it). Those pairs are selected
            from ``distances`` and the per-pair contributions scattered back with
            :func:`numpy.bincount`, which avoids evaluating the expensive powers
            over the (mostly far) full matrix. The smooth desolvation cross term
            is still summed over every receptor atom.
        """
        probe = self.atom_par[probe_type]
        n_points = distances.shape[0]
        result = np.zeros(n_points)

        w_vdw = self.weights.get("vdW", 0.0)
        w_hb = self.weights.get("hbond", 0.0)

        # --- van der Waals (12-6) + H-bond (12-10): near pairs only ---
        if w_vdw > 0 or w_hb > 0:
            if near is None:
                rows, cols = np.nonzero(distances <= self.nb_cutoff)
                dd = distances[rows, cols]
            else:
                rows, cols, dd = near
            if rows.size:
                if w_vdw > 0:
                    vdw_rij = (probe["rii"] + rec_params["rii"][cols]) / 2.0
                    vdw_eps = np.sqrt(probe["epsii"] * rec_params["epsii"][cols])
                    A = self._coefficient(vdw_eps, vdw_rij, 6, 12)
                    B = self._coefficient(vdw_eps, vdw_rij, 12, 6)
                    r = self._smooth_distance_1d(dd, vdw_rij)
                    inv = 1.0 / np.maximum(r, _DISTANCE_FLOOR)
                    pair = A * inv**12 - B * inv**6
                    result += w_vdw * np.bincount(rows, weights=pair, minlength=n_points)

                if w_hb > 0:
                    cols_par = {k: rec_params[k][cols] for k in ("hbond", "rij_hb", "epsij_hb")}
                    hb_rij, hb_eps = self._hbond_parameters(probe, cols_par)
                    C = self._coefficient(hb_eps, hb_rij, 10, 12)
                    D = self._coefficient(hb_eps, hb_rij, 12, 10)
                    active = (C != 0) & (dd <= self.hb_cutoff)
                    if np.any(active):
                        r = self._smooth_distance_1d(dd[active], hb_rij[active])
                        inv = 1.0 / np.maximum(r, _DISTANCE_FLOOR)
                        pair = C[active] * inv**12 - D[active] * inv**10
                        result += w_hb * np.bincount(rows[active], weights=pair, minlength=n_points)

        # --- type-dependent desolvation cross terms (smooth; all atoms) ---
        if self.weights.get("desolv", 0.0) > 0:
            cross = (
                probe["solpar"] * rec_params["vol"]
                + (rec_params["solpar"] + self.desolvation_k * np.abs(rec_q)) * probe["vol"]
            )
            if np.any(cross):
                gauss = np.exp(-0.5 * (distances**2 / _DESOLV_SIGMA**2))
                result += self.weights["desolv"] * (cross[None, :] * gauss).sum(axis=1)

        return result

    @staticmethod
    def _hbond_parameters(
        probe: dict[str, float], rec_params: dict[str, Float[np.ndarray, "n_receptor"]]
    ) -> tuple[Float[np.ndarray, "n_receptor"], Float[np.ndarray, "n_receptor"]]:
        """Return the per-receptor-atom H-bond ``(reqm, eps)`` for ``probe``.

        Args:
            probe: The probe atom's parameter dict.
            rec_params: Per-receptor-atom parameter arrays.

        Returns:
            A 2-tuple ``(hb_rij, hb_eps)`` of ``(n_receptor,)`` arrays, non-zero
            only for receptor atoms that form an H-bond with the probe.
        """
        n = rec_params["hbond"].shape[0]
        hb_rij = np.zeros(n)
        hb_eps = np.zeros(n)
        rec_hb = rec_params["hbond"]

        if probe["hbond"] in (1, 2):  # probe is a donor
            acceptors = np.isin(rec_hb, [3, 4, 5])
            hb_rij[acceptors] = rec_params["rij_hb"][acceptors]
            hb_eps[acceptors] = rec_params["epsij_hb"][acceptors]
        elif probe["hbond"] in (3, 4, 5):  # probe is an acceptor
            donors = np.isin(rec_hb, [1, 2])
            hb_rij[donors] = probe["rij_hb"]
            hb_eps[donors] = probe["epsij_hb"]

        return hb_rij, hb_eps

    def electrostatic_term(
        self,
        distances: Float[np.ndarray, "n_points n_receptor"],
        rec_q: Float[np.ndarray, "n_receptor"],
    ) -> Float[np.ndarray, "n_points"]:
        """Return the electrostatic potential per unit charge at each grid point.

        Args:
            distances: ``(n_points, n_receptor)`` distance matrix.
            rec_q: Per-receptor-atom partial charges ``(n_receptor,)``.

        Returns:
            The electrostatic potential per unit charge at each grid point
            ``(n_points,)``.
        """
        w = self.weights.get("estat", 0.0)
        if w == 0:
            return np.zeros(distances.shape[0])

        r = np.maximum(distances, _DISTANCE_FLOOR)
        if self.dielectric < 0:
            eps_r = self._distance_dependent_dielectric(r)
        else:
            eps_r = self.dielectric

        contrib = rec_q[None, :] / (r * eps_r)
        contrib[distances > self.elec_cutoff] = 0.0
        return self.elec_scale * w * contrib.sum(axis=1)

    def desolvation_term(
        self,
        distances: Float[np.ndarray, "n_points n_receptor"],
        rec_vol: Float[np.ndarray, "n_receptor"],
    ) -> Float[np.ndarray, "n_points"]:
        """Return the (charge-independent) desolvation map value at each grid point.

        Args:
            distances: ``(n_points, n_receptor)`` distance matrix.
            rec_vol: Per-receptor-atom atomic volumes ``(n_receptor,)``.

        Returns:
            The desolvation map value at each grid point ``(n_points,)``.
        """
        w = self.weights.get("desolv", 0.0)
        if w == 0:
            return np.zeros(distances.shape[0])

        gauss = np.exp(-0.5 * (distances**2 / _DESOLV_SIGMA**2))
        return w * self.desolvation_k * (rec_vol[None, :] * gauss).sum(axis=1)


# Minimum grid points before map generation is split across processes.
_MAP_PARALLEL_MIN_POINTS = 8000

# Shared, read-only inputs for parallel map generation; the parent fills this in
# before forking the worker pool, so workers inherit it via copy-on-write.
_MAP_WORKER_CTX: dict = {}


def _compute_map_arrays(
    grid_points: Coords,
    rec_xyz: Coords,
    rec_params: dict[str, Float[np.ndarray, "n_receptor"]],
    rec_q: Float[np.ndarray, "n_receptor"],
    forcefield: _AutoDockGridForceField,
    atom_types: Sequence[str],
    block: int,
) -> tuple[dict[str, Scalars], Scalars, Scalars]:
    """Evaluate every map over ``grid_points`` in memory-bounded blocks.

    The grid-point-to-receptor distance matrix and the within-``nb_cutoff`` pair
    list are computed once per block and shared across all atom-type maps, the
    electrostatics map and the desolvation map (no quantity is recomputed).

    Args:
        grid_points: ``(N, 3)`` grid coordinates.
        rec_xyz: ``(n_receptor, 3)`` receptor coordinates.
        rec_params: Per-receptor-atom parameter arrays.
        rec_q: Per-receptor-atom partial charges ``(n_receptor,)``.
        forcefield: The grid force field.
        atom_types: Ligand probe atom types to build maps for.
        block: Number of grid points per distance-matrix block.

    Returns:
        A 3-tuple ``(flats, elec, desolv)`` where ``flats`` maps each atom type
        to its ``(N,)`` affinity array and ``elec``/``desolv`` are the ``(N,)``
        electrostatics and desolvation arrays.
    """
    n = grid_points.shape[0]
    flats = {atom_type: np.empty(n) for atom_type in atom_types}
    elec = np.empty(n)
    desolv = np.empty(n)

    for start in range(0, n, block):
        stop = min(start + block, n)
        d = spatial.distance.cdist(grid_points[start:stop], rec_xyz)
        # Within-cutoff pairs, computed once and shared by all atom maps.
        rows, cols = np.nonzero(d <= forcefield.nb_cutoff)
        near = (rows, cols, d[rows, cols])
        for atom_type in atom_types:
            flats[atom_type][start:stop] = forcefield.atom_map_term(
                atom_type, d, rec_params, rec_q, near=near
            )
        elec[start:stop] = forcefield.electrostatic_term(d, rec_q)
        desolv[start:stop] = forcefield.desolvation_term(d, rec_params["vol"])

    return flats, elec, desolv


def _map_worker_range(rng: tuple[int, int]) -> tuple[dict[str, Scalars], Scalars, Scalars]:
    """Map-generation pool worker: evaluate the maps for one grid-point range.

    Reads the shared receptor field / grid from :data:`_MAP_WORKER_CTX`, which
    the worker inherited from the parent process at fork time.

    Args:
        rng: The ``(start, stop)`` grid-point range handled by this worker.

    Returns:
        The ``(flats, elec, desolv)`` arrays for the range (see
        :func:`_compute_map_arrays`).
    """
    start, stop = rng
    ctx = _MAP_WORKER_CTX
    return _compute_map_arrays(
        ctx["grid_points"][start:stop],
        ctx["rec_xyz"],
        ctx["rec_params"],
        ctx["rec_q"],
        ctx["forcefield"],
        ctx["atom_types"],
        ctx["block"],
    )


def generate_maps(
    receptor: Molecule,
    atom_types: str | Sequence[str],
    box_center: Float[np.ndarray, "3"],
    box_size: Float[np.ndarray, "3"],
    spacing: float = 0.375,
    forcefield: _AutoDockGridForceField | str | Path | None = None,
    n_jobs: int = 1,
) -> Map:
    """Generate AutoDock affinity grids in pure Python (an AutoGrid replacement).

    Mirrors the role of ``waterkit.autogrid.AutoGrid.run``: it lays out the same
    box (``npts`` rounded up to an even number of voxels, i.e. an odd number of
    grid points) and evaluates the AutoDock4 force field on every grid point. It
    produces one affinity map per entry of ``atom_types`` (van der Waals +
    H-bond + type-dependent desolvation, exactly as AutoGrid bakes into its atom
    maps) plus an ``Electrostatics`` map (potential per unit charge) and a
    ``Desolvation`` map. The returned :class:`Map` is structurally identical to
    one built by :meth:`Map.from_fld`.

    Args:
        receptor: A :class:`~waterkit.core.molecule.Molecule`. Its
            ``coordinates()``, ``atom_types()`` and ``partial_charges()`` define
            the receptor field.
        atom_types: The ligand atom types to build maps for.
        box_center: 3D center of the grid box (Angstrom).
        box_size: 3D size of the grid box (Angstrom).
        spacing: Grid spacing (default: 0.375 Angstrom).
        forcefield: An :class:`_AutoDockGridForceField`, a path to a parameter
            file, or ``None`` (uses the packaged ``AD4_parameters.dat``).
        n_jobs: Number of worker processes to split the grid over (``-1`` = all
            cores, ``1`` = serial). Grid points are partitioned across a fork
            pool; the result is independent of ``n_jobs``.

    Returns:
        The populated :class:`Map`.

    Performance:
        Cost scales with ``n_grid_points * n_receptor_atoms``. The steep
        van der Waals / H-bond terms are summed only over receptor atoms within
        ``nb_cutoff`` (the dominant saving), the distance matrix is computed once
        per block and shared across maps, and the grid is evaluated in parallel
        across ``n_jobs`` processes.
    """
    box_center = np.asarray(box_center, dtype=float)
    box_size = np.asarray(box_size, dtype=float)

    if box_center.shape[0] != 3:
        raise MapError("The center of the box must be defined by 3 coordinates (x, y, z).")
    if box_size.shape[0] != 3:
        raise MapError("The size of the box must be defined by 3 coordinates (x, y, z).")

    if not isinstance(atom_types, (list, tuple)):
        atom_types = [atom_types]

    if forcefield is None:
        forcefield = _AutoDockGridForceField(data_path("AD4_parameters.dat"))
    elif isinstance(forcefield, (str, Path)):
        forcefield = _AutoDockGridForceField(forcefield)

    # Number of voxels (even), then number of grid points (odd), as AutoGrid does.
    nelements = np.ceil(box_size / spacing).astype(int)
    nelements[nelements % 2 == 1] += 1
    npts = nelements + 1

    ad_map = Map._create_empty(npts, spacing, box_center)

    # Receptor field.
    rec_xyz = np.asarray(receptor.coordinates(), dtype=float)
    rec_q = np.asarray(receptor.partial_charges(), dtype=float)
    rec_types = receptor.atom_types()
    rec_params = forcefield.receptor_parameters(rec_types)
    n_receptor = rec_xyz.shape[0]

    # Grid points in [ix, iy, iz] (C-order) to match the interpolator's edges.
    x, y, z = ad_map._edges
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.stack((gx.ravel(), gy.ravel(), gz.ravel()), axis=-1)

    logger.info(
        "Generating %d maps over %d grid points against %d receptor atoms.",
        len(atom_types) + 2,
        grid_points.shape[0],
        n_receptor,
    )

    def _clamp(values: Scalars) -> Scalars:
        """Replace non-finite energies and clamp to ``+/-ENERGY_CLAMP``."""
        values = np.nan_to_num(values, nan=ENERGY_CLAMP, posinf=ENERGY_CLAMP, neginf=-ENERGY_CLAMP)
        return np.clip(values, -ENERGY_CLAMP, ENERGY_CLAMP)

    n = grid_points.shape[0]
    # Keep the (block x receptor) distance matrix near ~4M elements.
    block = min(max(256, int(4_000_000 / max(n_receptor, 1))), 20000)

    import multiprocessing as mp

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    use_fork = n_jobs > 1 and n >= _MAP_PARALLEL_MIN_POINTS and "fork" in mp.get_all_start_methods()

    if not use_fork:
        flats, elec_flat, desolv_flat = _compute_map_arrays(
            grid_points, rec_xyz, rec_params, rec_q, forcefield, atom_types, block
        )
    else:
        ranges = [
            (int(chunk[0]), int(chunk[-1]) + 1)
            for chunk in np.array_split(np.arange(n), n_jobs)
            if chunk.size
        ]
        # Publish read-only inputs so the forked workers inherit them (no pickling
        # of the grid/receptor); clear them again afterwards.
        _MAP_WORKER_CTX.update(
            grid_points=grid_points,
            rec_xyz=rec_xyz,
            rec_params=rec_params,
            rec_q=rec_q,
            forcefield=forcefield,
            atom_types=atom_types,
            block=block,
        )
        try:
            with mp.get_context("fork").Pool(len(ranges)) as pool:
                parts = pool.map(_map_worker_range, ranges)
        finally:
            _MAP_WORKER_CTX.clear()
        flats = {
            atom_type: np.concatenate([part[0][atom_type] for part in parts])
            for atom_type in atom_types
        }
        elec_flat = np.concatenate([part[1] for part in parts])
        desolv_flat = np.concatenate([part[2] for part in parts])

    for atom_type in atom_types:
        ad_map.add_map(atom_type, _clamp(flats[atom_type]).reshape(npts))
        logger.info("  %s map done.", atom_type)
    ad_map.add_map("Electrostatics", _clamp(elec_flat).reshape(npts))
    logger.info("  Electrostatics map done.")
    ad_map.add_map("Desolvation", _clamp(desolv_flat).reshape(npts))
    logger.info("  Desolvation map done.")

    return ad_map


# ---------------------------------------------------------------------------- #
# Spherical water map
# ---------------------------------------------------------------------------- #
def _water_grid_calculation(
    xyzs: Coords,
    ad_map: Map,
    atom_types: Sequence[str],
    temperature: float,
    water_orientations: Float[np.ndarray, "n_orient n_satellite 3"],
    verbose: bool = False,
) -> Scalars:
    """Compute the Boltzmann-averaged water energy at each grid point.

    For every grid point not too close to the box edge, the oxygen energy is
    read from the first water map and each pre-generated orientation adds the
    energy of its satellite atoms (hydrogens / lone pairs). The point energy is
    the Boltzmann average over orientations when any orientation is favourable,
    otherwise the plain mean (a clash inside the receptor).

    Args:
        xyzs: ``(N, 3)`` grid coordinates.
        ad_map: The receptor map providing per-atom-type energies.
        atom_types: Water atom types, oxygen first then satellites.
        temperature: Temperature in Kelvin.
        water_orientations: ``(n_orient, n_satellite, 3)`` offsets.
        verbose: If True, log progress.

    Returns:
        ``(N,)`` array of per-grid-point water energies.
    """
    energy = np.zeros(shape=(xyzs.shape[0]))

    # Avoid edge effects by skipping grid points too close to the box edge.
    not_close_to_edges = ~ad_map.is_close_to_edge(xyzs, distance=1.0)

    energy[not_close_to_edges] = ad_map.energy_coordinates(xyzs[not_close_to_edges], atom_types[0])

    n = 0
    n_total = np.sum(not_close_to_edges)

    for i in np.where(not_close_to_edges)[0]:
        tmp_energy = energy[i]
        new_orientations = xyzs[i] + water_orientations

        # Add the energy of each satellite atom across all orientations.
        for j, atom_type in enumerate(atom_types[1:]):
            tmp_energy += ad_map.energy_coordinates(new_orientations[:, j], atom_type)

        # inf values mean an orientation pokes outside the box; drop them.
        tmp_energy = tmp_energy[tmp_energy != np.inf]

        if np.any(tmp_energy < 0):
            p = boltzmann_probabilities(tmp_energy, temperature)
            energy[i] = np.sum(tmp_energy * p)
        else:
            energy[i] = np.mean(tmp_energy)

        if verbose and (n % 100 == 0) and n_total > 0:
            logger.info("Grid points calculated: %5.2f / 100 %%", float(n) / n_total * 100.0)

        n += 1

    return energy


# Minimum number of grid points before the SW build is split across processes
# (below this the multiprocessing setup costs more than it saves).
_SW_PARALLEL_MIN_POINTS = 2000

# Shared, read-only inputs for the parallel spherical-water calculation. The
# parent fills this in before forking the worker pool; the forked workers inherit
# it via copy-on-write, so the (large) receptor map is never pickled or re-built.
_SW_WORKER_CTX: dict = {}


def _sw_worker_chunk(xyzs_chunk: Coords) -> Scalars:
    """Multiprocessing pool worker: compute SW energies for one chunk of points.

    Reads the shared receptor map / orientation set from :data:`_SW_WORKER_CTX`,
    which the worker inherited from the parent process at fork time.

    Args:
        xyzs_chunk: ``(m, 3)`` grid coordinates handled by this worker.

    Returns:
        ``(m,)`` per-grid-point water energies for the chunk.
    """
    ctx = _SW_WORKER_CTX
    return _water_grid_calculation(
        xyzs_chunk,
        ctx["ad_map"],
        ctx["atom_types"],
        ctx["temperature"],
        ctx["water_orientations"],
    )


class SphericalWaterMap:
    """Builds the Boltzmann-averaged spherical-water ("SW") affinity map.

    The map is computed by averaging the energy of many pre-generated water
    orientations (only the satellite atoms vary) at every grid point, on top of
    the oxygen affinity map. This is by far the most expensive step in a run, so
    it is sped up three ways: the grid points are split across ``n_jobs`` worker
    processes, the orientation set can be subsampled via ``n_orientations``, and
    the finished map can be cached to disk (keyed on the receptor maps, box,
    temperature and orientation count) when ``cache_dir`` is given.
    """

    def __init__(
        self,
        water_model: str = "tip3p",
        temperature: float = 300.0,
        n_jobs: int = -1,
        verbose: bool = False,
        n_orientations: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize the spherical-water map builder.

        Args:
            water_model: ``"tip3p"`` or ``"tip5p"``.
            temperature: Temperature in Kelvin.
            n_jobs: Number of worker processes to split the grid over
                (``-1`` = all cores). The build is parallel across grid points.
            verbose: If True, log progress.
            n_orientations: If set, deterministically subsample this many of the
                pre-generated water orientations (evenly strided) instead of using
                all of them. Fewer orientations trade a little accuracy for a
                roughly linear speedup. ``None`` (default) uses every orientation
                and reproduces the original result exactly.
            cache_dir: If given, read/write the finished SW map as a compressed
                ``.npz`` in this directory, keyed on a content hash of the
                receptor maps, box geometry, temperature and orientation count.
                ``None`` (default) disables caching.

        Raises:
            MapError: If ``water_model`` is unknown.
        """
        self._temperature = temperature
        self._water_model = water_model
        self._verbose = verbose
        self._n_orientations = n_orientations
        self._cache_dir = cache_dir

        if n_jobs == -1:
            self._n_jobs = os.cpu_count() or 1
        else:
            self._n_jobs = n_jobs

        if water_model == "tip3p":
            self._atom_types = ["OW", "HW", "HW"]
        elif water_model == "tip5p":
            self._atom_types = ["OT", "HT", "HT", "LP", "LP"]
        else:
            raise MapError(f"Water model {water_model} unknown.")

        # Load pre-generated water orientations (satellite-atom positions only).
        n_atoms = 2
        usecols = [0, 1, 2, 3, 4, 5]
        if water_model == "tip5p":
            usecols += [6, 7, 8, 9, 10, 11]
            n_atoms += 2

        w_orientation_file = data_path("water_orientations.txt")
        water_orientations = np.loadtxt(w_orientation_file, usecols=usecols)
        shape = (water_orientations.shape[0], n_atoms, 3)
        self._water_orientations = water_orientations.reshape(shape)

    def run(self, ad_map: Map, name: str = "SW") -> bool:
        """Compute and add the SW map onto ``ad_map`` in place.

        Args:
            ad_map: The receptor map to add the SW map to.
            name: Name of the new map (default: ``"SW"``).

        Returns:
            True on success, False if a map called ``name`` already exists.
        """
        if name in ad_map._maps:
            logger.error("map %s already exists.", name)
            return False

        water_orientations = self._select_orientations()
        xyzs = ad_map._kdtree.data

        # Fast path: return a cached SW map if one matches this exact input.
        cache_path = None
        if self._cache_dir is not None:
            key = self._cache_key(ad_map, water_orientations)
            cache_path = Path(self._cache_dir) / f"sw_{key}.npz"
            cached = self._load_sw_cache(cache_path, ad_map._npts)
            if cached is not None:
                logger.debug("Loaded cached SW map from %s", cache_path)
                ad_map.add_map(name, cached)
                return True

        m_copy = ad_map.copy()
        results = self._compute(m_copy, xyzs, water_orientations)

        # Replace NaN values with the maximum energy.
        max_energy = np.nanmax(results)
        results[np.isnan(results)] = max_energy
        # Reshape back to a 3D grid (matching the kdtree point ordering).
        # The kdtree points are built with np.meshgrid(x, y, z) (default 'xy'
        # indexing), so the flat order is (ny, nx, nz); reshape to that, then
        # swapaxes(0, 1) -> (nx, ny, nz) == _npts, matching the affinity maps.
        npts = ad_map._npts
        new_map = np.swapaxes(results.reshape((npts[1], npts[0], npts[2])), 0, 1)

        if cache_path is not None:
            self._save_sw_cache(cache_path, new_map)

        ad_map.add_map(name, new_map)

        return True

    def _select_orientations(self) -> Float[np.ndarray, "n_orient n_satellite 3"]:
        """Return the orientation set to use, subsampled if ``n_orientations`` is set.

        Returns:
            All pre-generated orientations when ``n_orientations`` is ``None`` (or
            not smaller than the total), otherwise an evenly-strided,
            deterministic subset of ``n_orientations`` orientations.
        """
        orientations = self._water_orientations
        n_total = orientations.shape[0]
        if self._n_orientations is None or self._n_orientations >= n_total:
            return orientations
        n = max(1, int(self._n_orientations))
        idx = np.unique(np.linspace(0, n_total - 1, n).round().astype(int))
        return orientations[idx]

    def _compute(
        self,
        ad_map: Map,
        xyzs: Coords,
        water_orientations: Float[np.ndarray, "n_orient n_satellite 3"],
    ) -> Scalars:
        """Evaluate the per-grid-point SW energies, in parallel when worthwhile.

        Splits the grid points into ``n_jobs`` contiguous chunks and runs
        :func:`_water_grid_calculation` on each in a worker process, falling back
        to a single serial call for one job or small grids. Chunk results are
        concatenated in grid-point order, so the result is identical to the
        serial computation.

        Args:
            ad_map: The receptor map providing per-atom-type energies.
            xyzs: ``(N, 3)`` grid coordinates.
            water_orientations: ``(n_orient, n_satellite, 3)`` satellite offsets.

        Returns:
            ``(N,)`` per-grid-point water energies.
        """
        import multiprocessing as mp

        serial = (
            self._n_jobs <= 1
            or xyzs.shape[0] < _SW_PARALLEL_MIN_POINTS
            # Workers inherit the receptor map via fork (copy-on-write). Without
            # fork (e.g. Windows) a spawn pool would re-import the package and
            # re-pickle the map per worker, which is slower than serial here.
            or "fork" not in mp.get_all_start_methods()
        )
        if serial:
            return _water_grid_calculation(
                xyzs, ad_map, self._atom_types, self._temperature, water_orientations, self._verbose
            )

        n_jobs = min(self._n_jobs, xyzs.shape[0])
        chunks = [chunk for chunk in np.array_split(xyzs, n_jobs) if chunk.shape[0] > 0]

        # Publish the read-only inputs so the forked workers inherit them (no
        # pickling of the large receptor map); clear them again afterwards.
        _SW_WORKER_CTX.update(
            ad_map=ad_map,
            atom_types=self._atom_types,
            temperature=self._temperature,
            water_orientations=water_orientations,
        )
        try:
            with mp.get_context("fork").Pool(len(chunks)) as pool:
                parts = pool.map(_sw_worker_chunk, chunks)
        finally:
            _SW_WORKER_CTX.clear()
        return np.concatenate(parts)

    def _cache_key(
        self, ad_map: Map, water_orientations: Float[np.ndarray, "n_orient n_satellite 3"]
    ) -> str:
        """Return a content hash identifying this SW computation.

        The key changes whenever any input that affects the SW map changes: the
        oxygen/satellite receptor maps, the box geometry, the temperature or the
        number of orientations. This makes the on-disk cache safe across
        different receptors and boxes.

        Args:
            ad_map: The receptor map the SW map is built from.
            water_orientations: The (possibly subsampled) orientation set.

        Returns:
            A short hexadecimal digest.
        """
        h = hashlib.sha1()
        for atom_type in self._atom_types:
            if atom_type in ad_map._maps:
                h.update(np.ascontiguousarray(ad_map._maps[atom_type], dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(ad_map._center, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(ad_map._npts, dtype=np.int64).tobytes())
        h.update(np.float64(ad_map._spacing).tobytes())
        h.update(np.float64(self._temperature).tobytes())
        h.update(np.int64(water_orientations.shape[0]).tobytes())
        h.update("|".join(self._atom_types).encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _load_sw_cache(cache_path: Path, expected_npts: Int[np.ndarray, "3"]) -> Grid3D | None:
        """Load a cached SW grid if present and shape-compatible, else ``None``.

        Args:
            cache_path: Path to the ``.npz`` cache file.
            expected_npts: Expected grid shape ``(nx, ny, nz)``.

        Returns:
            The cached SW grid, or ``None`` if missing, unreadable or mismatched.
        """
        if not cache_path.exists():
            return None
        try:
            with np.load(cache_path) as data:
                sw = data["SW"]
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("Ignoring unreadable SW-map cache %s: %s", cache_path, exc)
            return None
        if not np.array_equal(sw.shape, expected_npts):
            return None
        return sw

    @staticmethod
    def _save_sw_cache(cache_path: Path, sw: Grid3D) -> None:
        """Write the SW grid to the cache (best effort; failures are logged).

        Args:
            cache_path: Destination ``.npz`` path.
            sw: The finished SW grid to cache.
        """
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, SW=sw)
            logger.debug("Cached SW map to %s", cache_path)
        except OSError as exc:
            logger.warning("Could not cache SW map to %s: %s", cache_path, exc)
