"""Hydration-site identification from GIST density maps.

Port of ``waterkit.analysis.hydration_sites``. Identifies hydration sites from
an oxygen density map (gO) by iteratively picking the highest-density voxel
while enforcing a minimum inter-site distance, then assigns Gaussian-smoothed
energies from a GIST thermodynamic map. Reads ``.dx`` grids via
``griddataformats`` and exports sites to PDB.

``griddataformats`` is an optional dependency: it is imported lazily inside the
methods that need it, so this module imports cleanly without it installed and
only fails (with :class:`~waterkit.utils.exceptions.WaterKitError`) when such a
method is actually called.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree, distance

from ..utils.exceptions import WaterKitError
from ..utils.typing import Coords, Float, Indices, Scalars, Vec3

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gridData import Grid

logger = logging.getLogger("waterkit.analysis.hydration_sites")


def _require_grid() -> type:
    """Return the ``gridData.Grid`` class, importing ``griddataformats`` lazily.

    Returns:
        The ``gridData.Grid`` class.

    Raises:
        WaterKitError: If ``griddataformats`` is not installed.
    """
    try:
        from gridData import Grid
    except ImportError as exc:
        raise WaterKitError(
            "griddataformats is required for analysis; install with `pip install griddataformats`"
        ) from exc
    return Grid


def _gaussian_weights(center: Vec3, grid_points: Coords, sigma: float) -> Scalars:
    """Return Gaussian weights for ``grid_points`` based on distance from ``center``.

    Args:
        center: ``(3,)`` coordinates of the Gaussian center.
        grid_points: ``(N, 3)`` array of grid-point coordinates.
        sigma: Standard deviation of the Gaussian.

    Returns:
        Array of ``N`` weights (shape ``(N,)``), one per grid point.
    """
    d = distance.cdist([center], grid_points, "euclidean")[0]
    # Gaussian weight filtering based on distance from the center
    weights = np.exp(-(d**2) / (2.0 * sigma**2))
    return weights


def _coordinates_from_grid(grid: Grid) -> Coords:
    """Return the ``(N, 3)`` coordinates of every voxel midpoint in ``grid``.

    Args:
        grid: A ``gridData.Grid`` object.

    Returns:
        ``(N, 3)`` array of grid-point coordinates.
    """
    x, y, z = grid.midpoints
    X, Y, Z = np.meshgrid(x, y, z)
    grid_coordinates = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    return grid_coordinates


def _hydration_sites(
    coordinates: Coords,
    values: Scalars,
    density: float = 2,
    min_cutoff: float = 1.4,
    max_cutoff: float = 2.6,
) -> tuple[Coords, Coords, Indices]:
    """Cluster high-density voxels into hydration sites.

    Iteratively picks the highest-density remaining voxel as a new site center,
    then removes every voxel within ``max_cutoff`` from the pool while recording
    the voxels within ``min_cutoff`` as the site's isocontour.

    Args:
        coordinates: ``(N, 3)`` voxel coordinates.
        values: ``(N,)`` density value for each voxel.
        density: Minimum density a voxel must have to be considered.
        min_cutoff: Radius defining the isocontour around each center.
        max_cutoff: Minimum allowed distance between two hydration sites.

    Returns:
        Tuple of ``(centers, isocontour, labels)`` where ``centers`` are the
        ``(M, 3)`` hydration-site coordinates, ``isocontour`` the ``(K, 3)``
        clustered voxels and ``labels`` the ``(K,)`` integer cluster id of each
        isocontour point.
    """
    # Keep coordinates with a certain density
    tmp_coordinates = coordinates[values >= density]
    tmp_values = values[values >= density]
    mask = np.ones(tmp_values.shape, dtype=bool)

    centers: list = []
    isocontour: list = []
    labels: list = []
    i = 1

    while mask.any():
        center_idx = np.argmax(tmp_values)

        d = distance.cdist([tmp_coordinates[center_idx]], tmp_coordinates, "euclidean")[0]
        close_points = np.where((d <= max_cutoff) & mask)[0]
        extra_close_points = np.where((d <= min_cutoff) & mask)[0]

        # Add center
        centers.extend([tmp_coordinates[center_idx]])
        isocontour.extend([tmp_coordinates[center_idx]])
        labels.extend([i])

        # Remove center from the pool
        mask[center_idx] = False
        tmp_values[center_idx] = -1

        # Add the closest points and remove them from the pool
        if close_points.size > 0:
            isocontour.extend([tmp_coordinates[extra_close_points]])
            labels.extend([i] * extra_close_points.shape[0])

            mask[close_points] = False
            tmp_values[close_points] = -1.0

        i += 1

    isocontour = np.vstack(isocontour)
    centers = np.vstack(centers)
    labels = np.array(labels)

    return centers, isocontour, labels


def _objective_function(xyzs: Float[np.ndarray, "3*m"], grid: Grid) -> float:
    """Return the negative total interpolated density at the points ``xyzs``.

    Args:
        xyzs: Flattened ``(3 * M,)`` array of ``M`` candidate positions, reshaped
            internally to ``(M, 3)``.
        grid: A ``gridData.Grid`` object to interpolate.

    Returns:
        The negative summed interpolated density (for minimization).
    """
    xyzs = xyzs.reshape((int(xyzs.shape[0] / 3), 3))
    energy = np.sum(grid.interpolated(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]))
    return -energy


def _optimize_hydration_site_positions(
    coordinates: Coords, grid: Grid, method: str = "L-BFGS-B"
) -> Coords | None:
    """Refine hydration-site positions to maximize the interpolated density.

    Args:
        coordinates: ``(M, 3)`` initial hydration-site coordinates.
        grid: A ``gridData.Grid`` object to interpolate.
        method: ``scipy.optimize.minimize`` method name.

    Returns:
        The optimized ``(M, 3)`` coordinates, or ``None`` if the optimization
        did not converge.
    """
    results = minimize(
        _objective_function,
        coordinates.flatten(),
        method=method,
        args=(grid),
        tol=1e-6,
        options={"maxfun": 1e12},
    )

    if results["success"]:
        opt_coordinates = results["x"].reshape((int(results["x"].shape[0] / 3), 3))
        return opt_coordinates
    else:
        return None


class HydrationSites:
    """Locate hydration sites in a density map and score them from GIST maps."""

    def __init__(
        self,
        gridsize: float = 0.5,
        water_radius: float = 1.4,
        min_water_distance: float = 2.6,
        min_density: float = 2.0,
    ) -> None:
        """Initialize the hydration-site finder.

        Args:
            gridsize: Grid spacing in Angstrom, used to convert GIST densities
                from kcal/mol/A**3 to kcal/mol.
            water_radius: Radius of a water molecule in Angstrom; also the
                isocontour radius around each site.
            min_water_distance: Minimum allowed distance between two hydration
                sites in Angstrom.
            min_density: Minimum density a voxel must reach to seed a site.
        """
        self._water_radius = water_radius
        self._min_water_distance = min_water_distance
        self._min_density = min_density
        self._gridsize = gridsize

        self._grid_coordinates: Coords | None = None
        self._hydration_sites: Coords | None = None
        self._isocontour: Coords | None = None
        self._cluster_ids: Indices | None = None

    def find(self, grid: str | Grid) -> Coords | None:
        """Find all the hydration sites using the density gO map.

        Args:
            grid: Filename or ``gridData.Grid`` object of the gO (oxygen)
                density map from GIST.

        Returns:
            ``(M, 3)`` coordinates of all the hydration sites found, or ``None``
            if the grid file could not be read.
        """
        Grid = _require_grid()

        if not isinstance(grid, Grid):
            try:
                grid = Grid(grid)
            except FileNotFoundError:
                logger.error("File %s was not found.", grid)
                return None

        xyz = _coordinates_from_grid(grid)
        values = grid.interpolated(xyz[:, 0], xyz[:, 1], xyz[:, 2])

        # Quick positionning of hydration sites
        hydration_sites, self._isocontour, self._cluster_ids = _hydration_sites(
            xyz,
            values,
            self._min_density,
            self._water_radius,
            self._min_water_distance,
        )
        # Optimize positions for better fitting between hydration site positions and
        # water density. We correct the positions due to the roughness of the
        # "clustering" method.
        opt_hydration_sites = None
        # opt_hydration_sites = _optimize_hydration_site_positions(hydration_sites, grid)

        if opt_hydration_sites is not None:
            self._hydration_sites = opt_hydration_sites
        else:
            self._hydration_sites = hydration_sites

        self._grid_coordinates = xyz

        return self._hydration_sites

    def hydration_sites_energy(
        self,
        grid: Grid,
        gridsize: float | None = None,
        water_radius: float | None = None,
        hydration_sites: Coords | None = None,
    ) -> Scalars:
        """Get energy of all the hydration sites.

        Args:
            grid: Multidimensional ``gridData.Grid`` object containing GIST
                energies (kcal/mol/A**3).
            gridsize: Grid spacing in Angstrom (default: the value given at
                construction). If 0, the map is not converted to kcal/mol.
            water_radius: Radius of a water molecule (default: the value given
                at construction). If 0, no Gaussian filtering is applied.
            hydration_sites: ``(M, 3)`` hydration-site coordinates (default: the
                sites found by :meth:`find`).

        Returns:
            ``(M,)`` energy for each hydration site previously found (in
            kcal/mol if ``gridsize`` > 0).
        """
        Grid = _require_grid()
        assert isinstance(grid, Grid), "Argument passed (%s) is not a Grid object." % type(grid)

        if water_radius is None:
            water_radius = float(self._water_radius)
        if gridsize is None:
            gridsize = self._gridsize

        energy = []
        cutoff = water_radius + 0.5
        # Divide the radius by 3 in order to have 3 sigma (99.7 %) at the radius value
        sigma = water_radius / 3.0

        if hydration_sites is None and self._hydration_sites is not None:
            hydration_sites = self._hydration_sites
            grid_coordinates = self._grid_coordinates
        else:
            # If we pass hydration sites coordinates, not sure if it is the same grid
            grid_coordinates = _coordinates_from_grid(grid)

        # If the gridsize is set to 0, we keep the map as is
        # Otherwise we transform kcal/mol/A**3 to kcal/mol
        if gridsize == 0:
            grid_tmp = grid
        else:
            volume = gridsize**3
            grid_tmp = grid * volume

        # Initialize KDTree for faster search
        if water_radius > 0:
            kdtree = cKDTree(grid_coordinates)

        for hydration_site in hydration_sites:
            if water_radius > 0:
                index = kdtree.query_ball_point(hydration_site, cutoff, p=2)
                x = grid_coordinates[index][:, 0]
                y = grid_coordinates[index][:, 1]
                z = grid_coordinates[index][:, 2]

                # Gaussian filtering
                weights = _gaussian_weights(hydration_site, grid_coordinates[index], sigma)
                values = grid_tmp.interpolated(x, y, z) * weights
            else:
                x, y, z = hydration_site
                values = grid_tmp.interpolated(x, y, z)

            energy.append(np.sum(values))

        energy = np.array(energy)

        return energy

    def export_to_pdb(
        self,
        filename: str,
        hydration_sites: Coords,
        values: Scalars | None = None,
    ) -> None:
        """Export hydration sites to a PDB file.

        Args:
            filename: Output PDB filename.
            hydration_sites: ``(M, 3)`` hydration-site coordinates.
            values: ``(M,)`` value associated with each hydration site, written
                to the B-factor column (default: 1.0 for every site).

        Raises:
            AssertionError: If ``values`` is given but its length differs from
                the number of hydration sites.
        """
        if values is not None:
            assert len(hydration_sites) == len(values), (
                "Number of hydration sites and energy values are not equal."
            )
        else:
            values = np.ones(len(hydration_sites))

        i = 1
        output_str = ""
        pdb_str = "ATOM  %5d  %-4s%-3s%2s%4d    %8.3f%8.3f%8.3f  1.00%6.2f          %2s\n"

        for hydration_site, value in zip(hydration_sites, values):
            x, y, z = hydration_site
            output_str += pdb_str % (i, "OW", "WAT", "A", i, x, y, z, value, "O")
            i += 1

        with open(filename, "w") as w:
            w.write(output_str)
