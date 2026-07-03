"""GIST map post-processing utilities.

Port of ``waterkit.analysis.gist``. Provides Gaussian smoothing ("blurring") of
GIST thermodynamic grids, weighting neighbouring voxels by distance.

``griddataformats`` is an optional dependency: it is imported lazily inside the
functions that need it, so this module imports cleanly without it installed and
only fails (with :class:`~waterkit.utils.exceptions.WaterKitError`) when such a
function is actually called.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree, distance

from ..utils.exceptions import MapError, WaterKitError
from ..utils.typing import Coords, Float, Scalars, Vec3

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gridData import Grid

logger = logging.getLogger("waterkit.analysis.gist")


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


def blur_map(
    grid: Grid,
    radius: float = 1.4,
    gridsize: float = 0.5,
    center: Vec3 | None = None,
    box_size: Float[np.ndarray, "3"] | None = None,
    cutoff: float | None = None,
    autodock_format: bool = False,
) -> Grid:
    """Smooth a map by summing grid points within ``radius``, weighted by a Gaussian.

    Args:
        grid: Multidimensional ``gridData.Grid`` object containing GIST energies
            (kcal/mol/A**3).
        radius: Gaussian blurring radius (default: 1.4, water-molecule radius).
        gridsize: Grid spacing in Angstrom (default: 0.5). If 0, densities are
            not converted from kcal/mol/A**3 to kcal/mol.
        center: ``(3,)`` center ``(x, y, z)`` of the output box (default: None,
            use the input grid).
        box_size: ``(3,)`` dimensions ``(a, b, c)`` of the output box (default:
            None, use the input grid).
        cutoff: Filtering cutoff distance (default: None, ``radius + 0.5``).
        autodock_format: If True, force an odd number of grid points per axis
            (as AutoDock maps require) when building a custom box.

    Returns:
        The Gaussian-blurred ``gridData.Grid`` map (kcal/mol, or kcal/mol/A**3
        if ``gridsize`` equals 0).

    Raises:
        WaterKitError: If ``griddataformats`` is not installed.
        MapError: If a requested grid point falls outside the original box.
    """
    Grid = _require_grid()
    assert isinstance(grid, Grid), "Argument passed (%s) is not a Grid object." % type(grid)

    if cutoff is None:
        cutoff = radius + 0.5
    else:
        assert radius < cutoff, "Radius (%f) must be smaller than distance cutoff (%f)" % (
            radius,
            cutoff,
        )

    energy = []
    # Divide the radius by 3 in order to have 3 sigma (99.7 %) at the radius value
    sigma = float(radius / 3.0)

    # Get the grid coordinates
    if center is not None and box_size is not None:
        center = np.array(center)
        box_size = np.array(box_size)

        assert np.ravel(center).size == 3, "Error: center should contain only (x, y, z)."
        assert np.ravel(box_size).size == 3, "Error: grid size should contain only (a, b, c)."
        assert (box_size > 0).all(), "Error: grid size cannot contain negative numbers."

        # Number of grid points is odd for AutoDock maps
        if autodock_format:
            nelements = np.round(box_size / gridsize).astype(int)
            nelements = nelements // 2 * 2 + 1
            sd = (nelements - 1) * gridsize / 2
        else:
            sd = box_size / 2.0

        x, y, z = center
        x = np.arange(x - sd[0], x + sd[0] + gridsize, gridsize)
        y = np.arange(y - sd[1], y + sd[1] + gridsize, gridsize)
        z = np.arange(z - sd[2], z + sd[2] + gridsize, gridsize)
        X, Y, Z = np.meshgrid(x, y, z)
        grid_coordinates = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        new_shape = [x.shape[0], y.shape[0], z.shape[0]]
    else:
        grid_coordinates = _coordinates_from_grid(grid)
        new_shape = [grid.grid.shape[1], grid.grid.shape[0], grid.grid.shape[2]]

    # If the gridsize is set to 0, we keep the map as is
    # Otherwise we transform kcal/mol/A**3 to kcal/mol
    if gridsize == 0:
        grid_tmp = grid
        gridsize = grid.delta
    else:
        volume = gridsize**3
        grid_tmp = grid * volume

    # Create grid interpolator
    # The one from Scipy is better than the one in gridData
    grid_interpn = RegularGridInterpolator(grid_tmp.midpoints, grid_tmp.grid)
    # Initialize KDTree for fast search
    kdtree = cKDTree(grid_coordinates)

    for xyz in grid_coordinates:
        index = kdtree.query_ball_point(xyz, cutoff, p=2)

        weights = _gaussian_weights(xyz, grid_coordinates[index], sigma)
        try:
            weighted_values = grid_interpn(grid_coordinates[index]) * weights
        except ValueError as exc:
            raise MapError("Grid point outside the original box.") from exc

        energy.append(np.sum(weighted_values))

    energy = np.array(energy)

    # Swap axis 0 and 1
    new_grid = Grid(
        np.swapaxes(energy.reshape(new_shape), 0, 1),
        origin=grid_coordinates[0],
        delta=gridsize,
    )

    return new_grid
