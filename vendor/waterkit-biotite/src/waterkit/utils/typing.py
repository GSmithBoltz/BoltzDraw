"""Array shape/dtype annotations for waterkit (jaxtyping over numpy).

Every numpy array in waterkit is annotated with `jaxtyping
<https://docs.kidger.site/jaxtyping/>`_ so the **shape** and **dtype** of a
tensor are visible directly in the signature -- ``Float[np.ndarray, "n 3"]``
reads as "``n`` rows of 3 floats", which is far more informative than a bare
``np.ndarray`` or ``npt.NDArray[np.float64]``.

These annotations are *documentary only*. waterkit does not install a runtime
type checker (e.g. ``beartype``/``typeguard``), so jaxtyping never enforces the
shapes at run time -- passing a mismatched array will not raise. The goal is
readable, self-describing signatures and editor/tooling hints, not validation.

Axis-name convention
---------------------
The single-letter / short axis names below are used consistently across the
package so the same symbol means the same thing everywhere:

==============  =================================================================
``3``           a single 3D point or vector ``(x, y, z)``
``4``           a quaternion ``(w, x, y, z)`` (or four quaternion components)
``n``           a number of atoms / points / candidate states (context-dependent)
``n 3``         ``n`` cartesian coordinates
``n 4``         ``n`` quaternions
``3 3``         a rotation matrix
``4 4``         a quaternion ``W``/``Q`` matrix
``nx ny nz``    a scalar field sampled on the AutoDock grid
``...``         arbitrary shape (used where a helper is fully shape-polymorphic)
==============  =================================================================

Usage
-----
Import the array *kinds* (:data:`Float`, :data:`Int`, :data:`Bool`, ...) for
inline annotations, and/or the named aliases below for the common shapes::

    from waterkit.utils.typing import Float, Vec3, Points

    def recenter(coords: Points, center: Vec3) -> Points:
        return coords - center

    def pairwise_norm(a: Float[np.ndarray, "n 3"]) -> Float[np.ndarray, "n n"]:
        ...
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Float, Int, Num, Shaped

# jaxtyping array *kinds* (Bool/Float/Int/Num/Shaped) for inline annotations,
# plus the named shape aliases defined below.
__all__ = [
    "Bool",
    "BoolArray",
    "BoolMask",
    "Coords",
    "Float",
    "FloatArray",
    "Grid3D",
    "Indices",
    "Int",
    "IntArray",
    "Num",
    "Point3",
    "Points",
    "Quaternion",
    "QuaternionMatrix",
    "Quaternions",
    "RotationMatrix",
    "Scalars",
    "Shaped",
    "Vec3",
]

# --- floating-point arrays -------------------------------------------------
FloatArray = Float[np.ndarray, "..."]
"""A floating-point array of arbitrary shape."""

Vec3 = Float[np.ndarray, "3"]
"""A single 3D vector ``(x, y, z)``."""

Point3 = Float[np.ndarray, "3"]
"""A single 3D point ``(x, y, z)`` (alias of :data:`Vec3`)."""

Points = Float[np.ndarray, "n 3"]
"""``n`` cartesian coordinates, shape ``(n, 3)``."""

Coords = Float[np.ndarray, "n 3"]
"""``n`` cartesian coordinates, shape ``(n, 3)`` (alias of :data:`Points`)."""

Quaternion = Float[np.ndarray, "4"]
"""A single quaternion ``(w, x, y, z)``."""

Quaternions = Float[np.ndarray, "n 4"]
"""``n`` quaternions, shape ``(n, 4)``."""

RotationMatrix = Float[np.ndarray, "3 3"]
"""A 3x3 rotation matrix."""

QuaternionMatrix = Float[np.ndarray, "4 4"]
"""A 4x4 quaternion ``W``/``Q`` matrix."""

Scalars = Float[np.ndarray, "n"]
"""``n`` scalar values (e.g. energies, distances, probabilities)."""

Grid3D = Float[np.ndarray, "nx ny nz"]
"""A scalar field sampled on the AutoDock grid, shape ``(nx, ny, nz)``."""

# --- integer / boolean arrays ----------------------------------------------
IntArray = Int[np.ndarray, "..."]
"""An integer array of arbitrary shape."""

Indices = Int[np.ndarray, "n"]
"""``n`` integer indices."""

BoolArray = Bool[np.ndarray, "..."]
"""A boolean array of arbitrary shape."""

BoolMask = Bool[np.ndarray, "n"]
"""A boolean mask over ``n`` elements."""
