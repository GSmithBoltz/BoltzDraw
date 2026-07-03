"""Pure geometric utilities for 3D computations.

Ported from ``waterkit.utils`` (pure numpy/scipy, no chemistry). These are used
throughout water construction, hydrogen-bond vector placement, and the sampler.

The implementations are faithful, numerically identical ports of the original
WaterKit helpers; only type hints and docstrings have been added. Numpy arrays
are annotated with jaxtyping (see :mod:`waterkit.utils.typing`) so each tensor's
shape is visible in the signature. The single dependency swap is
:func:`sphere_grid_points`, which now imports ``scipy.spatial.distance``
explicitly instead of relying on a module-level ``spatial`` global.
"""

import numpy as np
from scipy.spatial import distance

from .typing import (
    Coords,
    Float,
    Point3,
    Points,
    Quaternion,
    QuaternionMatrix,
    Quaternions,
    RotationMatrix,
    Scalars,
    Vec3,
)


def vector(
    a: Float[np.ndarray, "*batch 3"], b: Float[np.ndarray, "*batch 3"]
) -> Float[np.ndarray, "*batch 3"]:
    """Return the vector from point ``a`` to point ``b``.

    Shape-polymorphic: works on a single 3-vector or a batch of them.

    Args:
        a: Tail point coordinate(s).
        b: Head point coordinate(s).

    Returns:
        The displacement vector ``b - a``.
    """
    return b - a


def normalize(a: Vec3) -> Vec3:
    """Return the unit vector in the direction of ``a``.

    Args:
        a: Input vector.

    Returns:
        The normalized vector ``a / |a|``.
    """
    return a / np.sqrt(np.sum(np.power(a, 2)))


def get_perpendicular_vector(u: Vec3) -> Vec3:
    """Return an arbitrary vector perpendicular to ``u``.

    A random direction is generated and crossed with ``u`` so the result is
    perpendicular to ``u`` but otherwise unconstrained.

    Args:
        u: Input vector.

    Returns:
        A vector orthogonal to ``u``.
    """
    v = normalize(np.random.rand(3))
    v = np.cross(u, v)

    return v


def get_euclidean_distance(a: Coords, b: Coords) -> Scalars:
    """Return the Euclidean distance(s) between ``a`` and ``b``.

    Args:
        a: Coordinates of shape ``(N, 3)``.
        b: Reference coordinates of shape ``(N, 3)`` (or broadcastable to ``a``).

    Returns:
        The per-row Euclidean distance(s).
    """
    return np.sqrt(np.sum(np.power(a - b, 2), axis=1))


def get_angle(
    a: Float[np.ndarray, "*batch 3"], b: Vec3, c: Vec3, degree: bool = True
) -> Float[np.ndarray, "*batch"]:
    """Return the angle at vertex ``b`` formed by points ``a``-``b``-``c``.

    Args:
        a: First point, or a batch of points of shape ``(N, 3)``.
        b: Vertex point.
        c: Third point.
        degree: If ``True`` return degrees, otherwise radians.

    Returns:
        The angle(s) between vectors ``b->a`` and ``b->c``.
    """
    ba = np.atleast_2d(vector(b, a))
    bc = vector(b, c)

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc))
    # Make sure values fit between -1 and 1 for arccos
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)

    if degree:
        return np.degrees(angle)

    return angle


def dihedral(p: Float[np.ndarray, "4 3"], degree: bool = False) -> float:
    """Return the dihedral (torsion) angle defined by 4 points.

    Source: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python

    Args:
        p: Sequence of 4 points ``(p0, p1, p2, p3)`` of shape ``(4, 3)``.
        degree: If ``True`` return degrees, otherwise radians.

    Returns:
        The dihedral angle.
    """
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that is fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    angle = np.arctan2(y, x)

    if degree:
        return np.degrees(angle)
    else:
        return angle


def get_rotation_matrix(a: Vec3, b: Vec3) -> RotationMatrix:
    """Return the 3D rotation matrix that aligns vector ``b`` onto ``a``.

    Sources:
    https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/897677

    Args:
        a: Target vector.
        b: Source vector.

    Returns:
        The ``(3, 3)`` rotation matrix.
    """
    v = np.cross(b, a)
    c = np.dot(b, a)
    s = np.linalg.norm(v)
    identity = np.identity(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = identity + k + np.matmul(k, k) * ((1 - c) / (s**2))

    return r


def rotation_axis(p0: Point3, p1: Point3, p2: Point3, origin: Point3 | None = None) -> Point3:
    """Return the rotation axis perpendicular to the plane of 3 points.

    Args:
        p0: First point (anchor for the axis direction).
        p1: Second point.
        p2: Third point.
        origin: If provided, the axis is centered at this origin; otherwise it
            is centered at ``p0``.

    Returns:
        A point such that ``axis = returned_point - center`` is a unit vector
        normal to the plane defined by the three points.
    """
    r = normalize(np.cross(vector(p1, p0), vector(p2, p0)))

    if origin is not None:
        return origin + r

    return p0 + r


def atom_to_move(o: Point3, p: Coords) -> Point3:
    """Return the coordinates of an atom placed just above acceptor/donor ``o``.

    The returned point lies one unit away from ``o`` in the direction opposite
    to the mean position of the neighbor coordinates ``p``.

    Args:
        o: Origin (acceptor/donor) coordinates.
        p: Neighbor coordinates of shape ``(N, 3)`` (single point allowed).

    Returns:
        The coordinates of the atom to move.
    """
    # It will not work if there is just one dimension
    p = np.atleast_2d(p)
    return o + normalize(-1.0 * vector(o, np.mean(p, axis=0)))


def rotate_point(p: Point3, p1: Point3, p2: Point3, angle: float) -> Point3:
    """Rotate the point ``p`` around the axis ``p1``-``p2`` by ``angle`` radians.

    Source: http://paulbourke.net/geometry/rotate/PointRotate.py

    Args:
        p: Point to rotate.
        p1: First point defining the rotation axis.
        p2: Second point defining the rotation axis.
        angle: Rotation angle in radians.

    Returns:
        The rotated point coordinates.
    """
    # Translate the point we want to rotate to the origin
    pn = p - p1

    # Get the unit vector from the axis p1-p2
    n = p2 - p1
    n = normalize(n)

    # Setup the rotation matrix
    c = np.cos(angle)
    t = 1.0 - np.cos(angle)
    s = np.sin(angle)
    x, y, z = n[0], n[1], n[2]

    rot = np.array(
        [
            [t * x**2 + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y**2 + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z**2 + c],
        ]
    )

    # ... and apply it
    ptr = np.dot(pn, rot)

    # And to finish, we put it back
    p = ptr + p1

    return p


def resize_vector(v: Vec3, length: float, origin: Point3 | None = None) -> Vec3:
    """Resize a vector ``v`` to a new length with respect to an ``origin``.

    Args:
        v: Vector (or point, when ``origin`` is given) to resize.
        length: Target length.
        origin: If provided, ``v`` is treated as a point and resized relative to
            this origin; otherwise it is treated as a free vector.

    Returns:
        The resized vector (or point).
    """
    if origin is not None:
        return (normalize(v - origin) * length) + origin
    else:
        return normalize(v) * length


def generate_random_sphere(center: Point3, radius: float = 1, size: int = 100) -> Points:
    """Generate uniformly random points on a sphere.

    Source: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf

    Args:
        center: Sphere center coordinates.
        radius: Sphere radius.
        size: Number of points to generate.

    Returns:
        Array of shape ``(size, 3)`` of points on the sphere.
    """
    z = np.random.uniform(-radius, radius, size)
    p = np.random.uniform(0, np.pi * 2, size)

    x = np.sqrt(radius**2 - z**2) * np.cos(p)
    y = np.sqrt(radius**2 - z**2) * np.sin(p)

    coordinates = np.stack((x, y, z), axis=-1)
    coordinates += center

    return coordinates


def generate_sphere(center: Point3, radius: float = 1, size: int = 100) -> Points:
    """Generate approximately equidistant points on a sphere.

    Source: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf

    Args:
        center: Sphere center coordinates.
        radius: Sphere radius.
        size: Target number of points (the actual count is approximate).

    Returns:
        Array of shape ``(N, 3)`` of points distributed over the sphere.
    """
    a = 4 * np.pi * radius**2 / size
    d = np.sqrt(a)

    m_v = int(np.round(np.pi / d))
    d_v = np.pi / m_v
    d_p = a / d_v

    coordinates = []

    for m in range(m_v):
        v = np.pi * (m + 0.5) / m_v
        m_p = int(np.round(2 * np.pi * np.sin(v) / d_p))

        for n in range(m_p):
            p = 2 * np.pi * n / m_p

            x = radius * np.sin(v) * np.cos(p)
            y = radius * np.sin(v) * np.sin(p)
            z = radius * np.cos(v)

            coordinates.append([x, y, z])

    coordinates = np.array(coordinates)
    coordinates += center

    return coordinates


def sphere_grid_points(
    center: Point3, spacing: float, radius: float, min_radius: float = 0
) -> Points:
    """Generate grid points contained within a spherical shell.

    Args:
        center: Sphere center coordinates.
        spacing: Grid spacing.
        radius: Outer radius of the shell.
        min_radius: Inner radius of the shell (points closer than this are
            discarded).

    Returns:
        Array of shape ``(N, 3)`` of grid points whose distance to ``center``
        lies in ``[min_radius, radius]``.
    """
    # Number of grid points based on the grid spacing
    n = int(np.rint(radius / spacing)) * 2
    # Transform even numbers to the nearest odd integer
    n = n // 2 * 2 + 1

    x = np.linspace(center[0] - radius, center[0] + radius, n)
    y = np.linspace(center[1] - radius, center[1] + radius, n)
    z = np.linspace(center[2] - radius, center[2] + radius, n)
    # Generate grid
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
    data = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T
    # Compute distance and keep only the ones in the sphere
    distances = distance.cdist(data, center.reshape(1, -1)).ravel()
    points_in_sphere = data[(distances >= min_radius) & (distances <= radius)]
    return points_in_sphere


def makeW(r1: float, r2: float, r3: float, r4: float = 0) -> QuaternionMatrix:
    """Build the W matrix involved in quaternion rotation.

    Source: https://github.com/charnley/rmsd

    Args:
        r1: First quaternion component.
        r2: Second quaternion component.
        r3: Third quaternion component.
        r4: Fourth (scalar) quaternion component.

    Returns:
        The ``(4, 4)`` W matrix.
    """
    w = np.asarray(
        [
            [r4, r3, -r2, r1],
            [-r3, r4, r1, r2],
            [r2, -r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return w


def makeQ(r1: float, r2: float, r3: float, r4: float = 0) -> QuaternionMatrix:
    """Build the Q matrix involved in quaternion rotation.

    Source: https://github.com/charnley/rmsd

    Args:
        r1: First quaternion component.
        r2: Second quaternion component.
        r3: Third quaternion component.
        r4: Fourth (scalar) quaternion component.

    Returns:
        The ``(4, 4)`` Q matrix.
    """
    q = np.asarray(
        [
            [r4, -r3, r2, r1],
            [r3, r4, -r1, r2],
            [-r2, r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return q


def quaternion_rotate(Y: Coords, X: Coords) -> Quaternion:
    """Calculate the rotation quaternion between two sets of coordinates.

    Source: https://github.com/charnley/rmsd

    Args:
        Y: ``(N, 3)`` matrix, where N is points.
        X: ``(N, 3)`` matrix, where N is points.

    Returns:
        The rotation quaternion aligning ``X`` onto ``Y``.
    """
    n = X.shape[0]
    w = np.asarray([makeW(*Y[k]) for k in range(n)])
    q = np.asarray([makeQ(*X[k]) for k in range(n)])
    qt_dot_w = np.asarray([np.dot(q[k].T, w[k]) for k in range(n)])
    a = np.sum(qt_dot_w, axis=0)
    eigen = np.linalg.eigh(a)
    r = eigen[1][:, eigen[0].argmax()]
    return r


def rotate_vector_by_quaternion(v: Vec3, q: Quaternion) -> Vec3:
    """Rotate a vector using a quaternion.

    Source: https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion

    Args:
        v: Vector to rotate.
        q: Quaternion ``(w, x, y, z)`` with the scalar part first.

    Returns:
        The rotated vector.
    """
    u = q[1:]
    s = q[0]
    v_prime = 2.0 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2.0 * s * np.cross(u, v)
    return v_prime


def shoemake(coordinates: Coords) -> Quaternions:
    """Apply the Shoemake transformation to map ``[0, 1]^3`` to unit quaternions.

    Source: http://planning.cs.uiuc.edu/node198.html

    Args:
        coordinates: Array of shape ``(N, 3)`` (single triplet allowed) of
            uniform random values in ``[0, 1]``.

    Returns:
        Array of shape ``(N, 4)`` of unit quaternions.
    """
    coordinates = np.atleast_2d(coordinates)
    t1 = np.sqrt(1.0 - coordinates[:, 0])
    t2 = np.sqrt(coordinates[:, 0])
    s1 = 2.0 * np.pi * coordinates[:, 1]
    s2 = 2.0 * np.pi * coordinates[:, 2]
    return np.dstack((t1 * np.sin(s1), t1 * np.cos(s1), t2 * np.sin(s2), t2 * np.cos(s2)))[0]


def random_quaternion(n: int = 1) -> Quaternions:
    """Create ``n`` random unit quaternions.

    Args:
        n: Number of quaternions to generate.

    Returns:
        Array of shape ``(n, 4)`` of random unit quaternions.
    """
    u = np.random.random(size=(n, 3))
    return shoemake(u)
