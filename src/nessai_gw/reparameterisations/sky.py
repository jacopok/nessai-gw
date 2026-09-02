"""Sky reparameterisations for single-site triangular detectors."""

import numpy as np
from nessai.reparameterisations import AnglePair
from scipy import stats

from .. import nessai_logger

logger = nessai_logger.getChild(__name__)


class RotatedAnglePair(AnglePair):
    """:class:`~nessai.reparameterisations.AnglePair` with a fixed frame rotation.

    Identical to ``AnglePair`` except that a constant orthogonal rotation
    ``rotation`` (3x3) is applied to the Cartesian output vector ``(x, y, z)``
    on the forward pass (and its transpose on the inverse).

    The intended use is the sky position of a single-site triangular detector
    (e.g. Einstein Telescope): with ``rotation`` the Earth-fixed-equatorial ->
    detector-frame map (``z`` along the detector-plane normal, GMST folded in),
    the group-mixture fundamental domain -- detector-frame azimuth in
    ``[0, pi/2)`` and the ``beta_f >= 0`` hemisphere -- becomes the
    axis-aligned octant ``x >= 0, y >= 0, z >= 0`` instead of an oblique wedge
    cutting across all three equatorial axes.  The single base flow then only
    has to fit a clean octant of the sphere.

    A rotation has unit Jacobian determinant and leaves the isotropic prime
    prior invariant, so nothing else about ``AnglePair`` changes.

    Parameters
    ----------
    rotation : array_like
        ``(3, 3)`` orthogonal matrix.  ``v_out = rotation @ v_cartesian``.
    radial_sigma : float or None, optional
        Standard deviation of the auxiliary radial coordinate.  ``AnglePair``
        draws it from ``chi(3)`` (mode ~1.4, but ~20% of the mass at ``r < 1``
        and non-negligible density down to ``r ~ 0``), which forces the base
        flow to model a cone tapering into the coordinate singularity at the
        origin.  The sky *direction* is all that carries information, so by
        default this replaces ``chi(3)`` with a ``Gamma`` concentrated at
        ``r = 1`` with this width (default ``0.15``): the prime points then lie
        in a thin unit shell and the base flow fits a 2-D patch of the sphere
        with a near-trivial radial dof -- nothing has to stretch across the
        origin.  Pass ``None`` to keep the ``chi(3)`` behaviour.
    **kwargs
        Forwarded to :class:`~nessai.reparameterisations.AnglePair`.
    """

    def __init__(self, rotation=None, radial_sigma=0.15, **kwargs):
        super().__init__(**kwargs)
        if rotation is None:
            raise ValueError("RotatedAnglePair requires a `rotation` matrix.")
        R = np.asarray(rotation, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"`rotation` must be (3, 3), got {R.shape}.")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            raise ValueError("`rotation` must be orthogonal.")
        self._rotation = R

        self.radial_sigma = radial_sigma
        if radial_sigma is not None and getattr(self, "chi", False):
            # Gamma(a, scale=1/a): mean 1, variance 1/a = radial_sigma**2,
            # strictly positive so the `r < 0` guard never trips.
            a = 1.0 / float(radial_sigma) ** 2
            self.chi = stats.gamma(a=a, scale=1.0 / a)

    @property
    def _cartesian_names(self):
        """The three Cartesian coordinate names (nessai-version agnostic)."""
        for attr in ("prime_parameters", "output_parameters"):
            names = getattr(self, attr, None)
            if names is not None and len(names) == 3:
                return list(names)
        return [self.x, self.y, self.z]

    def _rotate(self, x_prime, R):
        px, py, pz = self._cartesian_names
        vx = np.array(x_prime[px], copy=True)
        vy = np.array(x_prime[py], copy=True)
        vz = np.array(x_prime[pz], copy=True)
        x_prime[px] = R[0, 0] * vx + R[0, 1] * vy + R[0, 2] * vz
        x_prime[py] = R[1, 0] * vx + R[1, 1] * vy + R[1, 2] * vz
        x_prime[pz] = R[2, 0] * vx + R[2, 1] * vy + R[2, 2] * vz

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        x, x_prime, log_j = super().reparameterise(x, x_prime, log_j, **kwargs)
        self._rotate(x_prime, self._rotation)
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        unrotated = x_prime.copy()
        self._rotate(unrotated, self._rotation.T)
        x, _, log_j = super().inverse_reparameterise(
            x, unrotated, log_j, **kwargs
        )
        return x, x_prime, log_j
