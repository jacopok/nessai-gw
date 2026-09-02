"""Sky reparameterisations for single-site triangular detectors."""

import numpy as np
from nessai.reparameterisations import AnglePair

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
    **kwargs
        Forwarded to :class:`~nessai.reparameterisations.AnglePair`.
    """

    def __init__(self, rotation=None, **kwargs):
        super().__init__(**kwargs)
        if rotation is None:
            raise ValueError("RotatedAnglePair requires a `rotation` matrix.")
        R = np.asarray(rotation, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"`rotation` must be (3, 3), got {R.shape}.")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            raise ValueError("`rotation` must be orthogonal.")
        self._rotation = R

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
