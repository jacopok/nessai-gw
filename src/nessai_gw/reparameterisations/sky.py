"""Sky reparameterisations for single-site triangular detectors."""

import inspect

import numpy as np
from nessai.reparameterisations import Reparameterisation

from .. import nessai_logger

logger = nessai_logger.getChild(__name__)

_TWO_PI = 2.0 * np.pi


class EqualAreaSky(Reparameterisation):
    r"""Two-coordinate equal-area sky map for a single-site triangular detector.

    This maps ``(ra, dec)`` **directly to two coordinates**, with no auxiliary
    radius:

    * rotate the equatorial sky unit vector into the detector frame
      (``z`` along the plane normal, GMST folded in) with the fixed ``rotation``;
    * ``u = lambda_f / (2*pi)``           (detector-frame azimuth -> ``[0, 1)``)
    * ``v = (1 - z_f) / 2``               (Lambert cylindrical equal-area of the
      colatitude from the plane normal -> ``[0, 1)``)

    Under the uniform-on-sphere prior ``(u, v)`` are **exactly uniform on the
    unit square** and independent (``d(solid angle) = d lambda_f d(cos theta_f)
    = 2*pi * 2 du dv``). The ET-triangle group acts on ``(u, v)`` by
    ``u -> u + k/4 mod 1`` (the ``Z4`` in-plane rotation) and ``v -> 1 - v``
    (the ``Z2`` plane reflection), so the fundamental domain is the sub-square
    ``u in [0, 1/4), v in [0, 1/2)``; the probit that turns that sub-square back
    into two standard normals is applied *after* the group fold, by
    :class:`nessai_gw.group_mixture.SkyOctantProbit` (a ``canonical_transform``).

    The sky direction carries exactly its 2 physical degrees of freedom.

    Jacobian: ``|d(u, v) / d(ra, dec)| = cos(dec) / (4*pi)`` (a rotation
    preserves the solid-angle element).

    Parameters
    ----------
    parameters : list of str
        ``["ra", "dec"]`` (in either order).
    prior_bounds : dict
        Prior bounds; ``ra`` must span ``[0, 2*pi]`` and ``dec``
        ``[-pi/2, pi/2]``.
    rotation : array_like
        ``(3, 3)`` orthogonal matrix, the Earth-fixed-equatorial -> detector
        frame map (``TriangularDetectorGroupAction.sky_frame_rotation``).
    """

    one_to_one = False
    requires_bounded_prior = True

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        rotation=None,
        rng=None,
        **kwargs,
    ):
        parent_params = inspect.signature(
            Reparameterisation.__init__
        ).parameters
        call = dict(parameters=parameters, prior_bounds=prior_bounds)
        if "rng" in parent_params:
            call["rng"] = rng
        for key, value in kwargs.items():
            if key in parent_params:
                call[key] = value
        super().__init__(**call)

        if set(self.parameters) != {"ra", "dec"}:
            raise RuntimeError(
                f"EqualAreaSky must act on ['ra', 'dec']; got {self.parameters}"
            )
        # Fix a canonical (ra, dec) order regardless of dict iteration order.
        self.input_parameters = ["ra", "dec"]
        self.prime_parameters = ["sky_u", "sky_v"]
        self.output_parameters = ["sky_u", "sky_v"]

        if rotation is None:
            raise ValueError("EqualAreaSky requires a `rotation` matrix.")
        R = np.asarray(rotation, dtype=float)
        if R.shape != (3, 3) or not np.allclose(
            R @ R.T, np.eye(3), atol=1e-6
        ):
            raise ValueError("`rotation` must be a (3, 3) orthogonal matrix.")
        self._R = R

        ra_lo, ra_hi = self.prior_bounds["ra"]
        if not np.allclose([ra_lo, ra_hi], [0.0, _TWO_PI]):
            raise ValueError(
                f"EqualAreaSky needs ra in [0, 2pi]; got {(ra_lo, ra_hi)}"
            )
        self.prime_prior_bounds = {"sky_u": [0.0, 1.0], "sky_v": [0.0, 1.0]}

    def set_rotation(self, rotation):
        """Replace the frame rotation (a moved reference epoch)."""
        self._R = np.asarray(rotation, dtype=float)

    # -- transform -----------------------------------------------------------

    def _to_detector(self, ra, dec):
        cd = np.cos(dec)
        w = np.stack(
            [cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=0
        )
        return self._R @ w  # (3, N)

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        ra = np.asarray(x["ra"], dtype=float)
        dec = np.asarray(x["dec"], dtype=float)
        wd = self._to_detector(ra, dec)
        lam = np.mod(np.arctan2(wd[1], wd[0]), _TWO_PI)
        z = np.clip(wd[2], -1.0, 1.0)
        x_prime["sky_u"] = lam / _TWO_PI
        x_prime["sky_v"] = 0.5 * (1.0 - z)
        log_j = log_j + np.log(np.clip(np.cos(dec), 1e-12, None)) - np.log(
            4.0 * np.pi
        )
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        u = np.asarray(x_prime["sky_u"], dtype=float)
        v = np.asarray(x_prime["sky_v"], dtype=float)
        lam = _TWO_PI * u
        z = np.clip(1.0 - 2.0 * v, -1.0, 1.0)
        rho = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
        wd = np.stack([rho * np.cos(lam), rho * np.sin(lam), z], axis=0)
        we = self._R.T @ wd
        ra = np.mod(np.arctan2(we[1], we[0]), _TWO_PI)
        dec = np.arcsin(np.clip(we[2], -1.0, 1.0))
        x["ra"] = ra
        x["dec"] = dec
        log_j = log_j + np.log(4.0 * np.pi) - np.log(
            np.clip(np.cos(dec), 1e-12, None)
        )
        return x, x_prime, log_j
