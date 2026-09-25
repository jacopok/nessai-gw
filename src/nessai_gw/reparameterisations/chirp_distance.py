"""Chirp/effective-distance reparameterisation for a single triangular detector.

Roulet et al. 2022 (arXiv:2207.03508, Eqs. 9 and 18) replace ``luminosity_distance``
by ::

    chirp_distance = luminosity_distance / (chirp_mass ** (5/6) * |R_k0(iota, n, psi)|)

with ``R_k0`` the dominant-(2,2)-mode response (Eq. 9) at a fixed reference
("loudest") detector ``k0``.  This strips out the amplitude scaling that
inclination, sky position and polarisation angle impose through the antenna
pattern -- exactly the ``luminosity_distance`` <-> ``theta_jn`` correlation
that dominates a single-site triangular-detector posterior (``corr = 0.998``
on the ET-Delta live points, reduced to ``0.085`` by this coordinate; see
``xg_inference.replay_diagnostics.effective_distance_experiment``).

The geometry (``R_k``, choosing ``k0``) lives in :mod:`nessai_gw._effective_distance`
and is reused here unchanged; this module only wraps it as a
:class:`~nessai.reparameterisations.Reparameterisation`.  See that module's
docstring for why ``k0 = argmax_k |R_k(fiducial)|`` is the right way to pick
the reference detector for the co-located ET-EMR sub-interferometers (they
share one PSD, so the SNR ratio the paper sorts by is exactly the ``|R_k|``
ratio).
"""

from __future__ import annotations

import inspect

import numpy as np
from nessai.reparameterisations import Reparameterisation

from .. import nessai_logger
from .._effective_distance import dominant_detector, response_R
from .._ellipse import ideal_triangle_tensors
from .._geometry import greenwich_mean_sidereal_time

logger = nessai_logger.getChild(__name__)


class ChirpDistanceReparameterisation(Reparameterisation):
    """``luminosity_distance -> chirp_distance`` (Roulet et al. 2022, Eq. 18).

    Parameters
    ----------
    parameters : str or list
        Must be ``luminosity_distance`` (or a single-element list containing
        it).
    prior_bounds : optional
        Accepted for registry compatibility; unused (this is a residual-style
        map like :class:`~nessai_gw.reparameterisations.PolarisationEllipseReparameterisation`,
        not a rescaling to a bounded prime range -- the group-mixture wrapper
        standardises the prime coordinate regardless).
    plane_normal, reference_time, azimuth_offset : optional
        Forwarded to :func:`nessai_gw._ellipse.ideal_triangle_tensors` /
        :func:`nessai_gw._geometry.greenwich_mean_sidereal_time` to build the
        idealised-triangle antenna response, unless ``tensors``/``gmst`` are
        given directly.
    tensors, gmst : optional
        Precomputed detector tensors and Greenwich mean sidereal time, e.g.
        shared with an existing :class:`nessai_gw._ellipse.PolarisationEllipse`
        via its ``.tensors``/``.gmst`` attributes.
    fiducial : mapping, optional
        ``ra``, ``dec``, ``psi``, ``theta_jn`` of a reference signal (typically
        the injection or the maximum-likelihood point), used once at
        construction to pick the reference detector ``k0 = argmax_k |R_k|``.
        Required unless ``k0`` is given directly.
    k0 : int, optional
        The reference detector index directly, bypassing ``fiducial``.
    chirp_mass, theta_jn, ra, dec, psi : str, optional
        Names of the parameters ``R_k0`` depends on (defaults match the
        standard ``xg_inference`` parameter names).
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        plane_normal=None,
        reference_time=None,
        azimuth_offset=0.0,
        tensors=None,
        gmst=None,
        fiducial=None,
        k0=None,
        chirp_mass="chirp_mass",
        theta_jn="theta_jn",
        ra="ra",
        dec="dec",
        psi="psi",
        prior=None,
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

        if self.parameters != ["luminosity_distance"]:
            raise RuntimeError(
                "ChirpDistanceReparameterisation must act on "
                f"'luminosity_distance'; got {self.parameters}"
            )

        if tensors is None or gmst is None:
            if plane_normal is None or reference_time is None:
                raise ValueError(
                    "ChirpDistanceReparameterisation requires either "
                    "`tensors` and `gmst`, or `plane_normal` and "
                    "`reference_time`."
                )
            tensors = ideal_triangle_tensors(plane_normal, azimuth_offset)
            gmst = greenwich_mean_sidereal_time(reference_time)
        self.tensors = np.asarray(tensors, dtype=float)
        self.gmst = float(gmst)

        if k0 is None:
            if fiducial is None:
                raise ValueError(
                    "ChirpDistanceReparameterisation requires either `k0` "
                    "or `fiducial` (ra, dec, psi, theta_jn of a reference "
                    "signal) to pick the reference detector."
                )
            missing = {"ra", "dec", "psi", "theta_jn"} - set(fiducial)
            if missing:
                raise ValueError(f"`fiducial` is missing {sorted(missing)}.")
            k0, _ = dominant_detector(
                self.tensors,
                fiducial["ra"],
                fiducial["dec"],
                fiducial["psi"],
                fiducial["theta_jn"],
                self.gmst,
            )
        self.k0 = int(k0)
        logger.info(
            "ChirpDistanceReparameterisation: reference detector k0=%d",
            self.k0,
        )

        self._chirp_mass = chirp_mass
        self._theta_jn = theta_jn
        self._ra = ra
        self._dec = dec
        self._psi = psi
        self.requires = [chirp_mass, theta_jn, ra, dec, psi]

        self.prime_parameters = ["chirp_distance"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = ["chirp_distance"]
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + self.requires
                )
            )

    # ------------------------------------------------------------------
    def _abs_r(self, x):
        r = response_R(
            self.tensors,
            x[self._ra],
            x[self._dec],
            x[self._psi],
            x[self._theta_jn],
            self.gmst,
        )[..., self.k0]
        return np.abs(r)

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        c = x[self._chirp_mass] ** (5.0 / 6.0) * self._abs_r(x)
        x_prime[self.prime_parameters[0]] = x["luminosity_distance"] / c
        return x, x_prime, log_j - np.log(c)

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        c = x[self._chirp_mass] ** (5.0 / 6.0) * self._abs_r(x)
        x["luminosity_distance"] = x_prime[self.prime_parameters[0]] * c
        return x, x_prime, log_j + np.log(c)
