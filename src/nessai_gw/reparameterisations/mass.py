"""Reparameterisation for the chirp mass of a long signal."""

import inspect

import numpy as np
from nessai.reparameterisations import Reparameterisation

from .. import nessai_logger

logger = nessai_logger.getChild(__name__)


def doppler_factor(ra, dec, doppler_vector):
    r""":math:`D = 1 - \hat{n}(\alpha, \delta)\cdot\mathbf{v}/c`."""
    k = np.asarray(doppler_vector, dtype=float)
    cos_dec = np.cos(dec)
    return 1.0 - (
        cos_dec * np.cos(ra) * k[0] + cos_dec * np.sin(ra) * k[1]
        + np.sin(dec) * k[2]
    )


class DopplerChirpMassReparameterisation(Reparameterisation):
    r"""``chirp_mass -> chirp_mass_prime``, the Doppler-corrected chirp mass
    rescaled to its bounds.

    With the Earth's orbital motion in the detector response (bilby_xG's
    ``orbital_motion``), the sampled detector-frame masses are those of the
    Solar-System barycentre.  To first order in :math:`v/c` the signal only
    sees the masses in a frame comoving with the Earth,

    .. math::
        \mathcal{M}_\oplus = \mathcal{M}\,D(\alpha, \delta),
        \qquad D = 1 - \hat{n}\cdot\mathbf{v}_\oplus / c,

    and the Doppler factor varies by ~``2e-4`` over the sky: far more than the
    chirp-mass uncertainty of a long, loud signal.  The posterior on
    ``chirp_mass`` is then a thin ridge curved along the sky, and the flow's
    acceptance decays as it thins.  This samples ``M_earth`` instead
    (rescaled to ``[-1, 1]`` within its bounds, which follow the live points
    as the default ``mass`` rescaling does), which the data measure
    independently of the sky.

    ``doppler_vector`` is :math:`\mathbf{v}_\oplus / c` in the axes of
    ``(ra, dec)`` at the coalescence time (with bilby_xG,
    ``precession_matrix(t) @ earth_barycentric_position_velocity(t)[1] / c``);
    its variation over the ``geocent_time`` prior is negligible.  ``ra`` and
    ``dec`` are read from x-space on the inverse, so the sky must be inverted
    first.  The map is exactly invertible, with
    :math:`\log|\partial\mathcal{M}'/\partial\mathcal{M}| = \log D + \log(2 / (b - a))`.

    Parameters
    ----------
    parameters : str or list of str
        The chirp mass (default ``"chirp_mass"``).
    prior_bounds : dict
        Its prior bounds; the initial bounds of ``M_earth`` are these widened
        by the largest Doppler factor.
    doppler_vector : array_like
        :math:`\mathbf{v}_\oplus / c` (3 components).
    update_bounds : bool, optional
        Move the bounds to the range of the training points at each update
        (default ``True``, as the ``mass`` rescaling).
    ra, dec : str, optional
        Names of the sky coordinates.
    """

    requires_bounded_prior = True

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        doppler_vector=None,
        update_bounds=True,
        ra="ra",
        dec="dec",
        rng=None,
        **kwargs,
    ):
        parent_params = inspect.signature(
            Reparameterisation.__init__
        ).parameters
        if parameters is None and "input_parameters" in kwargs:
            parameters = kwargs.pop("input_parameters")
        if parameters is None:
            parameters = ["chirp_mass"]
        call = {"parameters": parameters, "prior_bounds": prior_bounds}
        if "rng" in parent_params:
            call["rng"] = rng
        for key, value in kwargs.items():
            if key in parent_params:
                call[key] = value
        super().__init__(**call)

        if len(self.parameters) != 1:
            raise RuntimeError(
                "DopplerChirpMassReparameterisation only supports one "
                f"parameter; got {self.parameters}"
            )
        if doppler_vector is None:
            raise RuntimeError("DopplerChirpMassReparameterisation needs "
                               "doppler_vector")
        self.doppler_vector = np.asarray(doppler_vector, dtype=float)
        if self.doppler_vector.shape != (3,):
            raise RuntimeError(
                f"doppler_vector must have 3 components; got "
                f"{self.doppler_vector}"
            )
        speed = float(np.linalg.norm(self.doppler_vector))
        if speed >= 1e-2:
            raise RuntimeError(f"|doppler_vector| = {speed} is not v/c")
        self._update = bool(update_bounds)
        self._ra, self._dec = ra, dec
        self.requires = [ra, dec]
        self.prime_parameters = [f"{self.parameters[0]}_prime"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = list(self.prime_parameters)
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + [ra, dec]
                )
            )
        lo, hi = self.prior_bounds[self.parameters[0]]
        self._initial_bounds = (
            float(lo) * (1.0 - speed), float(hi) * (1.0 + speed)
        )
        self.reset()

    def _doppler(self, x):
        return doppler_factor(x[self._ra], x[self._dec], self.doppler_vector)

    def _log_scale(self):
        lo, hi = self.bounds
        return np.log(2.0) - np.log(hi - lo)

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        d = self._doppler(x)
        y = x[self.parameters[0]] * d
        lo, hi = self.bounds
        x_prime[self.prime_parameters[0]] = 2.0 * (y - lo) / (hi - lo) - 1.0
        return x, x_prime, log_j + np.log(d) + self._log_scale()

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        d = self._doppler(x)
        lo, hi = self.bounds
        y = lo + 0.5 * (x_prime[self.prime_parameters[0]] + 1.0) * (hi - lo)
        x[self.parameters[0]] = y / d
        return x, x_prime, log_j - np.log(d) - self._log_scale()

    def update_bounds(self, x, x_prime=None):
        if not self._update or len(x) == 0:
            return
        y = x[self.parameters[0]] * self._doppler(x)
        lo, hi = float(np.min(y)), float(np.max(y))
        if hi > lo:
            self.bounds = (lo, hi)
            logger.debug("Doppler chirp-mass bounds: %s", self.bounds)

    def update(self, x, x_prime=None):
        self.update_bounds(x, x_prime=x_prime)

    def reset(self):
        self.bounds = self._initial_bounds


def check_spin_names(spins):
    """``spins`` as a tuple: two aligned-spin names (``chi_1``, ``chi_2``) or
    four precessing ones (``a_1``, ``tilt_1``, ``a_2``, ``tilt_2``); None
    stays None."""
    if spins is None:
        return None
    spins = tuple(spins)
    if len(spins) not in (2, 4):
        raise ValueError("`spins` must name two aligned spins or (a_1, tilt_1, "
                         f"a_2, tilt_2); got {spins}")
    return spins


def aligned_spins(x, spins):
    """``(chi_1, chi_2)`` from ``x``: the named aligned spins, or with four
    names the components along the orbital angular momentum ``a_i cos
    tilt_i`` (the coprecessing frame's aligned spins); zeros where absent."""
    n = len(x)
    if spins is None or not set(spins) <= set(x.dtype.names):
        return np.zeros(n), np.zeros(n)
    if len(spins) == 2:
        return (np.asarray(x[spins[0]], dtype=float),
                np.asarray(x[spins[1]], dtype=float))
    return tuple(np.asarray(x[spins[2 * i]], dtype=float)
                 * np.cos(np.asarray(x[spins[2 * i + 1]], dtype=float))
                 for i in (0, 1))
