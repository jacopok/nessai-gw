"""Reparameterisations for spin parameters."""

import numpy as np
from nessai.reparameterisations import Reparameterisation
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr, ndtri

from .. import nessai_logger

logger = nessai_logger.getChild(__name__)

_LOG_2PI = np.log(2.0 * np.pi)


class AlignedSpinReparameterisation(Reparameterisation):
    r"""Reparameterisation for an aligned (z-component) spin parameter.

    The standard aligned-spin prior (uniform spin magnitude, uniform
    :math:`\cos\theta`, e.g. bilby's ``AlignedSpin``) has a logarithmic cusp
    at :math:`\chi = 0`:

    .. math::
        p(\chi) = -\frac{1}{2 a_\mathrm{max}}
            \ln\left(\frac{|\chi|}{a_\mathrm{max}}\right),
            \quad |\chi| < a_\mathrm{max}.

    This cusp is hard for the normalising flow to model, which degrades the
    acceptance of flow proposals once they are reweighted to the prior.

    This reparameterisation removes the cusp by mapping :math:`\chi` through its
    own CDF and then through the inverse standard-normal CDF, so that the
    prime-space parameter has a **standard normal prior**:

    .. math::
        \chi' = \Phi^{-1}\left(F(\chi)\right),

    where :math:`F` is the aligned-spin CDF and :math:`\Phi` is the standard
    normal CDF. The forward transform and its Jacobian are analytic; the
    inverse map :math:`\chi(\chi')` has no closed form and is evaluated with a
    monotonic spline built once from ``a_max``.

    Parameters
    ----------
    parameters : Union[str, List[str]]
        Name of the aligned-spin parameter (e.g. ``"chi_1"``). Only a single
        parameter is supported.
    prior_bounds : Union[list, dict]
        Prior bounds for the parameter. The maximum spin magnitude is taken to
        be the larger of the two bounds in magnitude and the bounds are assumed
        to be symmetric about zero.
    n_interp : int, optional
        Number of points used per side to build the lookup table for the
        inverse transform.
    rng : numpy.random.Generator, optional
        Random number generator (passed to the parent class).
    """

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        n_interp=2000,
        rng=None,
    ):
        super().__init__(
            parameters=parameters, prior_bounds=prior_bounds, rng=rng
        )

        if len(self.parameters) > 1:
            raise RuntimeError(
                "AlignedSpinReparameterisation only supports one parameter"
            )

        lower, upper = self.prior_bounds[self.parameters[0]]
        self.a_max = float(max(abs(lower), abs(upper)))
        if not np.isclose(-lower, upper):
            logger.warning(
                "Aligned-spin prior bounds %s are not symmetric about zero; "
                "using a_max=%.3f",
                (lower, upper),
                self.a_max,
            )
        if self.a_max <= 0:
            raise RuntimeError("a_max must be positive for aligned spin")

        self.n_interp = int(n_interp)
        self._eps = 1e-12
        self._build_inverse_interpolant()

    def _cdf(self, chi):
        """Aligned-spin CDF evaluated at ``chi``."""
        chi = np.asarray(chi, dtype=float)
        frac = np.clip(np.abs(chi) / self.a_max, self._eps, 1.0)
        # G(|chi|) in [0, 0.5]
        g = 0.5 * frac * (1.0 - np.log(frac))
        return 0.5 + np.sign(chi) * g

    def _log_pdf(self, chi):
        """Log of the aligned-spin PDF evaluated at ``chi``."""
        frac = np.clip(np.abs(chi) / self.a_max, self._eps, 1.0 - self._eps)
        return np.log(-np.log(frac)) - np.log(2.0 * self.a_max)

    def _build_inverse_interpolant(self):
        """Build a monotonic lookup table for ``chi`` as a function of the
        CDF value ``p``.
        """
        # Cluster points near |chi| = 0 (where the PDF diverges) and near
        # |chi| = a_max (where the PDF vanishes and the CDF is flat).
        half = np.geomspace(self._eps, 0.5, self.n_interp // 2)
        q = np.unique(np.concatenate([half, 1.0 - half[::-1]]))
        q = q[q < 1.0 - 1e-15]
        chi_pos = self.a_max * q
        chi_grid = np.concatenate([-chi_pos[::-1], [0.0], chi_pos])
        p_grid = self._cdf(chi_grid)
        # Ensure strict monotonicity for the interpolator.
        p_grid[len(chi_pos)] = 0.5
        p_grid, idx = np.unique(p_grid, return_index=True)
        chi_grid = chi_grid[idx]
        self._p_min = float(p_grid[0])
        self._p_max = float(p_grid[-1])
        self._inverse_interp = PchipInterpolator(
            p_grid, chi_grid, extrapolate=False
        )

    def _inverse_cdf(self, p):
        """Invert the aligned-spin CDF using the lookup table."""
        p = np.clip(p, self._p_min, self._p_max)
        chi = self._inverse_interp(p)
        bound = self.a_max * (1.0 - self._eps)
        return np.clip(chi, -bound, bound)

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert from x-space to x'-space (chi -> standard normal)."""
        chi = x[self.parameters[0]]
        u = ndtri(self._cdf(chi))
        x_prime[self.prime_parameters[0]] = u
        # log|du/dchi| = log p(chi) - log phi(u)
        log_j = log_j + (self._log_pdf(chi) + 0.5 * u**2 + 0.5 * _LOG_2PI)
        return x, x_prime, log_j

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        """Convert from x'-space back to x-space (standard normal -> chi)."""
        u = x_prime[self.prime_parameters[0]]
        chi = self._inverse_cdf(ndtr(u))
        x[self.parameters[0]] = chi
        # log|dchi/du| = log phi(u) - log p(chi)
        log_j = log_j - (self._log_pdf(chi) + 0.5 * u**2 + 0.5 * _LOG_2PI)
        return x, x_prime, log_j
