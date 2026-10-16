"""Reparameterisation for the tidal deformabilities of a binary neutron star."""

import inspect

import numpy as np
from nessai.reparameterisations import Reparameterisation
from scipy.special import ndtr, ndtri

from .. import nessai_logger

logger = nessai_logger.getChild(__name__)

_LOG_2PI = np.log(2.0 * np.pi)


def lambda_tilde_ratio(q):
    r"""``r(q)`` with :math:`\tilde\Lambda \propto \Lambda_1 + r(q)\Lambda_2`.

    From Favata 2014 (Eq. 5), with :math:`q = m_2 / m_1 \le 1`:

    .. math::
        \tilde\Lambda = \frac{16}{13}
            \frac{(1 + 12 q)\Lambda_1 + q^4 (q + 12)\Lambda_2}{(1 + q)^5}.
    """
    q = np.asarray(q, dtype=float)
    return q**4 * (q + 12.0) / (1.0 + 12.0 * q)


def lambda_tilde(lambda_1, lambda_2, q):
    r""":math:`\tilde\Lambda` for ``q = m_2 / m_1``."""
    q = np.asarray(q, dtype=float)
    return (
        16.0 / 13.0 * (1.0 + 12.0 * q)
        * (lambda_1 + lambda_tilde_ratio(q) * lambda_2) / (1.0 + q) ** 5
    )


def _probit(lower, upper):
    """``Phi^{-1}(p)`` from ``p = lower`` and ``1 - p = upper``, taking the
    smaller side so both tails keep full relative precision."""
    return np.where(
        lower <= upper,
        ndtri(np.clip(lower, 0.0, 0.5)),
        -ndtri(np.clip(upper, 0.0, 0.5)),
    )


def _log_std_normal(u):
    return -0.5 * u**2 - 0.5 * _LOG_2PI


class EffectiveTidalDeformabilityReparameterisation(Reparameterisation):
    r"""``(lambda_1, lambda_2) -> (lambda_tilde_prime, lambda_diff_prime)``.

    The tidal analogue of
    :class:`~nessai_gw.reparameterisations.spin.EffectiveSpinReparameterisation`
    for independent uniform priors on the two deformabilities: at fixed mass
    ratio :math:`q = m_2 / m_1`,

    .. math::
        \mathrm{lambda\_tilde\_prime} = \Phi^{-1}\left(F_S(S \mid q)\right),
        \qquad
        \mathrm{lambda\_diff\_prime} = \Phi^{-1}\left(F_1(\Lambda_1 \mid S, q)\right),

    with :math:`S = \Lambda_1 + r(q)\Lambda_2` (:func:`lambda_tilde_ratio`):
    the Rosenblatt transform of the prior in (binary deformability, position
    along the line of constant binary deformability) order, followed by a
    probit.  The first is a monotonic function of :math:`\tilde\Lambda` alone
    at fixed ``q``, so a well-measured :math:`\tilde\Lambda` is one narrow
    axis instead of a diagonal ridge across ``lambda_1_prime`` /
    ``lambda_2_prime``; the second is the prior-weighted position along that
    ridge, which the data barely constrain.  The prior on both is exactly
    ``N(0, 1)``.

    Unlike the spins, everything is closed form: ``S`` is the sum of two
    uniforms, so ``F_S`` is the trapezoid CDF (piecewise quadratic, inverted
    with a square root), and the prior along the segment of constant ``S`` is
    uniform, so ``F_1`` is linear.  The map is exactly invertible and its
    Jacobian is

    .. math::
        \left|\frac{\partial(u, w)}{\partial(\Lambda_1, \Lambda_2)}\right|
            = \frac{p_1(\Lambda_1)\, p_2(\Lambda_2)}{\phi(u)\,\phi(w)}.

    Both coordinates depend on the mass ratio, read from x-space on the
    inverse, so the mass-ratio reparameterisation must be inverted first.
    Both are invariant under every group action in nessai-gw.  As with the
    spins, the inverse returns NaN for points with ``q < min_mass_ratio`` or
    non-finite inputs, which nessai then rejects on its prior-bounds check.

    Parameters
    ----------
    parameters : list of str
        The deformabilities of the heavier and lighter star, in that order
        (default ``["lambda_1", "lambda_2"]``).
    prior_bounds : dict
        Their (uniform) prior bounds.
    mass_ratio : str, optional
        Name of the mass ratio ``m_2 / m_1 <= 1`` (default ``"mass_ratio"``).
    min_mass_ratio : float, optional
        Smallest mass ratio the inverse is evaluated at (default ``1e-3``;
        ``r(q)`` scales as ``q^4``).
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        mass_ratio="mass_ratio",
        min_mass_ratio=1e-3,
        rng=None,
        **kwargs,
    ):
        parent_params = inspect.signature(
            Reparameterisation.__init__
        ).parameters
        if parameters is None and "input_parameters" in kwargs:
            parameters = kwargs.pop("input_parameters")
        if parameters is None:
            parameters = ["lambda_1", "lambda_2"]
        call = {"parameters": parameters, "prior_bounds": prior_bounds}
        if "rng" in parent_params:
            call["rng"] = rng
        for key, value in kwargs.items():
            if key in parent_params:
                call[key] = value
        super().__init__(**call)

        if len(self.parameters) != 2:
            raise RuntimeError(
                "EffectiveTidalDeformabilityReparameterisation needs exactly two "
                f"deformabilities (primary, secondary); got {self.parameters}"
            )
        (self.lo_1, self.hi_1), (self.lo_2, self.hi_2) = (
            (float(self.prior_bounds[n][0]), float(self.prior_bounds[n][1]))
            for n in self.parameters
        )
        if not (self.hi_1 > self.lo_1 and self.hi_2 > self.lo_2):
            raise RuntimeError(
                f"Empty deformability prior bounds {self.prior_bounds}"
            )
        self._log_prior = -np.log(self.hi_1 - self.lo_1) - np.log(
            self.hi_2 - self.lo_2
        )
        self._mass_ratio = mass_ratio
        self.min_mass_ratio = float(min_mass_ratio)
        self.requires = [mass_ratio]

        self.prime_parameters = ["lambda_tilde_prime", "lambda_diff_prime"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = list(self.prime_parameters)
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + [mass_ratio]
                )
            )

    def _widths(self, q):
        """``r`` and the widths ``A``, ``B`` of the two uniform summands of
        ``S - S_min``."""
        r = lambda_tilde_ratio(q)
        return r, self.hi_1 - self.lo_1, r * (self.hi_2 - self.lo_2)

    @staticmethod
    def _trapezoid_cdf(s0, a, b):
        """Lower and upper tail of ``s0 = X + Y``, ``X ~ U(0, a)``,
        ``Y ~ U(0, b)``, each computed without cancellation."""
        small, big = np.minimum(a, b), np.maximum(a, b)
        total = a + b
        s0 = np.clip(s0, 0.0, total)
        rise = s0 <= small
        fall = s0 >= big
        lower = np.where(
            rise, s0**2 / (2 * a * b), (s0 - 0.5 * small) / big
        )
        upper = np.where(
            fall, (total - s0) ** 2 / (2 * a * b),
            (total - 0.5 * small - s0) / big,
        )
        lower = np.where(fall, 1.0 - upper, lower)
        upper = np.where(rise, 1.0 - lower, upper)
        return lower, upper

    @staticmethod
    def _trapezoid_inverse(u, a, b):
        """``s0`` with ``Phi^{-1}(F(s0)) = u``, from the smaller tail."""
        small, big = np.minimum(a, b), np.maximum(a, b)
        total = a + b
        corner = 0.5 * small / big  # the CDF at s0 = small
        lower, upper = ndtr(u), ndtr(-u)
        return np.where(
            lower <= corner,
            np.sqrt(2 * a * b * lower),
            np.where(
                upper <= corner,
                total - np.sqrt(2 * a * b * upper),
                np.where(
                    u <= 0, big * lower + 0.5 * small,
                    total - 0.5 * small - big * upper,
                ),
            ),
        )

    def _segment(self, s, r):
        """``lambda_1`` range of the constant-``S`` segment inside the box."""
        return (
            np.maximum(self.lo_1, s - r * self.hi_2),
            np.minimum(self.hi_1, s - r * self.lo_2),
        )

    def _log_jacobian(self, u, w):
        return self._log_prior - _log_std_normal(u) - _log_std_normal(w)

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        l1 = np.asarray(x[self.parameters[0]], dtype=float)
        l2 = np.asarray(x[self.parameters[1]], dtype=float)
        r, a, b = self._widths(x[self._mass_ratio])
        s = l1 + r * l2
        u = _probit(*self._trapezoid_cdf(s - self.lo_1 - r * self.lo_2, a, b))
        lo, hi = self._segment(s, r)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = _probit((l1 - lo) / (hi - lo), (hi - l1) / (hi - lo))
        x_prime[self.prime_parameters[0]] = u
        x_prime[self.prime_parameters[1]] = w
        return x, x_prime, log_j + self._log_jacobian(u, w)

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        u = np.asarray(x_prime[self.prime_parameters[0]], dtype=float)
        w = np.asarray(x_prime[self.prime_parameters[1]], dtype=float)
        q = np.asarray(x[self._mass_ratio], dtype=float)
        valid = (
            np.isfinite(u) & np.isfinite(w) & np.isfinite(q)
            & (q >= self.min_mass_ratio)
        )
        q = np.where(valid, q, 1.0)
        r, a, b = self._widths(q)
        s = self._trapezoid_inverse(u, a, b) + self.lo_1 + r * self.lo_2
        lo, hi = self._segment(s, r)
        l1 = np.where(
            w <= 0, lo + (hi - lo) * ndtr(w), hi - (hi - lo) * ndtr(-w)
        )
        l2 = (s - l1) / r
        lj = self._log_jacobian(u, w)
        x[self.parameters[0]] = np.where(valid, l1, np.nan)
        x[self.parameters[1]] = np.where(valid, l2, np.nan)
        return x, x_prime, log_j - np.where(valid, lj, np.nan)
