"""Reparameterisations for spin parameters."""

import inspect

import numpy as np
from nessai.reparameterisations import Reparameterisation
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr, ndtri

from .. import nessai_logger

logger = nessai_logger.getChild(__name__)

_LOG_2PI = np.log(2.0 * np.pi)
#: Newton steps polishing the tabulated inverse CDF
_N_NEWTON = 3


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
        input_parameters=None,
        prior_bounds=None,
        n_interp=2000,
        rng=None,
        **kwargs,
    ):
        parent_params = inspect.signature(
            Reparameterisation.__init__
        ).parameters
        call = dict(parameters=parameters, prior_bounds=prior_bounds, rng=rng)
        # ``input_parameters`` / other kwargs are only accepted by newer nessai
        if (
            input_parameters is not None
            and "input_parameters" in parent_params
        ):
            call["input_parameters"] = input_parameters
        for key, value in kwargs.items():
            if key in parent_params:
                call[key] = value
        super().__init__(**call)

        if len(self.parameters) > 1:
            raise RuntimeError(
                "AlignedSpinReparameterisation only supports one parameter"
            )
        self.prime_parameters = [f"{self.parameters[0]}_prime"]

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
        """Invert the aligned-spin CDF: the lookup table, polished with Newton
        steps on the analytic CDF.

        The table alone is accurate to ~1e-7 in ``chi``, which leaves a ~1e-6
        mismatch between the forward and inverse log-Jacobians; nessai's
        ``verify_rescaling`` rejects that wherever the total log-Jacobian is
        close to zero. A step is kept only where it reduces the residual.
        """
        p = np.clip(p, self._p_min, self._p_max)
        chi = self._inverse_interp(p)
        bound = self.a_max * (1.0 - self._eps)
        chi = np.clip(chi, -bound, bound)
        res = self._cdf(chi) - p
        for _ in range(_N_NEWTON):
            new = np.clip(
                chi - res * np.exp(-self._log_pdf(chi)), -bound, bound
            )
            new_res = self._cdf(new) - p
            better = np.abs(new_res) < np.abs(res)
            chi = np.where(better, new, chi)
            res = np.where(better, new_res, res)
        return chi

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


class EffectiveSpinReparameterisation(Reparameterisation):
    r"""``(chi_1, chi_2) -> (chi_eff_prime, chi_diff_prime)`` for aligned spins.

    The Roulet et al. 2022 (arXiv:2207.03508, Sec. IV) effective-spin
    coordinates, built for bilby's ``AlignedSpin`` prior: at fixed mass ratio
    :math:`q = m_2 / m_1`,

    .. math::
        \mathrm{chi\_eff\_prime} = \Phi^{-1}\left(F_S(\chi_1 + q\chi_2 \mid q)\right),
        \qquad
        \mathrm{chi\_diff\_prime} = \Phi^{-1}\left(F_1(\chi_1 \mid \chi_1 + q\chi_2, q)\right),

    the Rosenblatt transform of the prior in (effective spin, position along
    the line of constant effective spin) order, followed by a probit.  The
    first is a monotonic function of
    :math:`\chi_\mathrm{eff} = (\chi_1 + q\chi_2) / (1 + q)` alone, so a
    well-measured effective spin becomes a single narrow axis rather than the
    curved ridge it traces in the per-spin ``aligned-spin`` coordinates; the
    second is the prior-weighted analogue of the paper's ``cumchidiff``, which
    the data barely constrain.  The prior on both is exactly ``N(0, 1)`` (no
    cusp, no hard edge), as with :class:`AlignedSpinReparameterisation`.  See
    :mod:`nessai_gw._effective_spin` for the numerics.  The map has no
    closed form: the forward pass costs a few quadratures per point
    (~30 us) and the inverse a Newton iteration on them (~150 us), cheap next
    to a waveform evaluation.

    Both coordinates are functions of the spins *and* the mass ratio, which is
    read from x-space on the inverse, so the mass-ratio reparameterisation
    must be inverted first (nessai orders this from ``inverse_input_parameters``).
    Both are invariant under every group action in nessai-gw, which leave the
    intrinsic parameters alone.

    Parameters
    ----------
    parameters : list of str
        The primary and secondary aligned spins, in that order (default
        ``["chi_1", "chi_2"]``).
    prior_bounds : dict
        Their prior bounds; each must be symmetric, ``[-a_max, a_max]``.
    mass_ratio : str, optional
        Name of the mass ratio ``m_2 / m_1 <= 1`` (default ``"mass_ratio"``).
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        mass_ratio="mass_ratio",
        rng=None,
        **kwargs,
    ):
        from .._effective_spin import EffectiveSpinTransform

        parent_params = inspect.signature(
            Reparameterisation.__init__
        ).parameters
        if parameters is None and "input_parameters" in kwargs:
            parameters = kwargs.pop("input_parameters")
        if parameters is None:
            parameters = ["chi_1", "chi_2"]
        call = {"parameters": parameters, "prior_bounds": prior_bounds}
        if "rng" in parent_params:
            call["rng"] = rng
        for key, value in kwargs.items():
            if key in parent_params:
                call[key] = value
        super().__init__(**call)

        if len(self.parameters) != 2:
            raise RuntimeError(
                "EffectiveSpinReparameterisation needs exactly two aligned "
                f"spins (primary, secondary); got {self.parameters}"
            )
        a_max = []
        for name in self.parameters:
            lower, upper = self.prior_bounds[name]
            if not np.isclose(-lower, upper):
                raise RuntimeError(
                    f"{name} prior bounds {(lower, upper)} are not symmetric "
                    "about zero, as the AlignedSpin prior requires."
                )
            a_max.append(float(upper))
        self._transform = EffectiveSpinTransform(*a_max)
        self._mass_ratio = mass_ratio
        self.requires = [mass_ratio]

        self.prime_parameters = ["chi_eff_prime", "chi_diff_prime"]
        if hasattr(self, "output_parameters"):
            self.output_parameters = list(self.prime_parameters)
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + [mass_ratio]
                )
            )

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        chi_1, chi_2 = self.parameters
        u, w, lj = self._transform.forward(
            x[chi_1], x[chi_2], x[self._mass_ratio]
        )
        x_prime[self.prime_parameters[0]] = u
        x_prime[self.prime_parameters[1]] = w
        return x, x_prime, log_j + lj

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        chi_1, chi_2 = self.parameters
        c1, c2, lj = self._transform.inverse(
            x_prime[self.prime_parameters[0]],
            x_prime[self.prime_parameters[1]],
            x[self._mass_ratio],
        )
        x[chi_1] = c1
        x[chi_2] = c2
        return x, x_prime, log_j - lj
