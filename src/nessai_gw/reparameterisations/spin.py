"""Reparameterisations for spin parameters."""

import inspect

import numpy as np
from nessai.reparameterisations import Reparameterisation
from scipy.interpolate import PchipInterpolator
from scipy.special import log_ndtr, ndtr, ndtri

from .. import nessai_logger

logger = nessai_logger.getChild(__name__)

_LOG_2PI = np.log(2.0 * np.pi)
#: Newton steps polishing the tabulated inverse CDF
_N_NEWTON = 3

#: The charts of the aligned spins in :class:`EffectiveSpinReparameterisation`
#: and :class:`PrecessingSpinReparameterisation` (``spin_chart``): the
#: effective spin's prior CDF (``"effective"``), or the 1.5PN phase
#: coefficient (``"spin-orbit"``,
#: :class:`~nessai_gw._effective_spin_table.SpinOrbitChart`).
SPIN_CHARTS = ("effective", "spin-orbit")


def _with_spin_chart(transform, chart):
    """``transform`` (an effective-spin transform, or a
    :class:`~nessai_gw._effective_spin_table.SpinOrbitChart` of one) in the
    chart ``chart``."""
    from .._effective_spin_table import (
        SpinOrbitChart,
        TabulatedEffectiveSpinTransform,
    )

    if chart not in SPIN_CHARTS:
        raise ValueError(f"spin_chart must be one of {SPIN_CHARTS}, not {chart!r}")
    inner = transform.inner if isinstance(transform, SpinOrbitChart) else transform
    if chart == "effective":
        return inner
    if not isinstance(inner, TabulatedEffectiveSpinTransform):
        raise ValueError("the spin-orbit chart needs the tabulated effective-spin "
                         "transform (tabulated=True)")
    return SpinOrbitChart(inner)


class _SpinChartMixin:
    """``spin_chart``: read from, and set on, the effective-spin transform
    (so a proposal pickled before the option is in the ``effective`` chart,
    and a resumed one can be switched)."""

    @property
    def spin_chart(self):
        from .._effective_spin_table import SpinOrbitChart

        if isinstance(self._transform, SpinOrbitChart):
            return "spin-orbit"
        return "effective"

    @spin_chart.setter
    def spin_chart(self, chart):
        self._transform = _with_spin_chart(self._transform, chart)


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


class EffectiveSpinReparameterisation(_SpinChartMixin, Reparameterisation):
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
    intrinsic parameters alone (the only spin parameter they act on is the
    precessing ``phi_jl``; see :class:`PrecessingSpinReparameterisation`).

    The flow occasionally proposes a mass ratio far outside its prior (with
    ``log-mass-ratio``, ``q = exp(ln q)`` underflows to zero), where
    ``chi_2 = (S - chi_1) / q`` is undetermined and the quadratures divide by
    zero.  The inverse skips points with ``q < min_mass_ratio`` or non-finite
    inputs and returns NaN spins and log-Jacobian for them; they lie outside
    any sensible mass-ratio prior, so nessai rejects them on its prior-bounds
    check as it would anyway.

    Parameters
    ----------
    parameters : list of str
        The primary and secondary aligned spins, in that order (default
        ``["chi_1", "chi_2"]``).
    prior_bounds : dict
        Their prior bounds; each must be symmetric, ``[-a_max, a_max]``.
    mass_ratio : str, optional
        Name of the mass ratio ``m_2 / m_1 <= 1`` (default ``"mass_ratio"``).
    min_mass_ratio : float, optional
        Smallest mass ratio the inverse is evaluated at (default ``1e-6``;
        the round trip holds to ~1e-8 in the log-Jacobian there and degrades
        below it).  Must not exceed the lower bound of the mass-ratio prior.
    tabulated : bool, optional
        Use :class:`~nessai_gw._effective_spin_table.TabulatedEffectiveSpinTransform`:
        the CDFs are tabulated once and the map is defined by the tables, so
        it is exactly invertible with an exact Jacobian but the prior on the
        new coordinates is only approximately ``N(0, I)`` (rms ~0.01 from
        the exact transform); its inverse is ~3x cheaper.  Default ``True``.
    spin_chart : {"effective", "spin-orbit"}, optional
        ``"spin-orbit"`` (needs ``tabulated``): ``chi_eff_prime`` is instead
        an affine function of the 1.5PN phase coefficient, linear in the
        spin-orbit combination :math:`\beta = a_1 \chi_1 + a_2 \chi_2` and
        not normal under the prior, and ``chi_diff_prime`` the position
        along the line of constant beta
        (:class:`~nessai_gw._effective_spin_table.SpinOrbitChart`).  On the
        hardest checkpoints of 40 pop250 BNS runs it cut the likelihood
        calls per new point by 12-15% (paired seeds, several flows), at the
        same coverage.  Default ``"effective"``.
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        mass_ratio="mass_ratio",
        min_mass_ratio=1e-6,
        tabulated=True,
        spin_chart="effective",
        rng=None,
        **kwargs,
    ):
        from .._effective_spin import EffectiveSpinTransform
        from .._effective_spin_table import TabulatedEffectiveSpinTransform

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
        self.tabulated = bool(tabulated)
        if self.tabulated:
            self._transform = TabulatedEffectiveSpinTransform(*a_max)
        else:
            self._transform = EffectiveSpinTransform(*a_max)
        self.spin_chart = spin_chart
        self._mass_ratio = mass_ratio
        self.min_mass_ratio = float(min_mass_ratio)
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
        u = x_prime[self.prime_parameters[0]]
        w = x_prime[self.prime_parameters[1]]
        q = x[self._mass_ratio]
        valid = (
            np.isfinite(u) & np.isfinite(w) & np.isfinite(q)
            & (q >= self.min_mass_ratio)
        )
        c1 = np.full(len(q), np.nan)
        c2 = np.full(len(q), np.nan)
        lj = np.full(len(q), np.nan)
        if valid.any():
            # beyond the tables' resolution (|u| or |w| >~ 8) the spins land
            # on the prior box edge, where the log-Jacobian can be +-inf or
            # nan; a -inf at an interior point would give an infinite
            # rejection weight, so treat any such point as out of bounds
            with np.errstate(divide="ignore", invalid="ignore"):
                c1[valid], c2[valid], lj[valid] = self._transform.inverse(
                    u[valid], w[valid], q[valid]
                )
            bad = ~np.isfinite(lj)
            c1[bad] = np.nan
            c2[bad] = np.nan
        x[chi_1] = c1
        x[chi_2] = c2
        return x, x_prime, log_j - lj


#: Smallest ``|chi_z|`` and spin magnitude the precessing-spin chart uses
#: (both are zero only on sets of zero prior measure).
_TINY = 1e-300

#: The charts of :class:`PrecessingSpinReparameterisation`.
PRECESSING_SPIN_COORDINATES = ("cogwheel", "cartesian")
#: Prime names of the ``cogwheel`` chart's two azimuths, angles in
#: ``[-pi, pi)`` (periodic dimensions of the flow).
PRECESSING_SPIN_ANGLES = ("phi_jl_angle", "phi_12_angle")
_PRIME_PARAMETERS = {
    "cogwheel": ("chi_eff_prime", "chi_diff_prime", "r1_prime", "r2_prime",
                 *PRECESSING_SPIN_ANGLES),
    "cartesian": ("chi_eff_prime", "chi_diff_prime", "s1_x", "s1_y", "s2_x",
                  "s2_y"),
}


def _in_plane_forward(a, tilt, a_max):
    r"""The in-plane radius ``rho`` of :class:`PrecessingSpinReparameterisation`
    and ``log |d(chi_z, s_x, s_y) / d(a, tilt, phi)|``.

    ``u = ln(a / |chi_z|) / ln(a_max / |chi_z|)`` and
    ``rho = sqrt(-2 ln(1 - u))``; the log-Jacobian is
    ``log sin(tilt) - log ln(a_max / a)``.
    """
    a = np.maximum(a, _TINY)
    sin2 = np.sin(tilt) ** 2
    abs_cos = np.maximum(np.abs(np.cos(tilt)), _TINY)
    log_max_over_a = np.log(a_max / a)
    with np.errstate(divide="ignore", invalid="ignore"):
        # ln(a / |chi_z|) = -ln|cos tilt|, accurately near tilt = 0, pi
        log_ratio = np.where(
            sin2 < 0.5, -0.5 * np.log1p(-sin2), -np.log(abs_cos)
        )
        total = log_max_over_a + log_ratio  # ln(a_max / |chi_z|)
        u = log_ratio / total
        # 1 - u = ln(a_max / a) / ln(a_max / |chi_z|), directly where u ~ 1
        log_one_minus_u = np.where(
            u < 0.5, np.log1p(-u), np.log(log_max_over_a / total)
        )
        rho = np.sqrt(np.maximum(-2.0 * log_one_minus_u, 0.0))
        log_j = 0.5 * np.log(sin2) - np.log(log_max_over_a)
    return rho, log_j


def _in_plane_inverse(chi_z, rho, a_max):
    """Inverse of :func:`_in_plane_forward` at fixed ``chi_z``: ``(a, tilt,
    log_j)``, with ``log_j`` the forward log-Jacobian."""
    abs_z = np.maximum(np.abs(chi_z), _TINY)
    total = np.log(a_max / abs_z)  # ln(a_max / |chi_z|)
    half_rho2 = 0.5 * rho**2
    log_max_over_a = np.exp(-half_rho2) * total  # (1 - u) ln(a_max / |chi_z|)
    log_a_over_z = -np.expm1(-half_rho2) * total  # u ln(a_max / |chi_z|)
    a = a_max * np.exp(-log_max_over_a)
    # a^2 - chi_z^2, accurately where a ~ |chi_z|
    in_plane2 = abs_z * np.expm1(log_a_over_z) * (a + abs_z)
    tilt = np.arctan2(np.sqrt(in_plane2), chi_z)
    with np.errstate(divide="ignore"):
        log_j = 0.5 * np.log(in_plane2) - np.log(a) - np.log(log_max_over_a)
    return a, tilt, log_j


class PrecessingSpinReparameterisation(_SpinChartMixin, Reparameterisation):
    r"""Precessing spins ``(a_1, tilt_1, a_2, tilt_2, phi_12, phi_jl) ->
    (chi_eff_prime, chi_diff_prime, r1_prime, r2_prime, phi_jl_angle,
    phi_12_angle)`` (``cogwheel``, the default) or ``(chi_eff_prime,
    chi_diff_prime, s1_x, s1_y, s2_x, s2_y)`` (``cartesian``).

    For the usual prior on precessing spins, magnitudes uniform in
    ``[0, a_max]`` and isotropic directions (``tilt_i`` sine-distributed,
    ``phi_12`` and ``phi_jl`` uniform in ``[0, 2 pi)``), the six new
    coordinates are a standard normal under the prior (to the ~0.01 of the
    tabulated effective-spin map).

    * The spin components along the orbital angular momentum, ``chi_iz = a_i
      cos(tilt_i)``, are distributed as bilby's ``AlignedSpin``; they become
      ``chi_eff_prime``, ``chi_diff_prime`` exactly as in
      :class:`EffectiveSpinReparameterisation` (tabulated), so the
      well-measured effective spin is still a single axis.
    * Given ``chi_iz``, the in-plane magnitude has the prior CDF
      :math:`u_i = \ln(a_i / |\chi_{iz}|) / \ln(a_\mathrm{max} / |\chi_{iz}|)`;
      the Box-Muller radius :math:`\rho_i = \sqrt{-2 \ln(1 - u_i)}` with an
      azimuth makes a 2D standard normal,
      ``(s_x, s_y) = rho (cos phi, sin phi)``. The azimuth is ``phi_jl`` for
      the first spin and ``phi_12`` for the second. ``rho`` is proportional to
      the in-plane spin near zero, so the ``cartesian`` chart is smooth there
      and the angles need no periodic boundaries.
    * The ``cogwheel`` chart (Roulet et al. 2022, arXiv:2207.03508, whose
      ``cums1r_s1z``, ``cums2r_s2z`` are the ``u_i``) keeps the polar
      coordinates instead: ``r_i = Phi^{-1}(u_i)``, a standard normal, and
      the azimuths as angles in ``[-pi, pi)``, ``phi_12_angle = phi_12`` and
      ``phi_jl_angle = phi_jl + pi [theta_jn > pi / 2]``, uniform under the
      prior.  The flow must treat the angles as periodic.  The shift of
      ``phi_jl`` with the inclination is the paper's: the improper elements of
      every nessai-gw group send ``theta_jn -> pi - theta_jn`` together with
      ``phi_jl -> phi_jl + pi`` (see
      :class:`~nessai_gw.group_mixture.TriangularDetectorGroupAction`), which
      leaves ``phi_jl_angle`` unchanged, so the chart is invariant under every
      group action.  It reads ``theta_jn`` (``inclination``) from x-space, on
      the inverse too.  Measured on the PS noorb runs' live points, a fresh
      flow fits it ~0.5 nats better (KL to the live points) than the
      ``cartesian`` chart (xG_inference studies/noorb_runs/proposal_benchmark.py).

    The log-Jacobian is that of the effective spins plus
    :math:`\sum_i [\log \sin \mathrm{tilt}_i - \log \ln(a_{\mathrm{max}, i} /
    a_i)]` (``cartesian``), and for ``cogwheel`` that of
    ``(s_x, s_y) -> (r_i, phi)``,
    :math:`\sum_i [-\rho_i^2 / 2 - \log \varphi(r_i)]`.

    The ``cartesian`` coordinates are invariant under the sky rotations of the
    triangular detector group; under its plane reflection ``phi_jl -> phi_jl +
    pi``, i.e. ``(s1_x, s1_y) -> -(s1_x, s1_y)``.

    Parameters
    ----------
    parameters : list of str
        ``[a_1, tilt_1, a_2, tilt_2, phi_12, phi_jl]`` (the default), in this
        order.
    prior_bounds : dict
        Their prior bounds; the magnitudes' must be ``[0, a_max]``.
    mass_ratio : str, optional
        Name of the mass ratio ``m_2 / m_1 <= 1`` (default ``"mass_ratio"``),
        read from x-space on the inverse.
    min_mass_ratio : float, optional
        As for :class:`EffectiveSpinReparameterisation`.
    coordinates : {"cogwheel", "cartesian"}, optional
        The chart (above); default ``"cogwheel"``.
    inclination : str, optional
        Name of ``theta_jn``, read by the ``cogwheel`` chart (default
        ``"theta_jn"``).
    spin_chart : {"effective", "spin-orbit"}, optional
        The chart of the aligned components ``(chi_eff_prime,
        chi_diff_prime)``, as for :class:`EffectiveSpinReparameterisation`
        (default ``"effective"``).  The in-plane coordinates are the same in
        either.
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        mass_ratio="mass_ratio",
        min_mass_ratio=1e-6,
        coordinates="cogwheel",
        inclination="theta_jn",
        spin_chart="effective",
        rng=None,
        **kwargs,
    ):
        from .._effective_spin_table import TabulatedEffectiveSpinTransform

        if coordinates not in PRECESSING_SPIN_COORDINATES:
            raise ValueError(
                f"coordinates must be one of {PRECESSING_SPIN_COORDINATES}, "
                f"not {coordinates!r}"
            )

        parent_params = inspect.signature(
            Reparameterisation.__init__
        ).parameters
        if parameters is None and "input_parameters" in kwargs:
            parameters = kwargs.pop("input_parameters")
        if parameters is None:
            parameters = ["a_1", "tilt_1", "a_2", "tilt_2", "phi_12", "phi_jl"]
        call = {"parameters": parameters, "prior_bounds": prior_bounds}
        if "rng" in parent_params:
            call["rng"] = rng
        for key, value in kwargs.items():
            if key in parent_params:
                call[key] = value
        super().__init__(**call)

        if len(self.parameters) != 6:
            raise RuntimeError(
                "PrecessingSpinReparameterisation needs [a_1, tilt_1, a_2, "
                f"tilt_2, phi_12, phi_jl]; got {self.parameters}"
            )
        self.a_max = []
        for name in self.parameters[0:4:2]:
            lower, upper = self.prior_bounds[name]
            if lower != 0 or upper <= 0:
                raise RuntimeError(
                    f"{name} prior bounds {(lower, upper)} are not [0, a_max]"
                )
            self.a_max.append(float(upper))
        for name, bounds in zip(
            self.parameters[1:],
            [(0, np.pi), None, (0, np.pi), (0, 2 * np.pi), (0, 2 * np.pi)],
        ):
            if bounds is not None and not np.allclose(
                self.prior_bounds[name], bounds
            ):
                logger.warning(
                    "%s prior bounds %s are not %s, which the "
                    "precessing-spin prior assumes",
                    name, tuple(self.prior_bounds[name]), bounds,
                )
        self._transform = TabulatedEffectiveSpinTransform(*self.a_max)
        self.spin_chart = spin_chart
        self._mass_ratio = mass_ratio
        self.min_mass_ratio = float(min_mass_ratio)
        self.coordinates = coordinates
        self._inclination = inclination
        reads = [mass_ratio]
        if coordinates == "cogwheel":
            reads.append(inclination)
        self.requires = list(reads)

        self.prime_parameters = list(_PRIME_PARAMETERS[coordinates])
        if hasattr(self, "output_parameters"):
            self.output_parameters = list(self.prime_parameters)
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(list(self.inverse_input_parameters or []) + reads)
            )

    def __setstate__(self, state):
        # pickled before the cogwheel chart: the Cartesian one
        state.setdefault("coordinates", "cartesian")
        state.setdefault("_inclination", "theta_jn")
        self.__dict__.update(state)

    def _spins(self):
        """``(magnitude, tilt, azimuth, (x, y) prime names, a_max)`` per spin:
        for ``cogwheel`` the names are ``(r_i, azimuth angle)``."""
        a_1, tilt_1, a_2, tilt_2, phi_12, phi_jl = self.parameters
        if self.coordinates == "cogwheel":
            names = (("r1_prime", "phi_jl_angle"), ("r2_prime", "phi_12_angle"))
        else:
            names = (("s1_x", "s1_y"), ("s2_x", "s2_y"))
        return (
            (a_1, tilt_1, phi_jl, names[0], self.a_max[0]),
            (a_2, tilt_2, phi_12, names[1], self.a_max[1]),
        )

    def _azimuth_shifts(self, x):
        """What the ``cogwheel`` chart adds to ``(phi_jl, phi_12)``."""
        if self.coordinates != "cogwheel":
            return 0.0, 0.0
        return np.pi * (x[self._inclination] > 0.5 * np.pi), 0.0

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        (a_1, tilt_1, *_), (a_2, tilt_2, *_) = self._spins()
        chi_1z = x[a_1] * np.cos(x[tilt_1])
        chi_2z = x[a_2] * np.cos(x[tilt_2])
        u, w, lj = self._transform.forward(chi_1z, chi_2z, x[self._mass_ratio])
        x_prime["chi_eff_prime"] = u
        x_prime["chi_diff_prime"] = w
        for (a, tilt, phi, (sx, sy), a_max), shift in zip(
            self._spins(), self._azimuth_shifts(x)
        ):
            rho, lj_i = _in_plane_forward(x[a], x[tilt], a_max)
            if self.coordinates == "cogwheel":
                # u_i = 1 - exp(-rho^2 / 2); (s_x, s_y) -> (rho, phi) ->
                # (Phi^-1(u_i), phi): -log rho + log(rho e^{-rho^2/2}) - log N
                half_rho2 = 0.5 * rho**2
                with np.errstate(divide="ignore"):
                    r = np.where(
                        half_rho2 < np.log(2.0),
                        ndtri(-np.expm1(-half_rho2)),
                        -ndtri(np.exp(-half_rho2)),
                    )
                x_prime[sx] = r
                x_prime[sy] = (
                    np.remainder(x[phi] + shift + np.pi, 2 * np.pi) - np.pi
                )
                lj_i = lj_i - half_rho2 + 0.5 * r**2 + 0.5 * _LOG_2PI
            else:
                x_prime[sx] = rho * np.cos(x[phi])
                x_prime[sy] = rho * np.sin(x[phi])
            lj = lj + lj_i
        return x, x_prime, log_j + lj

    def _radius_azimuth(self, x_prime, names):
        """``(rho, azimuth before the shift)`` of one spin from x'."""
        sx, sy = x_prime[names[0]], x_prime[names[1]]
        if self.coordinates == "cogwheel":
            # 1 - u_i = Phi(-r_i) = exp(-rho^2 / 2)
            return np.sqrt(np.maximum(-2.0 * log_ndtr(-sx), 0.0)), sy
        return np.hypot(sx, sy), np.arctan2(sy, sx)

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        u = x_prime["chi_eff_prime"]
        w = x_prime["chi_diff_prime"]
        q = x[self._mass_ratio]
        n = len(q)
        in_plane = [
            np.stack([x_prime[sx], x_prime[sy]])
            for *_, (sx, sy), _ in self._spins()
        ]
        valid = (
            np.isfinite(u) & np.isfinite(w) & np.isfinite(q)
            & (q >= self.min_mass_ratio)
            & np.all(np.isfinite(np.concatenate(in_plane)), axis=0)
        )
        out = {name: np.full(n, np.nan) for name in self.parameters}
        lj = np.full(n, np.nan)
        if valid.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                chi_1z, chi_2z, lj_v = self._transform.inverse(
                    u[valid], w[valid], q[valid]
                )
                shifts = self._azimuth_shifts(x)
                for (a, tilt, phi, names, a_max), z, shift in zip(
                    self._spins(), (chi_1z, chi_2z), shifts
                ):
                    rho, azimuth = self._radius_azimuth(x_prime[valid], names)
                    a_v, tilt_v, lj_i = _in_plane_inverse(z, rho, a_max)
                    out[a][valid] = a_v
                    out[tilt][valid] = tilt_v
                    shift = shift[valid] if np.ndim(shift) else shift
                    out[phi][valid] = np.remainder(azimuth - shift, 2 * np.pi)
                    if self.coordinates == "cogwheel":
                        r = x_prime[names[0]][valid]
                        lj_i = (lj_i - 0.5 * rho**2 + 0.5 * r**2
                                + 0.5 * _LOG_2PI)
                    lj_v = lj_v + lj_i
            lj[valid] = lj_v
            # as EffectiveSpinReparameterisation: a non-finite log-Jacobian
            # (beyond the tables' resolution, or on the prior's edge) is out
            # of bounds
            bad = ~np.isfinite(lj)
            for name in self.parameters:
                out[name][bad] = np.nan
        for name in self.parameters:
            x[name] = out[name]
        return x, x_prime, log_j - lj
