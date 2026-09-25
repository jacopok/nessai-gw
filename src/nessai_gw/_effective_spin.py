"""Effective-spin coordinates for two aligned spins under the aligned-spin prior.

Roulet et al. 2022 (arXiv:2207.03508, Sec. IV) sample aligned spins as the
well-measured effective spin

    chi_eff = (chi_1 + q chi_2) / (1 + q),      q = m_2 / m_1 <= 1,

plus ``cumchidiff``, the conditional cumulative of the *prior* along the
orthogonal direction (the line of constant ``chi_eff`` at fixed ``q``), which
the data barely constrain.  Their prior is flat in ``chi_eff`` and uniform
along that line, so ``cumchidiff`` is a linear function there (cogwheel's
``UniformEffectiveSpinPrior``).  Here the prior is fixed by the run -- bilby's
``AlignedSpin`` (uniform magnitude, isotropic direction), with its logarithmic
cusp at ``chi = 0`` -- so both coordinates are built as the Rosenblatt
transform of *that* prior, conditional on ``q``, followed by a probit:

    u = Phi^{-1}( F_S(S | q) ),                    S = chi_1 + q chi_2,
    w = Phi^{-1}( F_1(chi_1 | S, q) ),

``F_S`` the prior CDF of ``S`` and ``F_1`` the prior CDF of ``chi_1`` along the
segment of constant ``S``.  ``u`` is a monotonic function of ``chi_eff`` alone
(at fixed ``q``) and ``w`` is the prior-weighted ``cumchidiff``.  The prior on
``(u, w)`` is exactly ``N(0, I)`` whatever ``q`` -- no cusp, no hard edge --
and the Jacobian is simply

    |d(u, w) / d(chi_1, chi_2)| = p_1(chi_1) p_2(chi_2) / (phi(u) phi(w)).

There is no closed form for ``F_S`` or ``F_1`` (a convolution of two
log-cusped densities), so they are one-dimensional integrals of
``p_1(x) p_2((S - x) / q)`` along the segment, done with tanh-sinh quadrature
split at the log singularities (``chi_1 = 0``, ``chi_2 = 0``) so that every
singularity is an endpoint.  That is accurate to ~1e-9.  The inverse is a
safeguarded Newton iteration on the same integrals, so forward and inverse
agree to the same precision.  Tails are taken from whichever side is smaller
(using the prior's ``chi -> -chi`` symmetry for ``F_S``), so ``u, w`` keep full
relative precision far from zero.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, ndtr, ndtri

_LOG_2PI = np.log(2.0 * np.pi)

# tanh-sinh nodes on [-1, 1]: t = k h, y = (pi / 2) sinh t.  With h = 0.15 and
# |t| <= 3.15 the endpoint distances reach ~1e-16 of the interval, which
# resolves the log singularities to ~1e-10.
_TS_H = 0.15
_TS_T = np.arange(-21, 22) * _TS_H
_TS_Y = 0.5 * np.pi * np.sinh(_TS_T)
_TS_W = _TS_H * 0.5 * np.pi * np.cosh(_TS_T) / np.cosh(_TS_Y) ** 2
#: distance of each node from the left / right endpoint, as a fraction of
#: the interval (computed without cancellation)
_TS_FL = expit(2.0 * _TS_Y)
_TS_FR = expit(-2.0 * _TS_Y)
_TS_LEFT = _TS_Y < 0

_NEWTON_MAX = 60
_NEWTON_TOL = 1e-12


def aligned_spin_log_pdf(chi, a_max):
    """Log-density of bilby's ``AlignedSpin`` prior (uniform magnitude on
    ``[0, a_max]``, isotropic direction): ``-ln(|chi| / a_max) / (2 a_max)``."""
    frac = np.abs(chi) / a_max
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(-np.log(frac)) - np.log(2.0 * a_max)


def _aligned_spin_pdf(chi, a_max):
    frac = np.abs(chi) / a_max
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.log(frac) / (2.0 * a_max)
    return np.where(frac < 1.0, out, 0.0)


def aligned_spin_cdf(chi, a_max):
    """CDF of bilby's ``AlignedSpin`` prior."""
    frac = np.clip(np.abs(chi) / a_max, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(frac > 0, 0.5 * frac * (1.0 - np.log(frac)), 0.0)
    return 0.5 + np.sign(chi) * g


def _log_std_normal(u):
    return -0.5 * u**2 - 0.5 * _LOG_2PI


class EffectiveSpinTransform:
    """``(chi_1, chi_2) <-> (u, w)`` at given mass ratio(s) ``q = m_2 / m_1``.

    Parameters
    ----------
    a_max_1, a_max_2 : float
        Maximum spin magnitudes of the two ``AlignedSpin`` priors.
    """

    def __init__(self, a_max_1, a_max_2):
        self.a1 = float(a_max_1)
        self.a2 = float(a_max_2)
        if self.a1 <= 0 or self.a2 <= 0:
            raise ValueError("a_max must be positive")

    # ------------------------------------------------------------------
    # quadrature
    # ------------------------------------------------------------------
    def _integrate(self, integrand, lo, hi, breaks):
        """``int_lo^hi integrand`` per point, split at ``breaks``.

        ``integrand(chi_1, d_s)`` receives ``chi_1`` and ``S - chi_1``, both
        computed from the nearer subinterval endpoint so that a node close to
        a breakpoint at ``chi_1 = 0`` or ``chi_1 = S`` keeps full relative
        precision.
        """
        lo = np.asarray(lo, dtype=float)
        hi = np.maximum(np.asarray(hi, dtype=float), lo)
        pts = [lo] + [np.clip(b, lo, hi) for b in breaks] + [hi]
        pts = np.sort(np.stack(pts, axis=-1), axis=-1)
        left, right = pts[..., :-1, None], pts[..., 1:, None]
        width = right - left
        dl, dr = width * _TS_FL, width * _TS_FR
        chi1 = np.where(_TS_LEFT, left + dl, right - dr)
        s = self._s[..., None, None]
        d_s = np.where(_TS_LEFT, (s - left) - dl, (s - right) + dr)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = integrand(chi1, d_s)
        f = np.where(width > 0, f, 0.0)
        return 0.5 * np.sum(width * _TS_W * f, axis=(-2, -1))

    def _setup(self, s, q):
        self._s = np.asarray(s, dtype=float)
        self._q = np.asarray(q, dtype=float)

    def _joint(self, chi1, d_s):
        q = self._q[..., None, None]
        return (
            _aligned_spin_pdf(chi1, self.a1)
            * _aligned_spin_pdf(d_s / q, self.a2)
        )

    def _segment(self):
        """``chi_1`` range of the constant-``S`` segment inside the prior box."""
        s, q = self._s, self._q
        return (
            np.maximum(-self.a1, s - q * self.a2),
            np.minimum(self.a1, s + q * self.a2),
        )

    def _lower_cdf_s(self):
        """``P(S' <= S)`` for ``S <= 0`` (use the ``chi -> -chi`` symmetry for
        ``S > 0``)."""
        s, q = self._s, self._q
        a1, a2 = self.a1, self.a2

        def f(chi1, d_s):
            qq = q[..., None, None]
            return _aligned_spin_pdf(chi1, a1) * aligned_spin_cdf(d_s / qq, a2)

        return self._integrate(
            f, np.full_like(s, -a1), np.minimum(a1, s + q * a2),
            [np.zeros_like(s), s, s - q * a2],
        )

    def _z(self):
        lo, hi = self._segment()
        return self._integrate(self._joint, lo, hi, [np.zeros_like(lo), self._s])

    def _g_split(self, chi1):
        lo, hi = self._segment()
        chi1 = np.clip(chi1, lo, hi)
        zero = np.zeros_like(lo)
        lower = self._integrate(self._joint, lo, chi1, [zero, self._s])
        upper = self._integrate(self._joint, chi1, hi, [zero, self._s])
        return lower, upper

    # ------------------------------------------------------------------
    # the two coordinates
    # ------------------------------------------------------------------
    def u_of_s(self, s, q, density=True):
        """``Phi^{-1}(F_S(S | q))`` and (optionally) the density
        ``p_S(S | q)``."""
        s = np.asarray(s, dtype=float)
        q = np.asarray(q, dtype=float)
        self._setup(-np.abs(s), q)
        v = ndtri(np.clip(self._lower_cdf_s(), 0.0, 0.5))
        u = np.where(s > 0, -v, v)
        if not density:
            return u, None
        self._setup(s, q)
        return u, self._z() / q

    def w_of_chi1(self, chi1, s, q):
        """``Phi^{-1}(F_1(chi_1 | S, q))`` and the conditional density."""
        self._setup(s, q)
        lower, upper = self._g_split(chi1)
        z = lower + upper
        w = np.where(
            lower <= upper,
            ndtri(np.clip(lower / z, 0.0, 0.5)),
            -ndtri(np.clip(upper / z, 0.0, 0.5)),
        )
        return w, z

    def forward(self, chi1, chi2, q):
        """``(u, w, log|d(u, w)/d(chi_1, chi_2)|)``."""
        chi1 = np.asarray(chi1, dtype=float)
        chi2 = np.asarray(chi2, dtype=float)
        q = np.asarray(q, dtype=float)
        s = chi1 + q * chi2
        u, _ = self.u_of_s(s, q, density=False)
        w, _ = self.w_of_chi1(chi1, s, q)
        return u, w, self.log_jacobian(chi1, chi2, u, w)

    def log_jacobian(self, chi1, chi2, u, w):
        return (
            aligned_spin_log_pdf(chi1, self.a1)
            + aligned_spin_log_pdf(chi2, self.a2)
            - _log_std_normal(u)
            - _log_std_normal(w)
        )

    def inverse(self, u, w, q):
        """``(chi_1, chi_2, log|d(u, w)/d(chi_1, chi_2)|)`` (the *forward*
        log-Jacobian at the returned point)."""
        u = np.asarray(u, dtype=float)
        w = np.asarray(w, dtype=float)
        q = np.asarray(q, dtype=float)
        u, w, q = np.broadcast_arrays(u, w, q)
        s_max = self.a1 + q * self.a2
        sd = np.sqrt((self.a1**2 + (q * self.a2) ** 2) / 9.0)
        s = self._newton(
            lambda s_, i: self.u_of_s(s_, q[i]),
            u, np.clip(u * sd, -0.99 * s_max, 0.99 * s_max),
            -s_max, s_max,
        )
        self._setup(s, q)
        lo, hi = self._segment()
        chi1 = self._newton(
            lambda c, i: self._w_and_density(c, s[i], q[i]),
            w, lo + (hi - lo) * ndtr(w), lo, hi,
        )
        chi2 = (s - chi1) / q
        return chi1, chi2, self.log_jacobian(chi1, chi2, u, w)

    def _w_and_density(self, chi1, s, q):
        w, z = self.w_of_chi1(chi1, s, q)
        dens = (
            _aligned_spin_pdf(chi1, self.a1)
            * _aligned_spin_pdf((s - chi1) / q, self.a2)
            / z
        )
        return w, dens

    @staticmethod
    def _newton(fn, target, x0, lo, hi):
        """Safeguarded Newton for the increasing ``y(x) = Phi^{-1}(CDF(x))``.

        ``fn(x, idx)`` returns ``(y, density)`` at the points ``idx`` (so
        ``dy/dx = density / phi(y)``); only unconverged points are evaluated.
        Steps leaving the running bracket fall back to bisection.
        """
        x = np.array(x0, dtype=float).ravel()
        lo = np.array(np.broadcast_to(lo, x.shape), dtype=float).ravel()
        hi = np.array(np.broadcast_to(hi, x.shape), dtype=float).ravel()
        target = np.ravel(target)
        idx = np.arange(x.size)
        for _ in range(_NEWTON_MAX):
            xi = x[idx]
            y, dens = fn(xi, idx)
            res = y - target[idx]
            lo[idx] = np.where(res < 0, xi, lo[idx])
            hi[idx] = np.where(res > 0, xi, hi[idx])
            done = (np.abs(res) < _NEWTON_TOL) | (
                hi[idx] - lo[idx] <= 4e-16 * np.abs(xi)
            )
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                new = xi - res / (dens * np.exp(-_log_std_normal(y)))
            bad = ~np.isfinite(new) | (new <= lo[idx]) | (new >= hi[idx])
            new = np.where(bad, 0.5 * (lo[idx] + hi[idx]), new)
            x[idx] = np.where(done, xi, new)
            idx = idx[~done]
            if idx.size == 0:
                break
        return x.reshape(np.shape(x0))
