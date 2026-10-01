"""Tabulated effective-spin coordinates: exactly invertible, approximately normal.

:class:`~nessai_gw._effective_spin.EffectiveSpinTransform` evaluates the
Rosenblatt transform of the ``AlignedSpin`` prior with quadratures (forward)
and a Newton iteration on them (inverse), so that the prior on ``(u, w)`` is
exactly ``N(0, I)``.  That exactness is not needed: nessai only needs a
bijection with an exact Jacobian; how close the prior is to ``N(0, I)`` only
affects how easy the flow's job is.  Here the two CDFs are tabulated once,
from the same quadratures, and the map is defined *by* the tables:

    u = Phi^{-1}( G(t | q) ),            t = S / s_max(q),  S = chi_1 + q chi_2,
    w = Phi^{-1}( F(r | t, q) ),         r = (chi_1 - lo) / (hi - lo),

``[lo, hi]`` the ``chi_1`` range of the constant-``S`` segment inside the
prior box.  ``G`` and ``F`` are piecewise linear in ``t`` and ``r`` with knots
shared by every table node, and linear interpolation between the nodes in
``q`` (and ``t`` for ``F``) is a convex combination of such functions, so at
fixed ``q`` each coordinate is a strictly increasing piecewise-linear
function: the inverse locates the cell and solves a linear equation, and the
round trip holds to rounding.  The Jacobian is that of the implemented map,

    |d(u, w) / d(chi_1, chi_2)| = q (dG/dS) (dF/dchi_1 at fixed S)
                                  / (phi(u) phi(w)),

with the slopes of the cells.  Tails keep full relative precision: ``G`` is
tabulated for ``S <= 0`` only (the prior is symmetric under ``chi -> -chi``,
so ``G(t) = 1 - G(-t)``), and ``F`` is stored together with its complement
``C = 1 - F`` (each from its own side's integral), ``w`` taken from whichever
is below 1/2.  The tables are built once per ``(a_max_1, a_max_2)`` and
process (~seconds) and are not pickled.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import ndtr, ndtri

from ._effective_spin import EffectiveSpinTransform, _log_std_normal


def _clustered(n_end, smallest=1e-10, n_uniform=0):
    """Knots on [0, 1], geometrically clustered at both ends."""
    d = np.logspace(np.log10(smallest), np.log10(0.5), n_end)
    knots = np.concatenate([[0.0, 1.0], d, 1.0 - d, np.linspace(0, 1, n_uniform)])
    return np.unique(knots)


def _half_t_knots(n_end, smallest, n_zero=40, smallest_zero=1e-4):
    """Knots on [-1, 0] (``S <= 0``): geometrically clustered at the prior
    edge ``t = -1``, where the CDF is tiny and needs relative precision, and
    more mildly at the cusp ``t = 0``, where it is ~1/2 and successive values
    must stay further apart than the quadrature error."""
    d_edge = np.logspace(np.log10(smallest), np.log10(0.5), n_end)
    d_zero = np.logspace(np.log10(smallest_zero), np.log10(0.5), n_zero)
    return np.unique(np.concatenate([[-1.0, 0.0], d_edge - 1.0, -d_zero]))


def _strictly_increasing(arr):
    """Make ``arr`` strictly increasing along its last axis, raising each
    value that does not exceed its predecessor by a relative ulp."""
    arr = np.array(arr, dtype=float)
    for k in range(1, arr.shape[-1]):
        prev = arr[..., k - 1]
        floor = prev + np.maximum(np.abs(prev) * 4e-16, 1e-300)
        arr[..., k] = np.maximum(arr[..., k], floor)
    return arr


@lru_cache(maxsize=4)
def _build_tables(a1, a2, q_min, n_q, n_t_u, n_t_w, n_r):
    exact = EffectiveSpinTransform(a1, a2)
    q_nodes = np.geomspace(q_min, 1.0, n_q)
    # -- G(t | q) for t in [-1, 0] -----------------------------------------
    t_u = _half_t_knots(n_t_u, 1e-7)
    qq, tt = np.meshgrid(q_nodes, t_u, indexing="ij")
    s = tt * (a1 + qq * a2)
    exact._setup(s, qq)
    g = exact._lower_cdf_s()
    g[:, 0] = 0.0
    g[:, -1] = 0.5
    # -- F(r | t, q) and its complement, for t in [-1, 0] -------------------
    t_w = _half_t_knots(n_t_w, 1e-7)[1:]  # the segment vanishes at t = -1
    r = _clustered(n_r, 1e-10, n_uniform=41)
    f = np.empty((n_q, len(t_w), len(r)))
    c = np.empty_like(f)
    for i, q in enumerate(q_nodes):
        s = t_w * (a1 + q * a2)
        qv = np.full_like(s, q)
        exact._setup(s, qv)
        lo, hi = exact._segment()
        knots = lo[:, None] + r[None, :] * (hi - lo)[:, None]
        a, b = knots[:, :-1], knots[:, 1:]
        exact._setup(np.broadcast_to(s[:, None], a.shape).copy(),
                     np.broadcast_to(qv[:, None], a.shape).copy())
        zero = np.zeros_like(a)
        pieces = exact._integrate(exact._joint, a, b, [zero, exact._s])
        lower = np.concatenate(
            [np.zeros((len(s), 1)), np.cumsum(pieces, axis=1)], axis=1
        )
        upper = np.concatenate(
            [np.cumsum(pieces[:, ::-1], axis=1)[:, ::-1], np.zeros((len(s), 1))],
            axis=1,
        )
        z = lower[:, -1:]
        f[i] = lower / z
        c[i] = upper / z
    # Near the corners of the prior box the density underflows to zero (the
    # cdf flattens below rounding); the map must stay strictly monotonic, so
    # lift ties by a relative ulp (negligible prior mass is involved).
    g = _strictly_increasing(g)
    f = _strictly_increasing(f)
    c = _strictly_increasing(c[..., ::-1])[..., ::-1]
    for name, arr in (("G", g), ("F", f), ("C", -c)):
        if not np.all(np.diff(arr, axis=-1) > 0):
            raise RuntimeError(f"tabulated {name} is not strictly monotonic")
    return dict(q=q_nodes, t_u=t_u, g=g, t_w=t_w, r=r, f=f, c=c)


def _bracket(nodes, x):
    """Cell index ``i`` and weight ``lam`` with ``x ~ nodes[i] .. nodes[i+1]``
    (clamped to the table)."""
    x = np.clip(x, nodes[0], nodes[-1])
    i = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2)
    lam = (x - nodes[i]) / (nodes[i + 1] - nodes[i])
    return i, lam


def _locate(rows, values):
    """Cell of each row (increasing along the last axis) containing
    ``values``: index ``k`` with ``rows[k] <= value <= rows[k + 1]``."""
    k = np.sum(rows <= values[:, None], axis=1) - 1
    return np.clip(k, 0, rows.shape[1] - 2)


class TabulatedEffectiveSpinTransform:
    """Drop-in for :class:`~nessai_gw._effective_spin.EffectiveSpinTransform`
    with tabulated CDFs (see the module docstring).

    Parameters
    ----------
    a_max_1, a_max_2 : float
        Maximum spin magnitudes of the two ``AlignedSpin`` priors.
    q_min : float, optional
        Smallest mass ratio of the tables; below it the ``q_min`` tables are
        used, which is still exactly invertible, only less normal.
    """

    def __init__(self, a_max_1, a_max_2, q_min=0.05, n_q=33, n_t_u=160,
                 n_t_w=40, n_r=80):
        self.a1 = float(a_max_1)
        self.a2 = float(a_max_2)
        if self.a1 <= 0 or self.a2 <= 0:
            raise ValueError("a_max must be positive")
        self._config = (self.a1, self.a2, float(q_min), int(n_q), int(n_t_u),
                        int(n_t_w), int(n_r))

    def __getstate__(self):
        return dict(a1=self.a1, a2=self.a2, _config=self._config)

    @property
    def _tables(self):
        return _build_tables(*self._config)

    # -- G -------------------------------------------------------------------
    def _g_rows(self, q):
        tab = self._tables
        i, lam = _bracket(tab["q"], q)
        return (1 - lam)[:, None] * tab["g"][i] + lam[:, None] * tab["g"][i + 1]

    def _u_and_slope(self, t, q):
        """``u`` and ``log dG/dt`` at ``t`` in [-1, 1]."""
        tab = self._tables
        rows = self._g_rows(q)
        tn = -np.abs(t)
        k, lam = _bracket(tab["t_u"], tn)
        n = np.arange(len(t))
        g0, g1 = rows[n, k], rows[n, k + 1]
        g = g0 + lam * (g1 - g0)
        slope = (g1 - g0) / (tab["t_u"][k + 1] - tab["t_u"][k])
        u = ndtri(g)
        return np.where(t > 0, -u, u), np.log(slope)

    def _t_of_u(self, u, q):
        tab = self._tables
        rows = self._g_rows(q)
        target = ndtr(-np.abs(u))
        k = _locate(rows, target)
        n = np.arange(len(u))
        g0, g1 = rows[n, k], rows[n, k + 1]
        t0, t1 = tab["t_u"][k], tab["t_u"][k + 1]
        tn = t0 + (target - g0) / (g1 - g0) * (t1 - t0)
        return np.where(u > 0, -tn, tn)

    # -- F -------------------------------------------------------------------
    def _f_rows(self, tn, q):
        """Interpolated ``F`` and ``C`` rows at ``tn <= 0``."""
        tab = self._tables
        i, a = _bracket(tab["q"], q)
        j, b = _bracket(tab["t_w"], tn)
        out = []
        for arr in (tab["f"], tab["c"]):
            out.append(
                ((1 - a) * (1 - b))[:, None] * arr[i, j]
                + ((1 - a) * b)[:, None] * arr[i, j + 1]
                + (a * (1 - b))[:, None] * arr[i + 1, j]
                + (a * b)[:, None] * arr[i + 1, j + 1]
            )
        return out

    def _segment(self, s, q):
        return (np.maximum(-self.a1, s - q * self.a2),
                np.minimum(self.a1, s + q * self.a2))

    def _w_and_slope(self, r, t, q):
        """``w`` and ``log dF/dr`` at ``r`` for the segment at ``t``."""
        tab = self._tables
        flip = t > 0
        rn = np.where(flip, 1.0 - r, r)
        f_rows, c_rows = self._f_rows(-np.abs(t), q)
        k, lam = _bracket(tab["r"], rn)
        n = np.arange(len(r))
        f0, f1 = f_rows[n, k], f_rows[n, k + 1]
        c0, c1 = c_rows[n, k], c_rows[n, k + 1]
        f = f0 + lam * (f1 - f0)
        cc = c0 + lam * (c1 - c0)
        slope = (f1 - f0) / (tab["r"][k + 1] - tab["r"][k])
        w = np.where(f <= cc, ndtri(f), -ndtri(cc))
        return np.where(flip, -w, w), np.log(slope)

    def _r_of_w(self, w, t, q):
        tab = self._tables
        flip = t > 0
        wn = np.where(flip, -w, w)
        f_rows, c_rows = self._f_rows(-np.abs(t), q)
        n = np.arange(len(w))
        low = wn <= 0
        # w <= 0: solve F(r) = Phi(w); else C(r) = Phi(-w) (C decreasing)
        target = ndtr(-np.abs(wn))
        rows = np.where(low[:, None], f_rows, -c_rows)
        k = _locate(rows, np.where(low, target, -target))
        y0, y1 = rows[n, k], rows[n, k + 1]
        yt = np.where(low, target, -target)
        r0, r1 = tab["r"][k], tab["r"][k + 1]
        rn = r0 + (yt - y0) / (y1 - y0) * (r1 - r0)
        return np.where(flip, 1.0 - rn, rn)

    # -- public interface ----------------------------------------------------
    def _log_jacobian(self, q, s_max, log_g_t, log_f_r, width, u, w):
        return (np.log(q) + log_g_t - np.log(s_max) + log_f_r - np.log(width)
                - _log_std_normal(u) - _log_std_normal(w))

    def forward(self, chi1, chi2, q):
        """``(u, w, log|d(u, w)/d(chi_1, chi_2)|)``."""
        chi1, chi2, q = np.broadcast_arrays(
            *(np.atleast_1d(np.asarray(a, dtype=float)) for a in (chi1, chi2, q))
        )
        s_max = self.a1 + q * self.a2
        s = chi1 + q * chi2
        t = s / s_max
        u, log_g = self._u_and_slope(t, q)
        lo, hi = self._segment(s, q)
        width = hi - lo
        r = (chi1 - lo) / width
        w, log_f = self._w_and_slope(r, t, q)
        return u, w, self._log_jacobian(q, s_max, log_g, log_f, width, u, w)

    def inverse(self, u, w, q):
        """``(chi_1, chi_2, log|d(u, w)/d(chi_1, chi_2)|)`` (the *forward*
        log-Jacobian at the returned point)."""
        u, w, q = np.broadcast_arrays(
            *(np.atleast_1d(np.asarray(a, dtype=float)) for a in (u, w, q))
        )
        s_max = self.a1 + q * self.a2
        t = self._t_of_u(u, q)
        s = t * s_max
        lo, hi = self._segment(s, q)
        width = hi - lo
        r = self._r_of_w(w, t, q)
        chi1 = lo + r * width
        chi2 = (s - chi1) / q
        _, log_g = self._u_and_slope(t, q)
        _, log_f = self._w_and_slope(r, t, q)
        return chi1, chi2, self._log_jacobian(
            q, s_max, log_g, log_f, width, u, w
        )
