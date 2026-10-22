"""The sheet chart: a proposal in coordinates along the continuous sky
degeneracy of a triangular detector.

Why
---
In the frozen long-wavelength limit a triangle's dominant-mode response is
``h_k = [alpha Z_k(n) + beta conj(Z_k(n))] A(f)`` (see
:mod:`nessai_gw._ellipse`), with

    alpha = c_+ e^{-2i psi} e^{2i s phase} / d_L,   c_+ = (1 + cos theta_jn)^2 / 4,
    beta  = c_- e^{+2i psi} e^{2i s phase} / d_L,   c_- = (1 - cos theta_jn)^2 / 4,

(``s = -1`` in bilby's convention). Every sky position ``n`` fits the data,
with ``(alpha, beta) = (alpha*, beta*)(n)`` up to a common complex factor:
the likelihood lives on a two-dimensional sheet over the sky in the
six-dimensional extrinsic space, broken only by the vertex separation, the
Earth's rotation and the higher modes. The discrete group folded by
:mod:`nessai_gw.group_mixture` is a subgroup of that degeneracy. Where the
posterior spreads along the sheet (the edge-on secondary mode of a loud
ET-Delta signal), the inclination, distance, polarisation and phase are each
broad but tied to the sky to ~1%: the live points sit on a sheet that is
thin in every plain coordinate system, and a flow trained on them leaves
most of its draws hundreds of nats below the contour.

The chart
---------
The flow coordinates are the sky, the arrival time at the detector vertex,
and the four residuals about the sheet,

    ln|alpha / alpha*(n)|, ln|beta / beta*(n)|,
    arg(alpha / alpha*(n)), arg(beta / beta*(n)),

(centred), with ``(alpha*, beta*)(n)`` the coefficients that reproduce, at
sky position ``n``, the response pattern of a fiducial point (the region's
highest-likelihood live point). The residuals are 25-100 times narrower than
``cos theta_jn``, ``ln d_L``, ``psi`` and ``phase`` on the ET-Delta v70 edge-on
mode, and a step along the sky at fixed residuals stays on the contour.

``(alpha, beta)`` fixes ``(psi, phase)`` only up to the four-element lattice
``G = {(0, 0), (pi/2, pi/2), (0, pi), (pi/2, 3 pi/2)}`` (mod ``(pi, 2 pi)``):
the four branches have the same dominant-mode signal and differ through the
higher modes. The branch is a discrete label ``k``, relative to the
``(psi, phase)`` the sheet predicts at ``n``, drawn with the branch shares of
the training points (plus one each) and not seen by the flow. The density of
a point is

    q(x) = p_k q_flow(y(x)) |dy/dx|,
    |d(ln|alpha|, ln|beta|, arg alpha, arg beta) / d(theta_jn, d_L, psi, phase)|
        = 32 / (d_L sin theta_jn).

All coordinates are whitened together (Cholesky factor of the training
points' covariance), refitted with the fiducial at every training.

The chart is singular where ``alpha*`` or ``beta*`` vanishes, at the eight
face-on sky points: it suits a region of the posterior away from face-on, and
:class:`SheetChartRegions` gives it to a partitioned run's regions whose live
points are mostly edge-on, leaving the others their copy of the group
proposal.

Two separated interferometers
-----------------------------
Two interferometers (ET 2L) measure two complex amplitudes ``A_k = alpha
Z_k(n) + beta conj(Z_k(n))`` and an arrival-time difference: five real
constraints on the six extrinsic parameters other than the time, so the
degenerate set is a curve, the ring about the baseline that the delay
fixes, with ``(alpha, beta) = M(n)^-1 (A_1, A_2)`` along it.
:class:`RingChart` samples the azimuth about the baseline, the cosine of
the angle to it, the arrival time at the first interferometer and the
amplitudes relative to the fiducial point's, with the same branch label.
On the live points of the stalled ET 2L v70 run (ln X -46) the amplitude
residuals are 0.02-0.03 wide while psi and phase cover their circles; the
chart is valid over the whole sky but where ``det M`` vanishes, so
:class:`RingChartFlowProposal` is a whole run's proposal.
"""

from __future__ import annotations

import copy
import logging

import numpy as np

from nessai import config as nessai_config
from nessai.livepoint import empty_structured_array, live_points_to_array
from nessai.proposal import FlowProposal

from . import nessai_logger
from ._ellipse import PolarisationEllipse, _complex_response
from ._geometry import geocenter_time_delay, greenwich_mean_sidereal_time

# a child of nessai's logger, so the messages reach the run's log
logger = nessai_logger.getChild(__name__)

_TWO_PI = 2.0 * np.pi
#: The physical parameters the chart replaces.
EXTRINSIC = (
    "ra",
    "dec",
    "geocent_time",
    "theta_jn",
    "luminosity_distance",
    "psi",
    "phase",
)
#: The (psi, phase) shifts that leave (alpha, beta) unchanged.
LATTICE = np.array(
    [(0.0, 0.0), (0.5 * np.pi, 0.5 * np.pi), (0.0, np.pi), (0.5 * np.pi, 1.5 * np.pi)]
)
#: Flow coordinates of the extrinsic block.
SHEET_COLUMNS = (
    "sheet_ra",
    "sheet_sin_dec",
    "sheet_time",
    "sheet_ln_alpha",
    "sheet_ln_beta",
    "sheet_arg_alpha",
    "sheet_arg_beta",
)
#: Intrinsic parameters sampled in their logarithm.
LOG_PARAMETERS = ("mass_ratio",)


def _wrap(x, period=_TWO_PI):
    """``x`` in ``[-period / 2, period / 2)``."""
    return np.remainder(np.asarray(x, dtype=float) + 0.5 * period, period) - 0.5 * period


def _circular_mean(angles):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(angles, dtype=float)))))


class _Chart:
    """What the charts share: the intrinsic columns (``mass_ratio`` as ``ln
    q``), the polarisation amplitudes ``(alpha, beta)``, the ``(psi,
    phase)`` branch label, the automatic phase sign, the whitening and the
    branch shares. A chart defines :meth:`fit` (which ends with
    :meth:`_finish_fit`), :meth:`columns_of`, :meth:`_inverse` and its
    column names."""

    #: a refit changes the automatic phase sign only if the other sign's
    #: residual phases spread less than this fraction of the current one's
    STICKY_SIGN = 0.5
    #: prefix of the column names
    prefix = "chart_"

    def _set_names(self, names, phase_sign, extrinsic_columns):
        self.names = list(names)
        missing = sorted(set(EXTRINSIC) - set(self.names))
        if missing:
            raise ValueError(f"the chart needs {missing} in the model parameters")
        if phase_sign not in (None, -1, 1):
            raise ValueError("phase_sign must be None, -1 or 1")
        self.fixed_phase_sign = phase_sign
        self.phase_sign = -1 if phase_sign is None else phase_sign
        self.intrinsic = [n for n in self.names if n not in EXTRINSIC]
        self.columns = [
            f"{self.prefix}ln_{n}" if n in LOG_PARAMETERS else f"{self.prefix}{n}"
            for n in self.intrinsic
        ] + list(extrinsic_columns)
        self.fitted = False
        self.fiducial = None

    @property
    def dims(self):
        return len(self.columns)

    @staticmethod
    def _alpha_beta(theta_jn, distance, psi, phase, sign):
        c = np.cos(theta_jn)
        alpha = (1 + c) ** 2 / 4 / distance * np.exp(1j * (-2 * psi + 2 * sign * phase))
        beta = (1 - c) ** 2 / 4 / distance * np.exp(1j * (2 * psi + 2 * sign * phase))
        return alpha, beta

    @staticmethod
    def _label(dpsi, dphase):
        """Index of the lattice element ``(dpsi, dphase)`` is (mod
        ``(pi, 2 pi)``)."""
        err = np.abs(_wrap(dpsi[:, None] - LATTICE[None, :, 0], np.pi)) + np.abs(
            _wrap(dphase[:, None] - LATTICE[None, :, 1], _TWO_PI)
        )
        return np.argmin(err, axis=1)

    def _fiducial_index(self, x):
        if len(x) < self.dims + 2:
            raise ValueError(f"cannot fit the chart to {len(x)} points")
        if "logL" in (x.dtype.names or ()) and np.isfinite(x["logL"]).any():
            return int(np.nanargmax(np.where(np.isfinite(x["logL"]), x["logL"], -np.inf)))
        return 0

    def _choose_sign(self, spread):
        """The phase sign from the residual-phase spread of each candidate:
        the narrower, but the sign of the previous fit is kept unless the
        other is clearly better (on broad live sets the two are close, and a
        flip would change the flow's coordinates between trainings)."""
        best = min(spread, key=spread.get)
        if not (
            self.fitted
            and len(spread) == 2
            and spread[best] > self.STICKY_SIGN * spread[self.phase_sign]
        ):
            self.phase_sign = best
        self.spread = spread[self.phase_sign]

    def _finish_fit(self, x):
        """The branch shares and the whitening, from the chart's columns of
        ``x`` (the rest of the fit done)."""
        self.mean = np.zeros(self.dims)
        self.chol = np.eye(self.dims)
        self.log_p_k = np.zeros(len(LATTICE))
        self.log_det_whiten = 0.0
        self.fitted = True
        c, k, _ = self.columns_of(x)
        counts = np.bincount(k, minlength=len(LATTICE))
        self.p_k = (counts + 1.0) / (len(k) + len(LATTICE))
        self.log_p_k = np.log(self.p_k)
        self.mean = c.mean(axis=0)
        cov = np.cov(c, rowvar=False)
        cov += np.diag(1e-12 * np.diag(cov) + 1e-300)
        self.chol = np.linalg.cholesky(cov)
        self.log_det_whiten = -float(np.sum(np.log(np.diag(self.chol))))
        return self

    def _intrinsic_columns(self, x):
        cols, log_j = [], np.zeros(len(x))
        for n in self.intrinsic:
            if n in LOG_PARAMETERS:
                cols.append(np.log(x[n]))
                log_j -= np.log(x[n])
            else:
                cols.append(np.asarray(x[n], dtype=float))
        return cols, log_j

    def _intrinsic_inverse(self, c):
        out, log_j = {}, np.zeros(len(c))
        for i, n in enumerate(self.intrinsic):
            if n in LOG_PARAMETERS:
                out[n] = np.exp(c[:, i])
                log_j -= c[:, i]
            else:
                out[n] = c[:, i]
        return out, log_j

    def forward(self, x):
        """Physical -> whitened chart ``y``, branch ``k`` and ``ln q`` terms
        (``ln |dy/dx| + ln p_k``)."""
        if not self.fitted:
            raise RuntimeError("the chart is used before it is fitted")
        c, k, log_j = self.columns_of(np.atleast_1d(x))
        y = np.linalg.solve(self.chol, (c - self.mean).T).T
        return y, k, log_j + self.log_det_whiten + self.log_p_k[k]

    def inverse(self, y, k):
        """Whitened chart ``y`` and branch ``k`` -> physical parameters (a
        dict of arrays), the mask of the rows inside the chart's domain and
        the forward ``ln q`` terms at them."""
        if not self.fitted:
            raise RuntimeError("the chart is used before it is fitted")
        # draws far out in the flow's tails overflow: they are flagged
        # invalid, without a warning each
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            y = np.atleast_2d(np.asarray(y, dtype=float))
            c = y @ self.chol.T + self.mean
            out, valid, log_j = self._inverse(c, np.asarray(k, dtype=int))
            log_j = log_j + self.log_det_whiten + self.log_p_k[k]
            valid &= np.all(np.isfinite(c), axis=1) & np.isfinite(log_j)
            for n in self.names:
                valid &= np.isfinite(out[n])
        return out, valid, np.where(valid, log_j, 0.0)

    def _column_sd(self):
        sd = np.sqrt(np.sum(self.chol**2, axis=1))
        return ", ".join(
            f"{n[len(self.prefix):]} {v:.3g}" for n, v in zip(self.columns, sd)
        )


class SheetChart(_Chart):
    """The map between the physical parameters and the sheet chart.

    Parameters
    ----------
    plane_normal : array_like
        Unit normal to the detector plane (Earth-fixed), for the ideal
        triangle the sheet is computed on
        (:func:`nessai_gw.group_mixture.detector_plane_normal`).
    vertex : array_like
        Earth-fixed position (m) the arrival time is taken at (one of the
        triangle's vertices).
    reference_time : float
        GPS time whose Earth orientation the sheet and the delay are taken
        at (the frozen limit).
    names : list of str
        The model's parameters: :data:`EXTRINSIC` and any others, which are
        carried as they are (``mass_ratio`` as ``ln q``).
    phase_sign : {None, -1, 1}
        ``s`` in ``e^{2i s phase}``; ``None`` picks the sign that makes the
        phase residuals narrower (bilby's waveforms: -1), at the first fit
        and at a refit where the other sign is clearly better
        (:attr:`STICKY_SIGN`).
    """

    prefix = "sheet_"

    def __init__(self, plane_normal, vertex, reference_time, names, phase_sign=None):
        self.plane_normal = np.asarray(plane_normal, dtype=float)
        self.vertex = np.asarray(vertex, dtype=float)
        self.reference_time = float(reference_time)
        self.gmst = greenwich_mean_sidereal_time(self.reference_time)
        self._set_names(names, phase_sign, SHEET_COLUMNS)
        self.ellipse = None

    # -- the sheet ------------------------------------------------------
    def _sheet(self, ra, dec):
        """``(alpha*, beta*)(n)`` of the fiducial pattern."""
        return self.ellipse.coefficients(np.asarray(ra, float), np.asarray(dec, float))

    def _centre(self, a0, b0):
        """The ``(psi, phase)`` the sheet predicts (one representative of
        its lattice class), from ``(alpha*, beta*)`` and the fitted offsets.
        The branch of ``arg`` is taken relative to the fiducial's, so its
        cut lies half a turn away from the fiducial on the sheet."""
        A = self._arg_a_fid + _wrap(np.angle(a0) - self._arg_a0_fid)
        B = self._arg_b_fid + _wrap(np.angle(b0) - self._arg_b0_fid)
        return 0.25 * (B - A), 0.25 * self.phase_sign * (A + B)

    def _offsets(self, r_a, r_b):
        """The ``(psi, phase)`` offset of residual phases ``(r_a, r_b)``."""
        return 0.25 * (r_b - r_a), 0.25 * self.phase_sign * (r_a + r_b)

    # -- fit --------------------------------------------------------------
    def fit(self, x):
        """Fit the fiducial, the offsets, the branch shares and the whitening
        to the points ``x`` (a structured array with the model parameters,
        and ``logL`` for the fiducial)."""
        x = np.atleast_1d(x)
        i = self._fiducial_index(x)
        self.fiducial = {n: float(x[n][i]) for n in ("ra", "dec", "psi", "theta_jn")}
        self.ellipse = PolarisationEllipse(self.plane_normal, self.reference_time, self.fiducial)
        a0, b0 = self._sheet(x["ra"], x["dec"])
        signs = (self.fixed_phase_sign,) if self.fixed_phase_sign else (-1, 1)
        spread = {}
        for sign in signs:
            a, b = self._alpha_beta(
                x["theta_jn"], x["luminosity_distance"], x["psi"], x["phase"], sign
            )
            ra_, rb_ = a / a0, b / b0
            # circular spread of the residual phases (1 - mean resultant length)
            spread[sign] = 2 - abs(np.mean(ra_ / np.abs(ra_))) - abs(np.mean(rb_ / np.abs(rb_)))
        self._choose_sign(spread)
        a, b = self._alpha_beta(
            x["theta_jn"], x["luminosity_distance"], x["psi"], x["phase"], self.phase_sign
        )
        self.ra_centre = _circular_mean(x["ra"])
        self.arg_a_offset = _circular_mean(np.angle(a / a0))
        self.arg_b_offset = _circular_mean(np.angle(b / b0))
        a0f, b0f = self._sheet(self.fiducial["ra"], self.fiducial["dec"])
        self._arg_a0_fid = float(np.angle(a0f))
        self._arg_b0_fid = float(np.angle(b0f))
        self._arg_a_fid = self._arg_a0_fid + self.arg_a_offset
        self._arg_b_fid = self._arg_b0_fid + self.arg_b_offset
        return self._finish_fit(x)

    # -- maps -----------------------------------------------------------
    def columns_of(self, x):
        """Physical -> unwhitened chart columns ``c``, branch ``k`` and
        ``ln |dc/dx|``."""
        x = np.atleast_1d(x)
        cols, log_j = self._intrinsic_columns(x)
        ra, dec = np.asarray(x["ra"], float), np.asarray(x["dec"], float)
        theta, dist = np.asarray(x["theta_jn"], float), np.asarray(x["luminosity_distance"], float)
        psi, phase = np.asarray(x["psi"], float), np.asarray(x["phase"], float)
        delay = geocenter_time_delay(self.vertex, self.gmst, ra, dec)
        cols += [
            _wrap(ra - self.ra_centre),
            np.sin(dec),
            (np.asarray(x["geocent_time"], float) - self.reference_time) + delay,
        ]
        log_j += np.log(np.cos(dec))
        a0, b0 = self._sheet(ra, dec)
        a, b = self._alpha_beta(theta, dist, psi, phase, self.phase_sign)
        ra_, rb_ = a / a0, b / b0
        r_a = _wrap(np.angle(ra_) - self.arg_a_offset)
        r_b = _wrap(np.angle(rb_) - self.arg_b_offset)
        cols += [np.log(np.abs(ra_)), np.log(np.abs(rb_)), r_a, r_b]
        log_j += np.log(32.0) - np.log(dist) - np.log(np.sin(theta))
        psi_c, phase_c = self._centre(a0, b0)
        dpsi, dphase = self._offsets(r_a, r_b)
        k = self._label(psi - psi_c - dpsi, phase - phase_c - dphase)
        return np.stack(cols, axis=1), k, log_j

    def _inverse(self, c, k):
        """Unwhitened columns ``c`` and branch ``k`` -> the physical
        parameters, the rows in the chart's domain and ``ln |dc/dx|``."""
        n_int = len(self.intrinsic)
        out, log_j = self._intrinsic_inverse(c)
        u, s, t, la, lb, r_a, r_b = (c[:, n_int + j] for j in range(7))
        valid = (
            (np.abs(s) < 1)
            & (u >= -np.pi) & (u < np.pi)
            & (r_a >= -np.pi) & (r_a < np.pi)
            & (r_b >= -np.pi) & (r_b < np.pi)
            & np.all(np.isfinite(c), axis=1)
        )
        s = np.where(valid, s, 0.0)
        ra = np.remainder(u + self.ra_centre, _TWO_PI)
        dec = np.arcsin(s)
        delay = geocenter_time_delay(self.vertex, self.gmst, ra, dec)
        a0, b0 = self._sheet(ra, dec)
        abs_a = np.exp(np.where(valid, la, 0.0)) * np.abs(a0)
        abs_b = np.exp(np.where(valid, lb, 0.0)) * np.abs(b0)
        sa, sb = np.sqrt(abs_a), np.sqrt(abs_b)
        cos_theta = (sa - sb) / (sa + sb)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        dist = (sa + sb) ** -2
        psi_c, phase_c = self._centre(a0, b0)
        dpsi, dphase = self._offsets(r_a, r_b)
        psi = np.remainder(psi_c + dpsi + LATTICE[k, 0], np.pi)
        phase = np.remainder(phase_c + dphase + LATTICE[k, 1], _TWO_PI)
        valid &= (sa > 0) & (sb > 0) & np.isfinite(dist) & (np.sin(theta) > 0)
        out.update(
            ra=ra,
            dec=dec,
            geocent_time=self.reference_time + t - delay,
            theta_jn=theta,
            luminosity_distance=dist,
            psi=psi,
            phase=phase,
        )
        log_j += (
            np.log(np.cos(dec))
            + np.log(32.0)
            - np.log(dist)
            - np.log(np.sin(theta))
        )
        return out, valid, log_j

    def describe(self):
        if not self.fitted:
            return "unfitted"
        return (
            f"fiducial ra {self.fiducial['ra']:.3f} dec {self.fiducial['dec']:.3f} "
            f"theta_jn {self.fiducial['theta_jn']:.3f} psi {self.fiducial['psi']:.3f}; "
            f"phase sign {self.phase_sign:+d}; branch shares "
            f"{np.round(self.p_k, 3).tolist()}; sd " + self._column_sd()
        )


#: Flow coordinates of the extrinsic block of the ring chart.
RING_COLUMNS = (
    "ring_azimuth",
    "ring_cos_alpha",
    "ring_time",
    "ring_ln_a1",
    "ring_ln_a2",
    "ring_arg_a1",
    "ring_arg_a2",
)


class RingChart(_Chart):
    """The ring chart of a network of two separated interferometers.

    Each interferometer measures one complex amplitude of the dominant mode,
    ``A_k = alpha Z_k(n) + beta conj(Z_k(n))`` (``Z_k = F_+ + i F_x`` at
    ``psi = 0``), and the two an arrival-time difference that holds the sky
    to a ring about their baseline. With the two amplitudes, the arrival
    time and the delay fixed, the extrinsic parameters keep one degree of
    freedom: the azimuth about the baseline, along which ``(theta_jn, d_L,
    psi, phase)`` follow ``(alpha, beta) = M(n)^-1 (A_1, A_2)``. The chart's
    coordinates are that azimuth, the cosine of the angle to the baseline,
    the arrival time at the first interferometer and the amplitudes' log
    moduli and phases relative to the fiducial point's (centred), with the
    ``(psi, phase)`` branch a label as in :class:`SheetChart`.

    The Jacobian of the amplitude block is
    ``|det M|^2 |alpha|^2 |beta|^2 32 / (d_L sin theta_jn) / (|A_1|^2
    |A_2|^2)``, of the sky ``cos dec``. The chart is singular where ``det M
    = Z_1 conj(Z_2) - conj(Z_1) Z_2`` vanishes (the sky directions where the
    two interferometers see the same polarisation): a set of measure zero,
    and away from the ring for ET 2L.

    Parameters
    ----------
    tensors : array_like, shape (2, 3, 3)
        Earth-fixed detector tensors of the two interferometers
        (:func:`nessai_gw.group_mixture.detector_tensors`).
    vertices : array_like, shape (2, 3)
        Their Earth-fixed vertices (m); the arrival time is taken at the
        first.
    reference_time, names, phase_sign
        As for :class:`SheetChart`.
    """

    prefix = "ring_"

    def __init__(self, tensors, vertices, reference_time, names, phase_sign=None):
        self.tensors = np.asarray(tensors, dtype=float)
        self.vertices = np.asarray(vertices, dtype=float)
        if self.tensors.shape != (2, 3, 3) or self.vertices.shape != (2, 3):
            raise ValueError("the ring chart needs two interferometers")
        self.reference_time = float(reference_time)
        self.gmst = greenwich_mean_sidereal_time(self.reference_time)
        baseline = self.vertices[1] - self.vertices[0]
        if not np.linalg.norm(baseline) > 0:
            raise ValueError("the two interferometers share a vertex: no baseline")
        self.axis = baseline / np.linalg.norm(baseline)
        pole = np.array([0.0, 0.0, 1.0]) if abs(self.axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(self.axis, pole)
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.axis, self.e1)
        self._set_names(names, phase_sign, RING_COLUMNS)

    # -- the ring ---------------------------------------------------------
    def _ring(self, ra, dec):
        """Azimuth about the baseline and cosine of the angle to it."""
        phi = np.asarray(ra, float) - self.gmst
        dec = np.asarray(dec, float)
        n = np.stack([np.cos(dec) * np.cos(phi), np.cos(dec) * np.sin(phi), np.sin(dec)], -1)
        return np.arctan2(n @ self.e2, n @ self.e1), n @ self.axis

    def _sky(self, azimuth, cos_alpha):
        sin_alpha = np.sqrt(np.clip(1.0 - cos_alpha**2, 0.0, None))
        n = (
            cos_alpha[:, None] * self.axis
            + (sin_alpha * np.cos(azimuth))[:, None] * self.e1
            + (sin_alpha * np.sin(azimuth))[:, None] * self.e2
        )
        ra = np.remainder(np.arctan2(n[:, 1], n[:, 0]) + self.gmst, _TWO_PI)
        return ra, np.arcsin(np.clip(n[:, 2], -1.0, 1.0))

    def _response(self, ra, dec):
        return _complex_response(
            self.tensors, np.atleast_1d(ra), np.atleast_1d(dec), self.gmst
        )

    @staticmethod
    def _amplitudes(z, a, b):
        return a[:, None] * z + b[:, None] * np.conj(z)

    @staticmethod
    def _det(z):
        return z[:, 0] * np.conj(z[:, 1]) - np.conj(z[:, 0]) * z[:, 1]

    def _centre(self, a, b):
        """The ``(psi, phase)`` of ``(alpha, beta)`` (one representative of
        the lattice class), with the branch of ``arg`` taken relative to the
        fiducial's."""
        A = self._arg_a_fid + _wrap(np.angle(a) - self._arg_a_fid)
        B = self._arg_b_fid + _wrap(np.angle(b) - self._arg_b_fid)
        return 0.25 * (B - A), 0.25 * self.phase_sign * (A + B)

    def _amplitude_log_j(self, z, a, b, amps, distance, theta):
        return (
            2 * np.log(np.abs(self._det(z)))
            + 2 * np.log(np.abs(a))
            + 2 * np.log(np.abs(b))
            + np.log(32.0)
            - np.log(distance)
            - np.log(np.sin(theta))
            - 2 * np.log(np.abs(amps[:, 0]))
            - 2 * np.log(np.abs(amps[:, 1]))
        )

    # -- fit --------------------------------------------------------------
    def fit(self, x):
        """Fit the fiducial amplitudes, the centres, the branch shares and
        the whitening to the points ``x`` (with ``logL`` for the
        fiducial)."""
        x = np.atleast_1d(x)
        i = self._fiducial_index(x)
        self.fiducial = {n: float(x[n][i]) for n in EXTRINSIC}
        z = self._response(x["ra"], x["dec"])
        signs = (self.fixed_phase_sign,) if self.fixed_phase_sign else (-1, 1)
        spread = {}
        for sign in signs:
            a, b = self._alpha_beta(
                x["theta_jn"], x["luminosity_distance"], x["psi"], x["phase"], sign
            )
            r = self._amplitudes(z, a, b)
            r = r / r[i]
            spread[sign] = 2 - sum(abs(np.mean(r[:, j] / np.abs(r[:, j]))) for j in range(2))
        self._choose_sign(spread)
        a, b = self._alpha_beta(
            x["theta_jn"], x["luminosity_distance"], x["psi"], x["phase"], self.phase_sign
        )
        amps = self._amplitudes(z, a, b)
        self.reference_amplitudes = amps[i].copy()
        self._arg_a_fid = float(np.angle(a[i]))
        self._arg_b_fid = float(np.angle(b[i]))
        r = amps / self.reference_amplitudes
        self.arg_offsets = np.array([_circular_mean(np.angle(r[:, j])) for j in range(2)])
        self.azimuth_centre = _circular_mean(self._ring(x["ra"], x["dec"])[0])
        return self._finish_fit(x)

    # -- maps -------------------------------------------------------------
    def columns_of(self, x):
        """Physical -> unwhitened chart columns ``c``, branch ``k`` and
        ``ln |dc/dx|``."""
        x = np.atleast_1d(x)
        cols, log_j = self._intrinsic_columns(x)
        ra, dec = np.asarray(x["ra"], float), np.asarray(x["dec"], float)
        theta, dist = np.asarray(x["theta_jn"], float), np.asarray(x["luminosity_distance"], float)
        psi, phase = np.asarray(x["psi"], float), np.asarray(x["phase"], float)
        azimuth, cos_alpha = self._ring(ra, dec)
        time = (np.asarray(x["geocent_time"], float) - self.reference_time) + (
            geocenter_time_delay(self.vertices[0], self.gmst, ra, dec)
        )
        z = self._response(ra, dec)
        a, b = self._alpha_beta(theta, dist, psi, phase, self.phase_sign)
        amps = self._amplitudes(z, a, b)
        r = amps / self.reference_amplitudes
        cols += [
            _wrap(azimuth - self.azimuth_centre),
            cos_alpha,
            time,
            np.log(np.abs(r[:, 0])),
            np.log(np.abs(r[:, 1])),
            _wrap(np.angle(r[:, 0]) - self.arg_offsets[0]),
            _wrap(np.angle(r[:, 1]) - self.arg_offsets[1]),
        ]
        log_j += np.log(np.cos(dec)) + self._amplitude_log_j(z, a, b, amps, dist, theta)
        psi_c, phase_c = self._centre(a, b)
        k = self._label(psi - psi_c, phase - phase_c)
        return np.stack(cols, axis=1), k, log_j

    def _inverse(self, c, k):
        """Unwhitened columns ``c`` and branch ``k`` -> the physical
        parameters, the rows in the chart's domain and ``ln |dc/dx|``."""
        n_int = len(self.intrinsic)
        out, log_j = self._intrinsic_inverse(c)
        u, s, t, l1, l2, r1, r2 = (c[:, n_int + j] for j in range(7))
        valid = (
            (np.abs(s) < 1)
            & (u >= -np.pi) & (u < np.pi)
            & (r1 >= -np.pi) & (r1 < np.pi)
            & (r2 >= -np.pi) & (r2 < np.pi)
            & np.all(np.isfinite(c), axis=1)
        )
        s = np.where(valid, s, 0.0)
        ra, dec = self._sky(np.where(valid, u, 0.0) + self.azimuth_centre, s)
        z = self._response(ra, dec)
        logs = np.stack([np.where(valid, l1, 0.0), np.where(valid, l2, 0.0)], axis=1)
        args = np.stack([r1, r2], axis=1) + self.arg_offsets
        amps = self.reference_amplitudes * np.exp(logs + 1j * np.where(valid[:, None], args, 0.0))
        det = self._det(z)
        a = (amps[:, 0] * np.conj(z[:, 1]) - np.conj(z[:, 0]) * amps[:, 1]) / det
        b = (z[:, 0] * amps[:, 1] - z[:, 1] * amps[:, 0]) / det
        sa, sb = np.sqrt(np.abs(a)), np.sqrt(np.abs(b))
        theta = np.arccos(np.clip((sa - sb) / (sa + sb), -1.0, 1.0))
        dist = (sa + sb) ** -2
        psi_c, phase_c = self._centre(a, b)
        valid &= (np.abs(det) > 0) & (sa > 0) & (sb > 0) & np.isfinite(dist) & (np.sin(theta) > 0)
        out.update(
            ra=ra,
            dec=dec,
            geocent_time=self.reference_time + t - geocenter_time_delay(
                self.vertices[0], self.gmst, ra, dec
            ),
            theta_jn=theta,
            luminosity_distance=dist,
            psi=np.remainder(psi_c + LATTICE[k, 0], np.pi),
            phase=np.remainder(phase_c + LATTICE[k, 1], _TWO_PI),
        )
        log_j += np.log(np.cos(dec)) + self._amplitude_log_j(z, a, b, amps, dist, theta)
        return out, valid, log_j

    def describe(self):
        if not self.fitted:
            return "unfitted"
        return (
            f"ring azimuth centre {self.azimuth_centre:.3f}; phase sign "
            f"{self.phase_sign:+d}; branch shares {np.round(self.p_k, 3).tolist()}; "
            f"sd " + self._column_sd()
        )


class ChartFlowProposal(FlowProposal):
    """A flow proposal in a chart (:class:`SheetChart`, :class:`RingChart`;
    see the module notes).

    The chart replaces nessai's reparameterisations: it is refitted to the
    training points (``check_state``) at every training, and the branch
    label is drawn on the inverse pass. Draws outside the chart's domain
    (a cosine at or beyond 1, a wrapped coordinate outside its period) are
    returned outside the prior bounds, so the prior-bounds check drops
    them.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
    chart : :class:`SheetChart` or :class:`RingChart`
    flow_config, training_config : dict, optional
        Of the flow; resuming keeps them (the sampler passes the
        configuration of its main proposal).
    kwargs
        Passed to :class:`nessai.proposal.FlowProposal`.
    """

    use_default_reparameterisations = False
    #: a partitioned run does not split the proposal's region again (nessai
    #: skips the check): the split coordinates need the group proposal's
    #: sky frame
    splittable = False

    def __init__(self, model, chart, flow_config=None, training_config=None, **kwargs):
        super().__init__(
            model,
            flow_config=copy.deepcopy(flow_config),
            training_config=copy.deepcopy(training_config),
            **kwargs,
        )
        self.chart = chart
        # (named for the sheet chart, which came first: kept for its checkpoints)
        self.sheet_flow_config = copy.deepcopy(flow_config or {})
        self.chart_history = []

    # -- coordinates ------------------------------------------------------
    def set_rescaling(self):
        self.parameters = list(self.model.names)
        self.prime_parameters = list(self.chart.columns)
        self._prime_parameters_internal = list(self.prime_parameters)
        self._x_dtype = None
        self._x_prime_dtype = None
        self._x_prime_internal_dtype = None
        self._reparameterisation = None
        self.should_update_reparameterisations = False
        self.rescaling_set = True
        self._truncation_scheme.configure(self)

    def verify_rescaling(self):
        """The chart is fitted at the first training; its round trip is
        checked there (:meth:`check_state`)."""

    def check_state(self, x):
        """Refit the chart to the training points and check its round
        trip on them."""
        self.chart.fit(x)
        y, k, log_j = self.chart.forward(x)
        out, valid, log_j_inv = self.chart.inverse(y, k)
        err = 0.0
        for n in self.parameters:
            a, b = np.asarray(x[n], float), out[n]
            period = {"ra": _TWO_PI, "psi": np.pi, "phase": _TWO_PI}.get(n)
            d = _wrap(a - b, period) if period else (a - b) / np.maximum(np.abs(a), 1.0)
            err = max(err, float(np.max(np.abs(np.where(valid, d, 0.0)))))
        jac = float(np.max(np.abs(np.where(valid, log_j - log_j_inv, 0.0))))
        entry = dict(
            training_count=int(self.training_count),
            fiducial=dict(self.chart.fiducial),
            phase_sign=int(self.chart.phase_sign),
            branch_shares=self.chart.p_k.tolist(),
            invalid=int((~valid).sum()),
            roundtrip_error=err,
            jacobian_error=jac,
        )
        self.chart_history.append(entry)
        level = logging.WARNING if (err > 1e-6 or jac > 1e-6 or not valid.all()) else logging.INFO
        logger.log(
            level,
            "Sheet chart refit: %s; round trip %.1e (log J %.1e), %d of %d training "
            "points outside the chart",
            self.chart.describe(), err, jac, int((~valid).sum()), len(valid),
        )

    def rescale(self, x, **kwargs):
        x = np.atleast_1d(x)
        y, _, log_j = self.chart.forward(x)
        x_prime = empty_structured_array(len(y), dtype=self.x_prime_internal_dtype)
        for i, n in enumerate(self.prime_parameters):
            x_prime[n] = y[:, i]
        for p in nessai_config.livepoints.non_sampling_parameters:
            x_prime[p] = x[p]
        return x_prime, log_j

    def inverse_rescale(self, x_prime, return_unit_hypercube=False, **kwargs):
        y = live_points_to_array(x_prime, self.prime_parameters, copy=True)
        y = np.atleast_2d(y).reshape(len(x_prime), -1)
        k = self.rng.choice(len(LATTICE), size=len(y), p=self.chart.p_k)
        out, valid, log_j = self.chart.inverse(y, k)
        x = empty_structured_array(len(y), dtype=self.x_dtype)
        for n in self.parameters:
            # outside the chart: outside the prior bounds, dropped there
            x[n] = np.where(valid, out[n], np.inf)
        for p in nessai_config.livepoints.non_sampling_parameters:
            x[p] = x_prime[p]
        return x, -log_j

    # -- life cycle -----------------------------------------------------
    def reset(self):
        self.indices = []
        self.samples = None
        self.x = None
        self.populated = False
        self.populated_count = 0
        self.population_acceptance = None
        self._poolsize_scale = 1.0
        self._checked_population = True
        self.acceptance = []
        self._truncation_scheme.reset()

    def resume(self, model, flow_config, weights_file=None):
        """Resume with the proposal's own flow configuration (the sampler
        passes the one of its main proposal)."""
        super().resume(model, copy.deepcopy(self.sheet_flow_config), weights_file)


#: Flow and training settings of the offline test (ET-Delta v70 region 1).
DEFAULT_FLOW_CONFIG = {
    "ftype": "realnvp",
    "n_blocks": 8,
    "n_layers": 2,
    "n_neurons": 64,
    "batch_norm_between_layers": True,
    "linear_transform": "lu",
}
DEFAULT_TRAINING_CONFIG = {
    "lr": 0.001,
    "batch_size": 256,
    "max_epochs": 1500,
    "patience": 60,
    "optimiser": "adamw",
    "optimiser_kwargs": {"weight_decay": 1e-5},
    "val_size": 0.1,
    "clip_grad_norm": 5.0,
}


class SheetChartFlowProposal(ChartFlowProposal):
    """A flow proposal in the sheet chart of a triangle
    (:class:`SheetChart`): for a partitioned run's edge-on region
    (:class:`SheetChartRegions`).

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
    plane_normal, vertex, reference_time, phase_sign
        As for :class:`SheetChart`.
    flow_config, training_config, kwargs
        As for :class:`ChartFlowProposal`.
    """

    def __init__(self, model, plane_normal, vertex, reference_time, phase_sign=None,
                 flow_config=None, training_config=None, **kwargs):
        chart = SheetChart(plane_normal, vertex, reference_time, model.names, phase_sign)
        super().__init__(model, chart, flow_config, training_config, **kwargs)


class RingChartFlowProposal(ChartFlowProposal):
    """A flow proposal in the ring chart of two separated interferometers
    (:class:`RingChart`): a whole run's proposal.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
    tensors, vertices, reference_time, phase_sign
        As for :class:`RingChart`.
    flow_config, training_config
        Default :data:`DEFAULT_FLOW_CONFIG`, :data:`DEFAULT_TRAINING_CONFIG`.
    kwargs
        As for :class:`ChartFlowProposal`.
    """

    def __init__(self, model, tensors, vertices, reference_time, phase_sign=None,
                 flow_config=None, training_config=None, **kwargs):
        chart = RingChart(tensors, vertices, reference_time, model.names, phase_sign)
        super().__init__(
            model,
            chart,
            flow_config or DEFAULT_FLOW_CONFIG,
            training_config or DEFAULT_TRAINING_CONFIG,
            **kwargs,
        )


class SheetChartRegions:
    """``region_proposal`` factory for nessai's partitioned nested sampling:
    a region whose live points are mostly edge-on (median ``|cos
    theta_jn|`` below ``max_abs_cos_theta_jn``) gets a
    :class:`SheetChartFlowProposal`; the others keep their copy of the
    parent's proposal.

    Parameters
    ----------
    plane_normal, vertex, reference_time, phase_sign
        As for :class:`SheetChart`.
    max_abs_cos_theta_jn : float
        The criterion above.
    flow_config, training_config : dict, optional
        Of the new proposal's flow (default :data:`DEFAULT_FLOW_CONFIG`,
        :data:`DEFAULT_TRAINING_CONFIG`).
    """

    #: settings copied from the region's copy of the parent's proposal
    copied = (
        "drawsize",
        "accumulate_weights",
        "check_acceptance",
        "threshold_from_live_points",
        "clip_population_weights",
        "population_weight_cap_mass",
    )

    def __init__(
        self,
        plane_normal,
        vertex,
        reference_time,
        phase_sign=None,
        max_abs_cos_theta_jn=0.5,
        flow_config=None,
        training_config=None,
    ):
        self.plane_normal = [float(v) for v in plane_normal]
        self.vertex = [float(v) for v in vertex]
        self.reference_time = float(reference_time)
        self.phase_sign = phase_sign
        self.max_abs_cos_theta_jn = float(max_abs_cos_theta_jn)
        self.flow_config = dict(flow_config or DEFAULT_FLOW_CONFIG)
        self.training_config = dict(training_config or DEFAULT_TRAINING_CONFIG)

    def wants(self, live_points):
        cos = np.abs(np.cos(np.asarray(live_points["theta_jn"], float)))
        return bool(len(cos)) and float(np.median(cos)) < self.max_abs_cos_theta_jn

    def __call__(self, clone, region, sampler):
        live = region.live_points
        median = float(np.median(np.abs(np.cos(live["theta_jn"])))) if len(live) else np.nan
        if not self.wants(live):
            logger.info(
                f"Sheet chart: region {region.label} keeps its copy (median "
                f"|cos theta_jn| {median:.2f} >= {self.max_abs_cos_theta_jn})"
            )
            return None
        kwargs = {k: getattr(clone, k) for k in self.copied if hasattr(clone, k)}
        proposal = SheetChartFlowProposal(
            sampler.model,
            self.plane_normal,
            self.vertex,
            self.reference_time,
            phase_sign=self.phase_sign,
            flow_config=self.flow_config,
            training_config=self.training_config,
            poolsize=getattr(clone, "_poolsize", None),
            update_poolsize=getattr(clone, "update_poolsize", True),
            max_poolsize_scale=getattr(clone, "max_poolsize_scale", 10),
            plot=False,
            **kwargs,
        )
        logger.info(
            f"Sheet chart: region {region.label} (median |cos theta_jn| "
            f"{median:.2f}) samples in the sheet chart"
        )
        return proposal
