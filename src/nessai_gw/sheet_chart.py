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

At face-on ``|beta| = sin^4(theta_jn / 2) / d_L`` vanishes as
``theta_jn^4``, so the isotropic prior has a ``|beta|^(-3/2)`` spike on
``beta = 0`` in the amplitude columns (and on ``alpha = 0`` at face-off),
which a flow does not follow: on the ET 2L v78 run the face-on live points
had weights ``e^9`` above the rest and the run lost three quarters of the
posterior at ``|cos theta_jn| > 0.95``. ``pole_regularisation`` (``t0``)
builds the amplitudes from ``c+ = C^4 + t0 C S^3`` and ``c- = S^4 + t0 S
C^3`` (``S, C`` the sine and cosine of ``theta_jn / 2``) instead of ``C^4,
S^4``: ``|beta|`` then grows as ``t0 theta_jn / 2`` and the prior is smooth
at the poles, while away from them (``tan(theta_jn / 2)^3 >> t0``) the
chart is the plain one.

Intrinsic coordinates
---------------------
By default the charts carry the intrinsic parameters as they are (the
mass ratio as ``ln q``). ``coordinates`` (:meth:`_Chart.set_coordinates`)
gives them the group proposals' coordinates, with the same nessai-gw
reparameterisations: the effective spins and tides, the chirp mass in the
Earth's frame ``chirp_mass (1 - n.v/c)`` (it reads the sky, which the
inverse takes first), the arrival time when the 22 mode passes a reference
frequency (a shift of the chart's time by ``tau(f) - tau_ref``) and the
phase at a reference frequency (``phase + a(theta) / 2`` in ``(alpha,
beta)``). The shifts depend on the intrinsic parameters and the sky only,
so the Jacobian stays triangular in the order (sky, intrinsic, time,
amplitudes) and only the Doppler factor adds to it. The reference
frequencies are chosen at every refit, on the training points.

One interferometer
------------------
A single interferometer (CE) measures one complex amplitude and an
arrival time: the sky, ``theta_jn`` and ``psi`` are free along the
degeneracy, and :class:`SingleChart` samples them with the arrival time,
``ln |A / A*|`` and ``arg(A / A*)``. On the live points of a CE BBH run the
detector phase ``arg A`` narrows to 0.01 rad by ln X -55 while
``phase + sign(cos theta_jn) psi``, which lacks the sky-dependent
``arg(Z + rho conj(Z))``, stays uniform.
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
from .reparameterisations.mass import doppler_factor
from .reparameterisations.phase import PolarisationPhaseReparameterisation
from .reparameterisations.spin import EffectiveSpinReparameterisation
from .reparameterisations.tides import EffectiveTidalDeformabilityReparameterisation
from .reparameterisations.time import DetectorCenterTimeReparameterisation

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
#: Keywords of :meth:`_Chart.set_coordinates` (``coordinates`` of the
#: proposals), off by default.
COORDINATES = (
    "effective_spin",
    "effective_tidal_deformability",
    "doppler_vector",
    "time_reference_frequency",
    "time_to_merger",
    "waveform_phase",
    "time_frequency_range",
)


def _wrap(x, period=_TWO_PI):
    """``x`` in ``[-period / 2, period / 2)``."""
    return np.remainder(np.asarray(x, dtype=float) + 0.5 * period, period) - 0.5 * period


def _circular_mean(angles):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(angles, dtype=float)))))


class _Chart:
    """What the charts share: the intrinsic columns (``mass_ratio`` as ``ln
    q``), the polarisation amplitudes ``(alpha, beta)``, the ``(psi,
    phase)`` branch label, the automatic phase sign, the whitening and the
    branch shares. A chart defines :meth:`fit` (which starts with
    :meth:`_update_coordinates` and ends with :meth:`_finish_fit`),
    :meth:`columns_of`, :meth:`_inverse`, its column names and the vertex
    its arrival time is taken at (``_timing_vertex``).

    :meth:`set_coordinates` swaps in the group proposals' coordinates of
    the intrinsic parameters and of their couplings to the extrinsic ones
    (see the module notes)."""

    #: prefix of the column names
    prefix = "chart_"
    #: (charts pickled before the coordinates: none of them)
    coordinates = None
    doppler_vector = None
    _spin = None
    _tides = None
    _time = None
    _phase = None

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
        self._extrinsic_columns = list(extrinsic_columns)
        self._set_columns()
        self.fitted = False
        self.fiducial = None

    def _set_columns(self):
        renamed = {n: f"ln_{n}" for n in LOG_PARAMETERS}
        if self.doppler_vector is not None:
            renamed["chirp_mass"] = "doppler_chirp_mass"
        for rep in (self._spin, self._tides):
            if rep is not None:
                renamed.update(zip(rep.parameters, rep.prime_parameters))
        self.columns = [
            f"{self.prefix}{renamed.get(n, n)}" for n in self.intrinsic
        ] + list(self._extrinsic_columns)

    # -- coordinates ----------------------------------------------------
    def set_coordinates(
        self,
        prior_bounds,
        effective_spin=False,
        effective_tidal_deformability=False,
        doppler_vector=None,
        time_reference_frequency=None,
        time_to_merger=None,
        waveform_phase=None,
        time_frequency_range=(8.0, 300.0),
    ):
        """Sample in the group proposals' coordinates of the intrinsic
        parameters (the same nessai-gw reparameterisations, see
        :func:`nessai_gw.group_mixture.triangular_group_reparameterisations`).

        ``effective_spin``: the aligned spins as ``(chi_eff_prime,
        chi_diff_prime)``; ``effective_tidal_deformability``: the tides as
        ``(lambda_tilde_prime, lambda_diff_prime)``; ``doppler_vector``
        (``v / c`` of the Earth): the chirp mass in the Earth's frame,
        ``chirp_mass (1 - n.v/c)``; ``time_reference_frequency`` (Hz or
        ``"adaptive"``; with ``time_to_merger`` and
        ``time_frequency_range``): the arrival time when the 22 mode passes
        that frequency instead of at merger; ``waveform_phase``: the phase
        at a reference frequency, ``phase + a(theta) / 2``, in ``(alpha,
        beta)``. The reference frequencies and their fits are updated at
        every :meth:`fit`, as the group proposals update them before every
        training. The spins and tides do nothing without both parameters,
        as there.

        Parameters
        ----------
        prior_bounds : dict
            The model's bounds (those of the spins and the tides are used).
        """
        names = set(self.names)
        spins = ["chi_1", "chi_2"] if {"chi_1", "chi_2"} <= names else None
        tides = ["lambda_1", "lambda_2"] if {"lambda_1", "lambda_2"} <= names else None
        bounds = {n: [float(v) for v in prior_bounds[n]] for n in self.names}
        if ((time_reference_frequency is not None or waveform_phase is not None)
                and not {"chirp_mass", "mass_ratio"} <= names):
            raise ValueError("time_reference_frequency and waveform_phase need "
                             "'chirp_mass' and 'mass_ratio'")
        if time_to_merger is not None and time_reference_frequency is None:
            raise ValueError("time_to_merger needs time_reference_frequency")
        if (effective_spin or effective_tidal_deformability) and "mass_ratio" not in names:
            raise ValueError("effective_spin and effective_tidal_deformability "
                             "need 'mass_ratio'")
        if doppler_vector is not None and "chirp_mass" not in names:
            raise ValueError("doppler_vector needs 'chirp_mass'")
        frequency_range = [float(f) for f in time_frequency_range]
        dv = None if doppler_vector is None else [float(v) for v in doppler_vector]
        self.doppler_vector = None if dv is None else np.asarray(dv)
        self._spin = (
            EffectiveSpinReparameterisation(
                parameters=spins, prior_bounds={n: bounds[n] for n in spins}
            )
            if effective_spin and spins else None
        )
        self._tides = (
            EffectiveTidalDeformabilityReparameterisation(
                parameters=tides, prior_bounds={n: bounds[n] for n in tides}
            )
            if effective_tidal_deformability and tides else None
        )
        self._time = (
            DetectorCenterTimeReparameterisation(
                parameters=["geocent_time"],
                prior_bounds={"geocent_time": bounds["geocent_time"]},
                vertex=self._timing_vertex,
                reference_time=self.reference_time,
                reference_frequency=time_reference_frequency,
                frequency_range=frequency_range,
                doppler_vector=dv,
                spins=spins,
                time_to_merger=time_to_merger,
                tides=tides if time_to_merger is not None else None,
            )
            if time_reference_frequency is not None else None
        )
        self._phase = (
            PolarisationPhaseReparameterisation(
                parameters=["phase"],
                prior_bounds={"phase": bounds["phase"]},
                waveform_phase=waveform_phase,
                frequency_range=frequency_range,
                doppler_vector=dv,
                spins=spins,
                tides=tides,
            )
            if waveform_phase is not None else None
        )
        self.coordinates = {
            k: v for k, v in dict(
                effective_spin=self._spin is not None,
                effective_tidal_deformability=self._tides is not None,
                doppler_vector=dv,
                time_reference_frequency=time_reference_frequency,
                time_to_merger=time_to_merger,
                waveform_phase=waveform_phase,
            ).items() if v
        }
        self._set_columns()
        self.fitted = False
        return self

    def _update_coordinates(self, x):
        """Choose the reference frequencies and refit the time and phase
        offsets on the points ``x`` (at the start of every :meth:`fit`)."""
        for rep in (self._time, self._phase):
            if rep is not None:
                rep.update(x)

    def _time_shift(self, x):
        """``tau(f) - tau_ref`` (s), subtracted from the arrival time (0
        with the time at merger)."""
        return 0.0 if self._time is None else self._time._shift(x)

    def _phase_shift(self, x):
        """``a(theta) / 2`` (rad), added to the phase in ``(alpha, beta)``
        (0 with the phase at merger)."""
        return 0.0 if self._phase is None else self._phase._shift(x)

    def _shifts_of(self, out):
        """The time and phase shifts at the inverse's intrinsic parameters
        and sky (``out``, a dict of arrays)."""
        if self._time is None and self._phase is None:
            return 0.0, 0.0
        n = len(out["ra"])
        x = np.empty(n, dtype=[(k, float) for k in out])
        for k, v in out.items():
            x[k] = v
        return self._time_shift(x), self._phase_shift(x)

    def _describe_coordinates(self):
        parts = []
        if self._time is not None and self._time.reference_frequency is not None:
            parts.append(f"time at {self._time.reference_frequency:.1f} Hz")
        if self._phase is not None:
            f = self._phase.reference_frequency
            parts.append("phase at merger" if f is None else f"phase at {f:.1f} Hz")
        return "".join(p + "; " for p in parts)

    @property
    def dims(self):
        return len(self.columns)

    @property
    def n_labels(self):
        """Number of discrete labels: the ``(psi, phase)`` branches (times
        the ring's arcs, :class:`RingChart`)."""
        return len(LATTICE)

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
        """The phase sign whose residual phases spread the least. (Keeping
        the previous fit's sign unless the other was twice as narrow held a
        sign picked on a broad live set: on the ET-Delta v78 edge-on region
        +1 from the split on, while -1 was 2-4 times narrower.) The sign is
        the waveform convention's: give it (``phase_sign``) when known."""
        self.phase_sign = min(spread, key=spread.get)
        self.spread = spread[self.phase_sign]

    def _finish_fit(self, x):
        """The branch shares and the whitening, from the chart's columns of
        ``x`` (the rest of the fit done)."""
        self.mean = np.zeros(self.dims)
        self.chol = np.eye(self.dims)
        self.log_p_k = np.zeros(self.n_labels)
        self.log_det_whiten = 0.0
        self.fitted = True
        c, k, _ = self.columns_of(x)
        counts = np.bincount(k, minlength=self.n_labels)
        self.p_k = (counts + 1.0) / (len(k) + self.n_labels)
        self.log_p_k = np.log(self.p_k)
        self.mean = c.mean(axis=0)
        cov = np.cov(c, rowvar=False)
        cov += np.diag(1e-12 * np.diag(cov) + 1e-300)
        self.chol = np.linalg.cholesky(cov)
        self.log_det_whiten = -float(np.sum(np.log(np.diag(self.chol))))
        return self

    def _intrinsic_columns(self, x):
        """The intrinsic columns of ``x`` (a list) and their ``ln |dc/dx|``
        (the Doppler chirp mass reads the sky)."""
        cols, log_j = {}, np.zeros(len(x))
        for n in self.intrinsic:
            if n in LOG_PARAMETERS:
                cols[n] = np.log(x[n])
                log_j -= np.log(x[n])
            elif n == "chirp_mass" and self.doppler_vector is not None:
                d = doppler_factor(x["ra"], x["dec"], self.doppler_vector)
                cols[n] = np.asarray(x[n], dtype=float) * d
                log_j += np.log(d)
            else:
                cols[n] = np.asarray(x[n], dtype=float)
        for rep in (self._spin, self._tides):
            if rep is not None:
                prime = {}
                _, _, log_j = rep.reparameterise(x, prime, log_j)
                for n, p in zip(rep.parameters, rep.prime_parameters):
                    cols[n] = prime[p]
        return [cols[n] for n in self.intrinsic], log_j

    def _intrinsic_inverse(self, c, ra=None, dec=None):
        """The intrinsic parameters of the columns ``c`` (a dict) and the
        forward ``ln |dc/dx|`` at them; ``ra``, ``dec`` for the Doppler
        chirp mass (the sky is inverted first)."""
        out, log_j = {}, np.zeros(len(c))
        for i, n in enumerate(self.intrinsic):
            if n in LOG_PARAMETERS:
                out[n] = np.exp(c[:, i])
                log_j -= c[:, i]
            elif n == "chirp_mass" and self.doppler_vector is not None:
                d = doppler_factor(ra, dec, self.doppler_vector)
                out[n] = c[:, i] / d
                log_j += np.log(d)
            else:
                out[n] = c[:, i]
        for rep in (self._spin, self._tides):
            if rep is not None:
                prime = {
                    p: c[:, self.intrinsic.index(n)]
                    for n, p in zip(rep.parameters, rep.prime_parameters)
                }
                # the inverse returns minus the forward ln |dc/dx|
                _, _, minus = rep.inverse_reparameterise(out, prime, np.zeros(len(c)))
                log_j -= minus
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
        ``s`` in ``e^{2i s phase}``, the waveform convention's (bilby's
        waveforms: -1); ``None`` picks, at every fit, the sign that makes
        the phase residuals narrower, which on broad live sets can be either.
    """

    prefix = "sheet_"

    def __init__(self, plane_normal, vertex, reference_time, names, phase_sign=None):
        self.plane_normal = np.asarray(plane_normal, dtype=float)
        self.vertex = np.asarray(vertex, dtype=float)
        self.reference_time = float(reference_time)
        self.gmst = greenwich_mean_sidereal_time(self.reference_time)
        self._set_names(names, phase_sign, SHEET_COLUMNS)
        self.ellipse = None

    @property
    def _timing_vertex(self):
        return self.vertex

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
        self._update_coordinates(x)
        phase = x["phase"] + self._phase_shift(x)
        self.fiducial = {n: float(x[n][i]) for n in ("ra", "dec", "psi", "theta_jn")}
        self.ellipse = PolarisationEllipse(self.plane_normal, self.reference_time, self.fiducial)
        a0, b0 = self._sheet(x["ra"], x["dec"])
        signs = (self.fixed_phase_sign,) if self.fixed_phase_sign else (-1, 1)
        spread = {}
        for sign in signs:
            a, b = self._alpha_beta(
                x["theta_jn"], x["luminosity_distance"], x["psi"], phase, sign
            )
            ra_, rb_ = a / a0, b / b0
            # circular spread of the residual phases (1 - mean resultant length)
            spread[sign] = 2 - abs(np.mean(ra_ / np.abs(ra_))) - abs(np.mean(rb_ / np.abs(rb_)))
        self._choose_sign(spread)
        a, b = self._alpha_beta(
            x["theta_jn"], x["luminosity_distance"], x["psi"], phase, self.phase_sign
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
        psi = np.asarray(x["psi"], float)
        phase = np.asarray(x["phase"], float) + self._phase_shift(x)
        delay = geocenter_time_delay(self.vertex, self.gmst, ra, dec)
        cols += [
            _wrap(ra - self.ra_centre),
            np.sin(dec),
            (np.asarray(x["geocent_time"], float) - self.reference_time) + delay
            - self._time_shift(x),
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
        out, log_j = self._intrinsic_inverse(c, ra, dec)
        t_shift, phase_shift = self._shifts_of(out | {"ra": ra, "dec": dec})
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
        phase = np.remainder(phase_c + dphase + LATTICE[k, 1] - phase_shift, _TWO_PI)
        valid &= (sa > 0) & (sb > 0) & np.isfinite(dist) & (np.sin(theta) > 0)
        out.update(
            ra=ra,
            dec=dec,
            geocent_time=self.reference_time + t + t_shift - delay,
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
            + self._describe_coordinates() + f"phase sign {self.phase_sign:+d}; branch shares "
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
    = Z_1 conj(Z_2) - conj(Z_1) Z_2 = 2i Im(Z_1 conj(Z_2))`` vanishes (the
    sky directions where the two interferometers see the same
    polarisation), a set of measure zero which the ring crosses (for ET 2L
    twice, about half a turn apart): there ``(alpha, beta)`` diverge, the
    distance goes to zero and with it the prior, and ``cos theta_jn``
    changes sign. The posterior along the ring is then separate arcs, which
    one flow joins with leakage into the gaps between them (on the ET 2L
    v78 run 40% of the draws, and the weight cap set by the live points on
    their flanks: a population acceptance of ~2e-3). With ``arcs`` the ring
    is cut at the crossings (refitted at every fit, along the ring at the
    median angle to the baseline), the arc is a label with the ``(psi,
    phase)`` branch and the azimuth column is the offset from the arc's
    centre, so the arcs overlap in the chart (a shift: the Jacobian is
    unchanged).

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
    pole_regularisation : float
        ``t0`` in ``[0, sqrt(2))`` (see the module notes): the amplitudes
        built from ``c+ = C^4 + t0 C S^3``, ``c- = S^4 + t0 S C^3`` so that
        the prior is smooth at face-on and face-off; 0 is the plain chart.
        With ``t0 > 0`` the amplitude block's Jacobian is ``|det M|^2
        |alpha| |beta| 8 |c+ c-' - c- c+'| / d_L^3 / (|A_1|^2 |A_2|^2)``.
    arcs : bool
        Cut the ring into its arcs at the crossings of ``det M = 0`` (see
        above).
    """

    prefix = "ring_"
    #: (charts pickled before the options: the plain chart)
    pole_regularisation = 0.0
    arcs = False
    arc_centres = None

    def __init__(self, tensors, vertices, reference_time, names, phase_sign=None,
                 pole_regularisation=0.0, arcs=False):
        if not 0.0 <= pole_regularisation < np.sqrt(2.0):
            raise ValueError("pole_regularisation must be in [0, sqrt(2))")
        self.pole_regularisation = float(pole_regularisation)
        self.arcs = bool(arcs)
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

    @property
    def _timing_vertex(self):
        return self.vertices[0]

    @property
    def n_labels(self):
        n_arcs = 1 if self.arc_centres is None else len(self.arc_centres)
        return len(LATTICE) * n_arcs

    # -- the arcs ---------------------------------------------------------
    def _fit_arcs(self, x):
        """The crossings of ``Im(Z_1 conj(Z_2)) = 0`` along the ring at the
        median angle to the baseline of ``x``, and the arcs between them
        (none, a single arc, if there are fewer than two)."""
        self.arc_centres = self.arc_half_widths = None
        if not self.arcs:
            return
        cos_alpha = float(np.median(self._ring(x["ra"], x["dec"])[1]))
        grid = np.linspace(-np.pi, np.pi, 4097)[:-1]
        z = self._response(*self._sky(grid, np.full(len(grid), cos_alpha)))
        f = np.imag(z[:, 0] * np.conj(z[:, 1]))
        f1 = np.roll(f, -1)
        j = np.flatnonzero(np.sign(f) != np.sign(f1))
        if len(j) < 2:
            logger.warning("Ring chart: %d crossing(s) of det M = 0 along the ring: one arc", len(j))
            return
        step = grid[1] - grid[0]
        bounds = np.sort(np.remainder(grid[j] + step * f[j] / (f[j] - f1[j]) + np.pi, _TWO_PI) - np.pi)
        lengths = np.remainder(np.roll(bounds, -1) - bounds, _TWO_PI)
        lengths[lengths == 0] = _TWO_PI
        self.arc_half_widths = 0.5 * lengths
        self.arc_centres = _wrap(bounds + self.arc_half_widths)

    def _arc_of(self, azimuth):
        """The arc each azimuth is on and its offset from the arc's centre."""
        u = _wrap(np.asarray(azimuth, float)[:, None] - self.arc_centres[None, :])
        j = np.argmax(self.arc_half_widths[None, :] - np.abs(u), axis=1)
        return j, u[np.arange(len(j)), j]

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

    def _moduli(self, theta):
        """``c+, c-`` and their derivatives in ``theta_jn``."""
        h = 0.5 * np.asarray(theta, dtype=float)
        s, c, t0 = np.sin(h), np.cos(h), self.pole_regularisation
        cp, cm = c**4 + t0 * c * s**3, s**4 + t0 * s * c**3
        dcp = 0.5 * (-4.0 * c**3 * s + t0 * (3.0 * c**2 * s**2 - s**4))
        dcm = 0.5 * (4.0 * s**3 * c + t0 * (c**4 - 3.0 * s**2 * c**2))
        return cp, cm, dcp, dcm

    def _chart_alpha_beta(self, theta, distance, psi, phase, sign):
        """``(alpha, beta)`` with the chart's moduli (the plain ones without
        pole regularisation)."""
        a, b = self._alpha_beta(theta, distance, psi, phase, sign)
        if not self.pole_regularisation:
            return a, b
        cp, cm, _, _ = self._moduli(theta)
        return cp / distance * np.exp(1j * np.angle(a)), cm / distance * np.exp(1j * np.angle(b))

    def _theta_of(self, a, b):
        """``theta_jn`` from the chart's ``(alpha, beta)``: ``|beta| / |alpha|
        = (u^4 + t0 u) / (1 + t0 u^3)``, ``u = tan(theta_jn / 2)``, monotone
        in ``u`` for ``t0 < sqrt(2)`` (bisection in ``ln u``, then Newton)."""
        t0 = self.pole_regularisation
        rho = np.abs(b) / np.abs(a)
        rho = np.where(np.isfinite(rho), rho, 1.0)
        lo, hi = np.full(rho.shape, -40.0), np.full(rho.shape, 40.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            u = np.exp(mid)
            below = (u**4 + t0 * u) < rho * (1.0 + t0 * u**3)
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        u = np.exp(0.5 * (lo + hi))
        for _ in range(3):
            g = u**4 - rho * t0 * u**3 + t0 * u - rho
            u = u - g / (4.0 * u**3 - 3.0 * rho * t0 * u**2 + t0)
        return 2.0 * np.arctan(u)

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
        if self.pole_regularisation:
            cp, cm, dcp, dcm = self._moduli(theta)
            return (
                2 * np.log(np.abs(self._det(z)))
                + np.log(np.abs(a))
                + np.log(np.abs(b))
                + np.log(8.0)
                + np.log(np.abs(cp * dcm - cm * dcp))
                - 3 * np.log(distance)
                - 2 * np.log(np.abs(amps[:, 0]))
                - 2 * np.log(np.abs(amps[:, 1]))
            )
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
        self._update_coordinates(x)
        phase = x["phase"] + self._phase_shift(x)
        self.fiducial = {n: float(x[n][i]) for n in EXTRINSIC}
        z = self._response(x["ra"], x["dec"])
        signs = (self.fixed_phase_sign,) if self.fixed_phase_sign else (-1, 1)
        spread = {}
        for sign in signs:
            a, b = self._chart_alpha_beta(
                x["theta_jn"], x["luminosity_distance"], x["psi"], phase, sign
            )
            r = self._amplitudes(z, a, b)
            r = r / r[i]
            spread[sign] = 2 - sum(abs(np.mean(r[:, j] / np.abs(r[:, j]))) for j in range(2))
        self._choose_sign(spread)
        a, b = self._chart_alpha_beta(
            x["theta_jn"], x["luminosity_distance"], x["psi"], phase, self.phase_sign
        )
        amps = self._amplitudes(z, a, b)
        self.reference_amplitudes = amps[i].copy()
        self._arg_a_fid = float(np.angle(a[i]))
        self._arg_b_fid = float(np.angle(b[i]))
        r = amps / self.reference_amplitudes
        self.arg_offsets = np.array([_circular_mean(np.angle(r[:, j])) for j in range(2)])
        self.azimuth_centre = _circular_mean(self._ring(x["ra"], x["dec"])[0])
        self._fit_arcs(x)
        return self._finish_fit(x)

    # -- maps -------------------------------------------------------------
    def columns_of(self, x):
        """Physical -> unwhitened chart columns ``c``, branch ``k`` and
        ``ln |dc/dx|``."""
        x = np.atleast_1d(x)
        cols, log_j = self._intrinsic_columns(x)
        ra, dec = np.asarray(x["ra"], float), np.asarray(x["dec"], float)
        theta, dist = np.asarray(x["theta_jn"], float), np.asarray(x["luminosity_distance"], float)
        psi = np.asarray(x["psi"], float)
        phase = np.asarray(x["phase"], float) + self._phase_shift(x)
        azimuth, cos_alpha = self._ring(ra, dec)
        time = (np.asarray(x["geocent_time"], float) - self.reference_time) + (
            geocenter_time_delay(self.vertices[0], self.gmst, ra, dec)
        ) - self._time_shift(x)
        z = self._response(ra, dec)
        a, b = self._chart_alpha_beta(theta, dist, psi, phase, self.phase_sign)
        amps = self._amplitudes(z, a, b)
        r = amps / self.reference_amplitudes
        if self.arc_centres is None:
            arc, offset = 0, _wrap(azimuth - self.azimuth_centre)
        else:
            arc, offset = self._arc_of(azimuth)
        cols += [
            offset,
            cos_alpha,
            time,
            np.log(np.abs(r[:, 0])),
            np.log(np.abs(r[:, 1])),
            _wrap(np.angle(r[:, 0]) - self.arg_offsets[0]),
            _wrap(np.angle(r[:, 1]) - self.arg_offsets[1]),
        ]
        log_j += np.log(np.cos(dec)) + self._amplitude_log_j(z, a, b, amps, dist, theta)
        psi_c, phase_c = self._centre(a, b)
        k = self._label(psi - psi_c, phase - phase_c) + len(LATTICE) * arc
        return np.stack(cols, axis=1), k, log_j

    def _inverse(self, c, k):
        """Unwhitened columns ``c`` and branch ``k`` -> the physical
        parameters, the rows in the chart's domain and ``ln |dc/dx|``."""
        n_int = len(self.intrinsic)
        u, s, t, l1, l2, r1, r2 = (c[:, n_int + j] for j in range(7))
        valid = (
            (np.abs(s) < 1)
            & (u >= -np.pi) & (u < np.pi)
            & (r1 >= -np.pi) & (r1 < np.pi)
            & (r2 >= -np.pi) & (r2 < np.pi)
            & np.all(np.isfinite(c), axis=1)
        )
        s = np.where(valid, s, 0.0)
        if self.arc_centres is None:
            centre = self.azimuth_centre
        else:
            arc = k // len(LATTICE)
            centre = self.arc_centres[arc]
            valid &= np.abs(u) < self.arc_half_widths[arc]
        k = k % len(LATTICE)
        ra, dec = self._sky(np.where(valid, u, 0.0) + centre, s)
        out, log_j = self._intrinsic_inverse(c, ra, dec)
        t_shift, phase_shift = self._shifts_of(out | {"ra": ra, "dec": dec})
        z = self._response(ra, dec)
        logs = np.stack([np.where(valid, l1, 0.0), np.where(valid, l2, 0.0)], axis=1)
        args = np.stack([r1, r2], axis=1) + self.arg_offsets
        amps = self.reference_amplitudes * np.exp(logs + 1j * np.where(valid[:, None], args, 0.0))
        det = self._det(z)
        a = (amps[:, 0] * np.conj(z[:, 1]) - np.conj(z[:, 0]) * amps[:, 1]) / det
        b = (z[:, 0] * amps[:, 1] - z[:, 1] * amps[:, 0]) / det
        if self.pole_regularisation:
            theta = self._theta_of(a, b)
            dist = self._moduli(theta)[0] / np.abs(a)
        else:
            sa, sb = np.sqrt(np.abs(a)), np.sqrt(np.abs(b))
            theta = np.arccos(np.clip((sa - sb) / (sa + sb), -1.0, 1.0))
            dist = (sa + sb) ** -2
        psi_c, phase_c = self._centre(a, b)
        valid &= (
            (np.abs(det) > 0) & (np.abs(a) > 0) & (np.abs(b) > 0) & np.isfinite(dist)
            & (np.sin(theta) > 0)
        )
        out.update(
            ra=ra,
            dec=dec,
            geocent_time=self.reference_time + t + t_shift - geocenter_time_delay(
                self.vertices[0], self.gmst, ra, dec
            ),
            theta_jn=theta,
            luminosity_distance=dist,
            psi=np.remainder(psi_c + LATTICE[k, 0], np.pi),
            phase=np.remainder(phase_c + LATTICE[k, 1] - phase_shift, _TWO_PI),
        )
        log_j += np.log(np.cos(dec)) + self._amplitude_log_j(z, a, b, amps, dist, theta)
        return out, valid, log_j

    def describe(self):
        if not self.fitted:
            return "unfitted"
        return (
            f"ring azimuth centre {self.azimuth_centre:.3f}; "
            + (f"pole regularisation {self.pole_regularisation:g}; "
               if self.pole_regularisation else "")
            + (f"arcs at {np.round(self.arc_centres, 2).tolist()} (half widths "
               f"{np.round(self.arc_half_widths, 2).tolist()}, shares "
               f"{np.round(self.p_k.reshape(-1, len(LATTICE)).sum(1), 3).tolist()}); "
               if self.arc_centres is not None else "")
            + self._describe_coordinates()
            + f"phase sign "
            f"{self.phase_sign:+d}; branch shares {np.round(self.p_k, 3).tolist()}; "
            f"sd " + self._column_sd()
        )


#: Flow coordinates of the extrinsic block of the single-interferometer chart.
SINGLE_COLUMNS = (
    "single_ra",
    "single_sin_dec",
    "single_time",
    "single_cos_theta_jn",
    "single_psi4",
    "single_ln_amplitude",
    "single_arg_amplitude",
)


class SingleChart(_Chart):
    """The chart of a single interferometer.

    One interferometer measures one complex amplitude of the dominant mode,
    ``A = alpha Z(n) + beta conj(Z(n))``, and its arrival time: three of the
    seven extrinsic parameters. The sky, ``theta_jn`` and ``psi`` are free
    along the degeneracy: at fixed ``(n, theta_jn, psi)`` the ratio
    ``rho = beta / alpha = tan^4(theta_jn / 2) e^{4 i psi}`` is fixed, so
    ``alpha = A / (Z + rho conj(Z))`` gives the distance and the phase. The
    chart's coordinates are the sky, the arrival time at the
    interferometer, ``cos theta_jn``, ``4 psi`` (centred and wrapped) and
    ``ln |A / A*|``, ``arg(A / A*)`` relative to the fiducial point's
    (centred), with the ``(psi, phase)`` branch a label as in
    :class:`SheetChart` (``4 psi`` fixes ``psi`` modulo ``pi / 2``, the
    amplitude the phase modulo ``pi``).

    The Jacobian is ``8 sin theta_jn / d_L`` times ``cos dec``; the chart is
    singular only where ``Z + rho conj(Z)`` vanishes (the interferometer
    blind to the polarisation), a set of measure zero.

    The sky group of :mod:`nessai_gw.group_mixture` 's single-interferometer
    proposal is not folded: the flow models the whole sky.

    Parameters
    ----------
    tensor : array_like, shape (3, 3)
        Earth-fixed detector tensor.
    vertex : array_like, shape (3,)
        Its Earth-fixed vertex (m), where the arrival time is taken.
    reference_time, names, phase_sign
        As for :class:`SheetChart`.
    """

    prefix = "single_"

    def __init__(self, tensor, vertex, reference_time, names, phase_sign=None):
        self.tensor = np.asarray(tensor, dtype=float)
        self.vertex = np.asarray(vertex, dtype=float)
        if self.tensor.shape != (3, 3) or self.vertex.shape != (3,):
            raise ValueError("the single chart needs one interferometer's tensor and vertex")
        self.reference_time = float(reference_time)
        self.gmst = greenwich_mean_sidereal_time(self.reference_time)
        self._set_names(names, phase_sign, SINGLE_COLUMNS)

    @property
    def _timing_vertex(self):
        return self.vertex

    def _response(self, ra, dec):
        return _complex_response(
            self.tensor[None], np.atleast_1d(ra), np.atleast_1d(dec), self.gmst
        )[:, 0]

    @staticmethod
    def _rho(cos_theta, psi):
        return ((1 - cos_theta) / (1 + cos_theta)) ** 2 * np.exp(4j * psi)

    def _amplitude(self, z, theta, distance, psi, phase, sign):
        a, b = self._alpha_beta(theta, distance, psi, phase, sign)
        return a * z + b * np.conj(z)

    def _psi_phase(self, amplitude, z, cos_theta, psi4):
        """``alpha``, ``psi`` (modulo ``pi / 2``) and the phase (modulo
        ``pi``) of an amplitude at ``(n, cos theta_jn, 4 psi)``."""
        psi = psi4 / 4
        alpha = amplitude / (z + self._rho(cos_theta, psi) * np.conj(z))
        phase = np.remainder((np.angle(alpha) + 2 * psi) / (2 * self.phase_sign), np.pi)
        return alpha, psi, phase

    # -- fit --------------------------------------------------------------
    def fit(self, x):
        """Fit the fiducial amplitude, the centres, the branch shares and
        the whitening to the points ``x`` (with ``logL`` for the
        fiducial)."""
        x = np.atleast_1d(x)
        i = self._fiducial_index(x)
        self._update_coordinates(x)
        phase = x["phase"] + self._phase_shift(x)
        self.fiducial = {n: float(x[n][i]) for n in EXTRINSIC}
        z = self._response(x["ra"], x["dec"])
        signs = (self.fixed_phase_sign,) if self.fixed_phase_sign else (-1, 1)
        spread = {}
        for sign in signs:
            amp = self._amplitude(
                z, x["theta_jn"], x["luminosity_distance"], x["psi"], phase, sign
            )
            r = amp / amp[i]
            spread[sign] = 1 - abs(np.mean(r / np.abs(r)))
        self._choose_sign(spread)
        amp = self._amplitude(
            z, x["theta_jn"], x["luminosity_distance"], x["psi"], phase, self.phase_sign
        )
        self.reference_amplitude = complex(amp[i])
        self.arg_offset = _circular_mean(np.angle(amp / self.reference_amplitude))
        self.ra_centre = _circular_mean(x["ra"])
        self.psi4_centre = _circular_mean(4 * np.asarray(x["psi"], float))
        return self._finish_fit(x)

    # -- maps -------------------------------------------------------------
    def columns_of(self, x):
        """Physical -> unwhitened chart columns ``c``, branch ``k`` and
        ``ln |dc/dx|``."""
        x = np.atleast_1d(x)
        cols, log_j = self._intrinsic_columns(x)
        ra, dec = np.asarray(x["ra"], float), np.asarray(x["dec"], float)
        theta, dist = np.asarray(x["theta_jn"], float), np.asarray(x["luminosity_distance"], float)
        psi = np.asarray(x["psi"], float)
        phase = np.asarray(x["phase"], float) + self._phase_shift(x)
        z = self._response(ra, dec)
        amp = self._amplitude(z, theta, dist, psi, phase, self.phase_sign)
        r = amp / self.reference_amplitude
        w = _wrap(4 * psi - self.psi4_centre)
        cols += [
            _wrap(ra - self.ra_centre),
            np.sin(dec),
            (np.asarray(x["geocent_time"], float) - self.reference_time)
            + geocenter_time_delay(self.vertex, self.gmst, ra, dec)
            - self._time_shift(x),
            np.cos(theta),
            w,
            np.log(np.abs(r)),
            _wrap(np.angle(r) - self.arg_offset),
        ]
        log_j += np.log(np.cos(dec)) + np.log(8.0) + np.log(np.sin(theta)) - np.log(dist)
        _, psi0, phase0 = self._psi_phase(amp, z, np.cos(theta), w + self.psi4_centre)
        k = self._label(psi - psi0, phase - phase0)
        return np.stack(cols, axis=1), k, log_j

    def _inverse(self, c, k):
        """Unwhitened columns ``c`` and branch ``k`` -> the physical
        parameters, the rows in the chart's domain and ``ln |dc/dx|``."""
        n_int = len(self.intrinsic)
        u, s, t, ct, w, la, r = (c[:, n_int + j] for j in range(7))
        valid = (
            (np.abs(s) < 1)
            & (np.abs(ct) < 1)
            & (u >= -np.pi) & (u < np.pi)
            & (w >= -np.pi) & (w < np.pi)
            & (r >= -np.pi) & (r < np.pi)
            & np.all(np.isfinite(c), axis=1)
        )
        s = np.where(valid, s, 0.0)
        ct = np.where(valid, ct, 0.0)
        ra = np.remainder(u + self.ra_centre, _TWO_PI)
        dec = np.arcsin(s)
        out, log_j = self._intrinsic_inverse(c, ra, dec)
        t_shift, phase_shift = self._shifts_of(out | {"ra": ra, "dec": dec})
        z = self._response(ra, dec)
        theta = np.arccos(ct)
        amp = self.reference_amplitude * np.exp(
            np.where(valid, la, 0.0) + 1j * (np.where(valid, r, 0.0) + self.arg_offset)
        )
        alpha, psi0, phase0 = self._psi_phase(
            amp, z, ct, np.where(valid, w, 0.0) + self.psi4_centre
        )
        dist = (1 + ct) ** 2 / 4 / np.abs(alpha)
        out.update(
            ra=ra,
            dec=dec,
            geocent_time=self.reference_time + t + t_shift - geocenter_time_delay(
                self.vertex, self.gmst, ra, dec
            ),
            theta_jn=theta,
            luminosity_distance=dist,
            psi=np.remainder(psi0 + LATTICE[k, 0], np.pi),
            phase=np.remainder(phase0 + LATTICE[k, 1] - phase_shift, _TWO_PI),
        )
        log_j += np.log(np.cos(dec)) + np.log(8.0) + np.log(np.sin(theta)) - np.log(dist)
        valid &= np.isfinite(dist) & (dist > 0) & (np.sin(theta) > 0)
        return out, valid, log_j

    def describe(self):
        if not self.fitted:
            return "unfitted"
        return (
            self._describe_coordinates()
            + f"phase sign {self.phase_sign:+d}; branch shares "
            f"{np.round(self.p_k, 3).tolist()}; sd " + self._column_sd()
        )


class ChartFlowProposal(FlowProposal):
    """A flow proposal in a chart (:class:`SheetChart`, :class:`RingChart`,
    :class:`SingleChart`; see the module notes).

    The chart replaces nessai's reparameterisations: it is refitted to the
    training points (``check_state``) at every training, and the branch
    label is drawn on the inverse pass. Draws outside the chart's domain
    (a cosine at or beyond 1, a wrapped coordinate outside its period) are
    returned outside the prior bounds, so the prior-bounds check drops
    them.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
    chart : :class:`SheetChart`, :class:`RingChart` or :class:`SingleChart`
    flow_config, training_config : dict, optional
        Of the flow; resuming keeps them (the sampler passes the
        configuration of its main proposal).
    coordinates : dict, optional
        Keywords of :meth:`_Chart.set_coordinates` (the model's bounds are
        added): the group proposals' intrinsic coordinates.
    kwargs
        Passed to :class:`nessai.proposal.FlowProposal`.
    """

    use_default_reparameterisations = False
    #: a partitioned run does not split the proposal's region again (nessai
    #: skips the check): the split coordinates need the group proposal's
    #: sky frame
    splittable = False
    #: (checkpoints from before the option: each batch normalised by its own
    #: largest weight)
    population_to_cap = False

    def __init__(self, model, chart, flow_config=None, training_config=None,
                 coordinates=None, population_to_cap=False, **kwargs):
        super().__init__(
            model,
            flow_config=copy.deepcopy(flow_config),
            training_config=copy.deepcopy(training_config),
            **kwargs,
        )
        if coordinates:
            chart.set_coordinates(model.bounds, **coordinates)
        self.chart = chart
        self.population_to_cap = bool(population_to_cap)
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

    def _get_population_log_weights(self, log_weights, min_yield=None):
        """With ``population_to_cap``, the rejection weights of a batch are
        ``min(w, c) / c`` for the truncation rule's weight cap ``c``: the pool
        follows ``min(p, c q)`` whatever the batch holds. nessai normalises
        each batch by its own largest weight instead, which takes a batch
        without a draw at the cap at a higher rate than the rest: with the
        charts' heavy weight tails (the top 1% of draws holding 70-90% of the
        capped weight) the pool then misses the regions the flow
        under-covers (face-on in the ring chart) and the acceptance reads
        6-125 times the true one."""
        if self.population_to_cap:
            scheme = getattr(self, "_truncation_scheme", None)
            log_cap = scheme.log_weight_cap if scheme is not None else np.inf
            if np.isfinite(log_cap):
                return np.minimum(np.asarray(log_weights, dtype=float) - log_cap, 0.0)
        return super()._get_population_log_weights(log_weights, min_yield=min_yield)

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
        k = self.rng.choice(len(self.chart.p_k), size=len(y), p=self.chart.p_k)
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
                 flow_config=None, training_config=None, coordinates=None, **kwargs):
        chart = SheetChart(plane_normal, vertex, reference_time, model.names, phase_sign)
        super().__init__(model, chart, flow_config, training_config,
                         coordinates=coordinates, **kwargs)


class RingChartFlowProposal(ChartFlowProposal):
    """A flow proposal in the ring chart of two separated interferometers
    (:class:`RingChart`): a whole run's proposal.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
    tensors, vertices, reference_time, phase_sign, pole_regularisation, arcs
        As for :class:`RingChart`.
    flow_config, training_config
        Default :data:`DEFAULT_FLOW_CONFIG`, :data:`DEFAULT_TRAINING_CONFIG`.
    coordinates, kwargs
        As for :class:`ChartFlowProposal`.
    """

    def __init__(self, model, tensors, vertices, reference_time, phase_sign=None,
                 pole_regularisation=0.0, flow_config=None, training_config=None,
                 coordinates=None, arcs=False, **kwargs):
        chart = RingChart(tensors, vertices, reference_time, model.names, phase_sign,
                          pole_regularisation, arcs)
        super().__init__(
            model,
            chart,
            flow_config or DEFAULT_FLOW_CONFIG,
            training_config or DEFAULT_TRAINING_CONFIG,
            coordinates=coordinates,
            **kwargs,
        )



class SingleChartFlowProposal(ChartFlowProposal):
    """A flow proposal in the chart of a single interferometer
    (:class:`SingleChart`): a whole run's proposal.

    Parameters
    ----------
    model : :obj:`nessai.model.Model`
    tensor, vertex, reference_time, phase_sign
        As for :class:`SingleChart`.
    flow_config, training_config
        Default :data:`DEFAULT_FLOW_CONFIG`, :data:`DEFAULT_TRAINING_CONFIG`.
    coordinates, kwargs
        As for :class:`ChartFlowProposal`.
    """

    def __init__(self, model, tensor, vertex, reference_time, phase_sign=None,
                 flow_config=None, training_config=None, coordinates=None, **kwargs):
        chart = SingleChart(tensor, vertex, reference_time, model.names, phase_sign)
        super().__init__(
            model,
            chart,
            flow_config or DEFAULT_FLOW_CONFIG,
            training_config or DEFAULT_TRAINING_CONFIG,
            coordinates=coordinates,
            **kwargs,
        )


def _coordinates(coordinates):
    """``coordinates`` checked and JSON-able (it goes to nessai's config)."""
    if not coordinates:
        return None
    unknown = sorted(set(coordinates) - set(COORDINATES))
    if unknown:
        raise ValueError(f"unknown chart coordinates {unknown}")
    out = dict(coordinates)
    for key in ("doppler_vector", "time_frequency_range"):
        if out.get(key) is not None:
            out[key] = [float(v) for v in out[key]]
    return out


def _bound_proposal_class(base, geometry, class_name):
    """``base`` with ``geometry`` bound into its constructor, published at
    module scope as ``class_name`` so checkpoints unpickle."""

    class _Bound(base):
        def __init__(self, model, flow_config=None, training_config=None, **kwargs):
            super().__init__(
                model,
                flow_config=flow_config,
                training_config=training_config,
                **geometry,
                **kwargs,
            )

    _Bound.__module__ = __name__
    _Bound.__qualname__ = class_name
    _Bound.__name__ = class_name
    globals()[class_name] = _Bound
    return _Bound


def single_chart_proposal_class(
    tensor, vertex, reference_time, phase_sign=None, coordinates=None,
    class_name="BoundSingleChartFlowProposal",
):
    """A :class:`SingleChartFlowProposal` subclass with the interferometer's
    geometry bound (see :func:`ring_chart_proposal_class`)."""
    geometry = dict(
        tensor=np.asarray(tensor, dtype=float).tolist(),
        vertex=np.asarray(vertex, dtype=float).tolist(),
        reference_time=float(reference_time),
        phase_sign=phase_sign,
        coordinates=_coordinates(coordinates),
    )
    return _bound_proposal_class(SingleChartFlowProposal, geometry, class_name)


def ring_chart_proposal_class(
    tensors, vertices, reference_time, phase_sign=None, pole_regularisation=0.0,
    coordinates=None, arcs=False, class_name="BoundRingChartFlowProposal",
):
    """A :class:`RingChartFlowProposal` subclass with the network's geometry
    bound, for nessai's ``flow_proposal_class``: nessai (and bilby's nessai
    plugin) pass a proposal only the keyword arguments nessai knows.

    The class is published at module scope as ``class_name`` so checkpoints
    unpickle: build it again (with the same geometry) before resuming, as
    for :func:`nessai_gw.group_mixture` 's proposal classes.

    Parameters
    ----------
    tensors, vertices, reference_time, phase_sign, pole_regularisation, arcs
        As for :class:`RingChart`.
    coordinates : dict, optional
        As for :class:`ChartFlowProposal`.
    class_name : str
        Name of the class at module scope.
    """
    geometry = dict(
        tensors=np.asarray(tensors, dtype=float).tolist(),
        vertices=np.asarray(vertices, dtype=float).tolist(),
        reference_time=float(reference_time),
        phase_sign=phase_sign,
        pole_regularisation=float(pole_regularisation),
        coordinates=_coordinates(coordinates),
        **({"arcs": True} if arcs else {}),
    )
    return _bound_proposal_class(RingChartFlowProposal, geometry, class_name)

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
    coordinates : dict, optional
        As for :class:`ChartFlowProposal`.
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
        coordinates=None,
    ):
        self.plane_normal = [float(v) for v in plane_normal]
        self.vertex = [float(v) for v in vertex]
        self.reference_time = float(reference_time)
        self.phase_sign = phase_sign
        self.max_abs_cos_theta_jn = float(max_abs_cos_theta_jn)
        self.flow_config = dict(flow_config or DEFAULT_FLOW_CONFIG)
        self.training_config = dict(training_config or DEFAULT_TRAINING_CONFIG)
        self.coordinates = _coordinates(coordinates)

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
            coordinates=getattr(self, "coordinates", None),
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
