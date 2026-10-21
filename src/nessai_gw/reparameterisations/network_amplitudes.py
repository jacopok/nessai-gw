"""Two detectors' complex (2, 2)-mode amplitudes as flow coordinates.

The two detectors are the two single-interferometer sites of a network (ET
2L), or the two signal channels A and E of a triangle (ET Delta, see
:func:`triangle_channel_tensors`); the notes below are written for two
sites, and the triangle's are at the end.

With two separated sites the arrival-time difference fixes the angle
``alpha`` between the source and the baseline, and the two complex
amplitudes the sites measure fix four of the remaining five extrinsic
parameters: at *any* azimuth ``beta`` about the baseline (``sky_u``) there is
a ``(d_L, iota, psi, phase)`` that fits the data, so the posterior is a ring
in ``beta`` whose other coordinates curve along it.  This reparameterisation
trades ``(luminosity_distance, theta_jn, psi, phase)`` for what the data
actually measure, the amplitudes

    Z_k = R_k(iota, n, psi) e^{2 i phase} chirp_mass^{5/6} / d_L,   k = 1, 2,

with ``R_k = (1 + cos^2 iota)/2 F_+,k - i cos iota F_x,k`` the dominant-mode
response (:func:`nessai_gw._effective_distance.response_R`), carried as
``(ln_amp_k, arg_amp_k) = (ln |Z_k|, arg Z_k)``.  Along the ring the ``Z_k``
stay put, so the flow sees the ring as the single coordinate ``sky_u`` and
a compact blob in the others.

Inverse
-------
Write ``F_+,k + i F_x,k = W_k e^{-2 i psi}`` (bilby's convention, ``W_k``
the response at ``psi = 0``) and ``c = cos iota``,
``D = d_L / chirp_mass^{5/6}``.  Then

    Z_k = W_k L + conj(W_k) R,
    L = (1 - c)^2 e^{2 i (phase - psi)} / (4 D),
    R = (1 + c)^2 e^{2 i (phase + psi)} / (4 D),

so at fixed sky ``(L, R) = M^{-1} (Z_1, Z_2)`` with ``M = [[W_1, conj W_1],
[W_2, conj W_2]]``, ``det M = 2 i Im(W_1 conj W_2)``.  ``|L| / |R| =
((1 - c) / (1 + c))^2 = tan^4(iota / 2)`` has a solution ``iota in [0, pi)``
for every ratio, then ``D = (1 + c)^2 / (4 |R|)``; the arguments give ``phase -+ psi`` modulo
``pi``, i.e. ``(phase, psi)`` up to the four elements ``(phase + m pi/2,
psi + m pi/2)``, ``m = 0..3``: the polarisation/phase quarter turn of
:mod:`nessai_gw.network_group`.  The map is therefore 4-to-1 and the
coordinates fold that group by themselves (use the network proposal with
``network_amplitudes``, which drops it); the inverse picks one of the four
branches uniformly at random, so the proposal density in the physical
space is ``q'(x') |J| / 4`` (the constant is left out of ``log_j``).

The map is singular where ``Im(W_1 conj W_2) = 0`` -- the sky positions
where the two sites see the same polarisation combination -- and at
``iota = 0, pi`` (``L`` or ``R`` vanishes: the usual ``phase +- psi``
degeneracy).  Both are measure zero, but near face-on (off) ``L`` (``R``)
goes as ``iota^4 / 4``: within ~0.01 rad of ``iota = 0, pi`` the inverse
recovers ``iota`` only to ~1e-7 rad, and below ~1e-3 rad not at all in double
precision (the data barely depend on it there either; that region holds
~1e-6 of the prior).

Jacobian
--------
``|d(ln|Z_1|, arg Z_1, ln|Z_2|, arg Z_2) / d(d_L, theta_jn, psi, phase)|``
is the product of ``chirp_mass^{-5/6}`` (``d_L -> D``), ``sin theta_jn``
(``theta_jn -> c``), ``(1 - c^2) / (4 D^3)`` (``(D, c) -> (|L|, |R|)``),
``8`` (``(psi, phase) -> (arg L, arg R)``), ``|L| |R|`` (polar to Cartesian),
``|det M|^2`` (a complex-linear map of ``C^2``) and ``1 / (|Z_1|^2 |Z_2|^2)``
(Cartesian to log-polar).

Triangle
--------
The three interferometers of a planar equilateral triangle have tensors with
``D_1 + D_2 + D_3 = 0``: the third channel combination (T) is null and the
signal lives in two, ``D_A = (2 D_1 - D_2 - D_3) / 3`` and ``D_E = (D_3 -
D_2) / sqrt(3)``, orthogonal with the norm of a single arm pair.  With these
as the two detectors everything above holds at every sky position off the
detector plane, where ``Im(W_A conj W_E)`` vanishes (the inverse loses
digits within a few hundredths of a radian of it, as near face-on).  The triangle's data are
also invariant under the eight sky elements of
:class:`~nessai_gw.group_mixture.TriangularDetectorGroupAction` (the quarter
turns about the plane normal and the reflection through the plane, with
``psi`` and ``iota`` carried along), so the ``Z`` are too: those still act,
on the sky alone (:class:`~nessai_gw.group_mixture.PrimeSpaceSkyGroupAction`).
Both statements are exact for a planar equilateral triangle; for the
``bilby_xG`` ET tensors (null channel 0.2% of an interferometer's) the
images' amplitudes differ by ~1e-3, as the 32-element group's images do.

Phase at a reference frequency
------------------------------
With ``waveform_phase`` the phase is shifted by the intrinsic-parameter
offset of :class:`~nessai_gw.reparameterisations.PolarisationPhaseReparameterisation`
(refitted at every :meth:`update`), which unwinds the arguments' helix around
the masses and spins; the shift depends only on the intrinsic parameters, so
the Jacobian is unchanged.
"""

from __future__ import annotations

import inspect

import numpy as np
from nessai.reparameterisations import Reparameterisation

from .. import nessai_logger
from .._ellipse import _complex_response
from .._geometry import greenwich_mean_sidereal_time
from .phase import PolarisationPhaseReparameterisation

logger = nessai_logger.getChild(__name__)

_TWO_PI = 2.0 * np.pi

#: Physical parameters the coordinates replace, in this order.
NETWORK_AMPLITUDE_PARAMETERS = ["luminosity_distance", "theta_jn", "psi", "phase"]

#: The flow coordinates; the ``arg_amp_*`` are angles in ``[-pi, pi)``.
NETWORK_AMPLITUDE_PRIME_PARAMETERS = [
    "ln_amp_1", "arg_amp_1", "ln_amp_2", "arg_amp_2",
]

#: The circular coordinates among them.
NETWORK_AMPLITUDE_ANGLES = ["arg_amp_1", "arg_amp_2"]


def triangle_channel_tensors(tensors):
    """``(2, 3, 3)`` tensors of the A and E channels of a triangle.

    ``tensors`` are the three interferometers' ``(3, 3, 3)``; ``D_A = (2 D_1
    - D_2 - D_3) / 3`` and ``D_E = (D_3 - D_2) / sqrt(3)``.  For a planar
    equilateral triangle they are orthogonal, with the norm of each
    interferometer's, and the null channel ``D_1 + D_2 + D_3`` vanishes.
    """
    d1, d2, d3 = np.asarray(tensors, dtype=float)
    return np.stack([(2 * d1 - d2 - d3) / 3, (d3 - d2) / np.sqrt(3.0)])


class NetworkAmplitudesReparameterisation(Reparameterisation):
    """``(luminosity_distance, theta_jn, psi, phase) -> (ln_amp_k, arg_amp_k)``.

    Parameters
    ----------
    parameters : list of str
        Must be ``["luminosity_distance", "theta_jn", "psi", "phase"]``.
    prior_bounds : optional
        Accepted for registry compatibility; unused.
    tensors : array_like
        ``(2, 3, 3)`` Earth-fixed tensors of one detector at each of the two
        sites, or ``(3, 3, 3)``: a triangle's three interferometers, whose A
        and E channels (:func:`triangle_channel_tensors`) are then used.
    gmst, reference_time : float, optional
        Greenwich mean sidereal time (rad), or the geocentric GPS time it is
        computed from.  Exactly one is needed.
    waveform_phase, frequency_range, correction_degree, spins, tides : optional
        Measure the phase at a reference frequency, as
        :class:`~nessai_gw.reparameterisations.PolarisationPhaseReparameterisation`.
    chirp_mass, mass_ratio : str, optional
        Names of the chirp mass and mass ratio.
    """

    one_to_one = False

    def __init__(
        self,
        parameters=None,
        prior_bounds=None,
        tensors=None,
        gmst=None,
        reference_time=None,
        waveform_phase=None,
        frequency_range=(8.0, 300.0),
        correction_degree=4,
        chirp_mass="chirp_mass",
        mass_ratio="mass_ratio",
        spins=("chi_1", "chi_2"),
        tides=("lambda_1", "lambda_2"),
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
        if getattr(self, "rng", None) is None:
            self.rng = np.random.default_rng(rng)

        if self.parameters != NETWORK_AMPLITUDE_PARAMETERS:
            raise RuntimeError(
                "NetworkAmplitudesReparameterisation must act on "
                f"{NETWORK_AMPLITUDE_PARAMETERS}; got {self.parameters}"
            )
        if tensors is None:
            raise ValueError(
                "NetworkAmplitudesReparameterisation requires the detector "
                "`tensors` of the two sites."
            )
        tensors = np.asarray(tensors, dtype=float)
        if tensors.shape == (3, 3, 3):
            tensors = triangle_channel_tensors(tensors)
        if tensors.shape != (2, 3, 3):
            raise ValueError(
                "`tensors` must have shape (2, 3, 3), one detector per site, "
                f"or (3, 3, 3), a triangle; got {tensors.shape}"
            )
        if (gmst is None) == (reference_time is None):
            raise ValueError(
                "NetworkAmplitudesReparameterisation requires exactly one of "
                "`gmst` and `reference_time`."
            )
        if gmst is None:
            gmst = greenwich_mean_sidereal_time(reference_time)
        self.tensors = tensors
        self.gmst = float(gmst)
        self._chirp_mass = chirp_mass

        # the reference-frequency phase offset, reused as is
        self._phase = PolarisationPhaseReparameterisation(
            parameters=["phase"],
            waveform_phase=waveform_phase,
            frequency_range=frequency_range,
            correction_degree=correction_degree,
            chirp_mass=chirp_mass,
            mass_ratio=mass_ratio,
            spins=spins,
            tides=tides,
        )
        self.requires = list(dict.fromkeys(
            ["ra", "dec", chirp_mass]
            + [r for r in self._phase.requires if r not in ("psi", "theta_jn")]
        ))

        self.prime_parameters = list(NETWORK_AMPLITUDE_PRIME_PARAMETERS)
        if hasattr(self, "output_parameters"):
            self.output_parameters = list(self.prime_parameters)
        if hasattr(self, "inverse_input_parameters"):
            self.inverse_input_parameters = list(
                dict.fromkeys(
                    list(self.inverse_input_parameters or []) + self.requires
                )
            )

    # ------------------------------------------------------------------
    def set_gmst(self, gmst):
        """Evaluate the responses at the Earth's orientation ``gmst``."""
        self.gmst = float(gmst)

    @property
    def reference_frequency(self):
        """Frequency (Hz) the phase is measured at; None: the merger."""
        return self._phase.reference_frequency

    def _w(self, x):
        """``W_k = F_+,k + i F_x,k`` at ``psi = 0``, shape ``(N, 2)``."""
        return _complex_response(self.tensors, x["ra"], x["dec"], self.gmst)

    def _scale(self, x):
        """``chirp_mass^{5/6}``: ``D = d_L / scale``."""
        return np.asarray(x[self._chirp_mass], dtype=float) ** (5.0 / 6.0)

    def amplitudes(self, x):
        """The complex amplitudes ``Z_k``, shape ``(N, 2)``."""
        w = self._w(x)
        c = np.cos(x["theta_jn"])
        d = x["luminosity_distance"] / self._scale(x)
        phase = x["phase"] + self._phase._shift(x)
        e = np.exp(2j * phase)
        lam = ((1 - c) ** 2 * np.exp(-2j * x["psi"]) * e / (4 * d))[:, None]
        rho = ((1 + c) ** 2 * np.exp(2j * x["psi"]) * e / (4 * d))[:, None]
        return w * lam + np.conj(w) * rho

    def _log_jacobian(self, x, z):
        """``ln |d x' / d x|`` (see the module notes)."""
        w = self._w(x)
        det = 2.0 * np.imag(w[:, 0] * np.conj(w[:, 1]))
        s = np.sin(x["theta_jn"])
        c = np.cos(x["theta_jn"])
        scale = self._scale(x)
        d = x["luminosity_distance"] / scale
        abs_l = (1 - c) ** 2 / (4 * d)
        abs_r = (1 + c) ** 2 / (4 * d)
        return (
            -np.log(scale) + np.log(np.abs(s))
            + np.log((1 - c ** 2) / (4 * d ** 3)) + np.log(8.0)
            + np.log(abs_l * abs_r) + 2 * np.log(np.abs(det))
            - 2 * np.log(np.abs(z[:, 0])) - 2 * np.log(np.abs(z[:, 1]))
        )

    def reparameterise(self, x, x_prime, log_j, **kwargs):
        z = self.amplitudes(x)
        for k in (0, 1):
            x_prime[f"ln_amp_{k + 1}"] = np.log(np.abs(z[:, k]))
            x_prime[f"arg_amp_{k + 1}"] = np.angle(z[:, k])
        return x, x_prime, log_j + self._log_jacobian(x, z)

    def inverse_reparameterise(self, x, x_prime, log_j, **kwargs):
        z = np.stack([
            np.exp(x_prime[f"ln_amp_{k}"] + 1j * x_prime[f"arg_amp_{k}"])
            for k in (1, 2)
        ], axis=-1)
        w = self._w(x)
        det = w[:, 0] * np.conj(w[:, 1]) - np.conj(w[:, 0]) * w[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (np.conj(w[:, 1]) * z[:, 0] - np.conj(w[:, 0]) * z[:, 1]) / det
            rho = (w[:, 0] * z[:, 1] - w[:, 1] * z[:, 0]) / det
            # t = tan^2(theta_jn / 2) = (1 - c) / (1 + c)
            t = np.sqrt(np.abs(lam) / np.abs(rho))
        theta_jn = 2 * np.arctan(np.sqrt(t))
        d = (1 + np.cos(theta_jn)) ** 2 / (4 * np.abs(rho))
        # arg L = 2 (phase - psi), arg R = 2 (phase + psi), each mod 2 pi:
        # one of the four branches (phase, psi) + m (pi/2, pi/2)
        minus = 0.5 * np.angle(lam)
        plus = 0.5 * np.angle(rho)
        m = self.rng.integers(0, 4, size=len(z))
        phase = 0.5 * (plus + minus) + 0.5 * np.pi * m
        psi = 0.5 * (plus - minus) + 0.5 * np.pi * m
        x["theta_jn"] = theta_jn
        x["psi"] = np.mod(psi, np.pi)
        x["luminosity_distance"] = d * self._scale(x)
        x["phase"] = np.mod(phase - self._phase._shift(x), _TWO_PI)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_j = log_j - self._log_jacobian(x, z)
        return x, x_prime, log_j

    def update(self, x, x_prime=None):
        """Refit the reference-frequency phase offset on ``x``."""
        self._phase.update(x, x_prime)
