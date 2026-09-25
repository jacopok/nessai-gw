"""The chirp/effective-distance coordinate (Roulet et al. 2022).

Roulet et al. 2022 (arXiv:2207.03508) replace the luminosity distance ``d_L``
by the "chirp distance"

    d_hat(d_L, M, iota, ra, dec, psi) := 1 / a_{k0}
                                        = d_L / (M^{5/6} |R_{k0}|),      (Eq. 18)

with ``a_k = M^{5/6}/d_L * |R_k(iota, n, psi)|`` (Eq. 11) the amplitude at
detector ``k`` and

    R_k(iota, n, psi) = (1 + cos^2 iota)/2 * F_+,k(n, psi)
                         - i cos(iota) * F_x,k(n, psi)                   (Eq. 9)

its dominant-(2,2)-mode response, ``k0`` the loudest ("reference") detector.
``d_hat`` is what the extrinsic likelihood actually measures well (Eqs.
15-18 of the paper): it is d_L stripped of the amplitude scaling that
``iota``, the sky position and ``psi`` impose through the antenna pattern,
which is the dominant visible ``distance`` <-> ``inclination`` correlation
in a single-detector (or single-site) posterior.

This module supplies exactly the antenna-pattern piece.  ``Z_k = F_+,k + i
F_x,k`` at ``psi = 0`` is evaluated with :mod:`nessai_gw._ellipse`'s
``_complex_response`` (bilby's convention, checked against
``Interferometer.antenna_response`` to machine precision), but on the *real*
detector tensors (``Interferometer.detector_tensor``;
:func:`nessai_gw.group_mixture.detector_tensors`,
:data:`nessai_gw.group_mixture.ET_EMR_DETECTOR_TENSORS`), not on
:func:`nessai_gw._ellipse.ideal_triangle_tensors`.  The idealised triangle
fixes the detector *plane* but not the arms' in-plane orientation (``e1`` is
the projected Earth axis, and ``azimuth_offset`` -- a fundamental-domain
seam knob -- rotates it further).  The polarisation ellipse only uses
rotation-invariant sums over all three sub-detectors, so it does not care, but
``|R_k0|`` singles out one sub-detector and does: for ET-EMR the idealised
``|R_k|`` is off by up to ~0.15 (on a typical ``|R| ~ 0.3``) at
``azimuth_offset = 0``.

``|R_k|`` is invariant under every element of the triangular group
(:class:`nessai_gw.group_mixture.TriangularDetectorGroupAction`: the quarter
turn sends ``Z -> -Z``, the plane reflection ``Z -> conj Z`` together with
``cos iota -> -cos iota``) -- exactly for a planar triangle of any in-plane
orientation, to ~1e-3 on the real ET-EMR tensors -- and exactly under the
network polarisation/phase ``Z4`` (``Z -> -Z``).  So the chirp distance is a
group-invariant coordinate and both prime-space group actions pass it through
untouched, as they do ``luminosity_distance``.

Choosing the reference detector
--------------------------------
Roulet et al. sort the detectors by SNR.  For a *network*
(:mod:`nessai_gw.network_group`) the per-detector SNRs are known
(``DetectorNetworkGeometry.from_interferometers`` reads
``meta_data["optimal_SNR"]``), so ``k0`` is simply the loudest detector.  The
three ET-EMR sub-detectors are co-located and share one PSD (verified on the
ET-Delta runs: the three ``power_spectral_density_array``s are
bit-identical), so by Eq. 11 their SNR ratio at any fixed ``(d_L, M)`` is
exactly the ratio of their ``|R_k|`` -- the PSD- and mass/distance-dependent
factors cancel.  :func:`dominant_detector` therefore picks
``k0 = argmax_k |R_k|`` at a single fiducial point (the injection, or the
maximum-likelihood point), which is equivalent to sorting by SNR without a
matched-filter integral.
"""

from __future__ import annotations

import numpy as np

from ._ellipse import _complex_response

__all__ = [
    "response_R",
    "dominant_detector",
    "chirp_distance",
    "luminosity_distance_from_chirp_distance",
]


def response_R(tensors, ra, dec, psi, theta_jn, gmst):
    """``R_k(iota, n, psi)`` (Eq. 9) at every sub-detector ``k`` at once.

    ``theta_jn`` stands in for the inclination ``iota`` -- exact for aligned
    (non-precessing) spins, the same dominant-(2,2)-mode approximation the
    paper itself works in (Sec. II).

    Parameters
    ----------
    tensors : array_like
        ``(n_det, 3, 3)`` Earth-fixed detector tensors, e.g. from
        :func:`nessai_gw.group_mixture.detector_tensors`.
    ra, dec, psi, theta_jn : array_like
        Broadcastable sky position, polarisation angle and inclination.
    gmst : float
        Greenwich mean sidereal time at the (fixed) reference time -- see
        :func:`nessai_gw._geometry.greenwich_mean_sidereal_time`.

    Returns
    -------
    complex ndarray, shape ``(..., n_det)``.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    psi = np.asarray(psi, dtype=float)
    theta_jn = np.asarray(theta_jn, dtype=float)
    z0 = _complex_response(tensors, ra, dec, gmst)  # (..., n_det), psi=0
    z = np.exp(-2j * psi)[..., None] * z0  # bilby convention: Z(psi) = e^{-2i psi} Z(0)
    fp, fc = z.real, z.imag
    c = np.cos(theta_jn)[..., None]
    return 0.5 * (1.0 + c**2) * fp - 1j * c * fc


def dominant_detector(tensors, ra, dec, psi, theta_jn, gmst, names=None):
    """The loudest sub-detector ``k0`` at a single fiducial point.

    Per the module docstring, ``SNR_k`` is monotonic in ``|R_k|`` alone when
    every candidate ``k`` shares one PSD (true for the co-located ET-EMR
    sub-interferometers), so ``k0 = argmax_k |R_k(fiducial)|`` is exactly
    Roulet et al.'s "sort detectors by SNR" convention.

    Returns
    -------
    k0 : int
    label : the corresponding entry of ``names`` if given, else ``k0``.
    """
    R = response_R(tensors, np.atleast_1d(ra), np.atleast_1d(dec),
                    np.atleast_1d(psi), np.atleast_1d(theta_jn), gmst)[0]
    k0 = int(np.argmax(np.abs(R)))
    return k0, (names[k0] if names is not None else k0)


def chirp_distance(luminosity_distance, chirp_mass, theta_jn, ra, dec, psi,
                    tensors, gmst, k0):
    """``d_hat`` (Eq. 18) at the reference detector ``k0``.

    All of ``luminosity_distance, chirp_mass, theta_jn, ra, dec, psi`` may be
    arrays of the same shape (e.g. one row per live point); everything else
    is broadcast against them.
    """
    R = response_R(tensors, ra, dec, psi, theta_jn, gmst)[..., k0]
    chirp_mass = np.asarray(chirp_mass, dtype=float)
    luminosity_distance = np.asarray(luminosity_distance, dtype=float)
    return luminosity_distance / (chirp_mass ** (5.0 / 6.0) * np.abs(R))


def luminosity_distance_from_chirp_distance(d_hat, chirp_mass, theta_jn, ra,
                                             dec, psi, tensors, gmst, k0):
    """Inverse of :func:`chirp_distance`: ``d_L = d_hat * M^{5/6} |R_k0|``."""
    R = response_R(tensors, ra, dec, psi, theta_jn, gmst)[..., k0]
    chirp_mass = np.asarray(chirp_mass, dtype=float)
    d_hat = np.asarray(d_hat, dtype=float)
    return d_hat * chirp_mass ** (5.0 / 6.0) * np.abs(R)
