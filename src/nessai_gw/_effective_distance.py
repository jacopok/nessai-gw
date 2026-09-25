"""The chirp/effective-distance coordinate for a single triangular detector.

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

This module supplies exactly the antenna-pattern piece, reusing
:mod:`nessai_gw._ellipse`'s idealised-triangle machinery (the same
``Z_k = F_+,k + i F_x,k`` at ``psi = 0`` that :class:`PolarisationEllipse`
already builds from ``ideal_triangle_tensors``/``_complex_response``, and
which the group action's own prime coordinates are built against -- see
:mod:`nessai_gw._ellipse`'s module docstring on the ~1e-3 accuracy of the
idealised tensors against the real ET-EMR geometry). It does not depend on
:mod:`nessai_gw._ellipse` at runtime beyond that shared geometry, and adds no
new detector model.

Choosing the reference detector
--------------------------------
For a *network* of distinct sites (:mod:`nessai_gw.network_group`), the
reference detector is chosen by passing per-detector SNRs
(``DetectorNetworkGeometry.from_interferometers``, reading
``meta_data["optimal_SNR"]``) and sorting.  The three ET-EMR sub-detectors
are co-located and share one PSD (verified on the ET-Delta runs: the three
``power_spectral_density_array``s are bit-identical), so by Eq. 11 their SNR
ratio at any fixed ``(d_L, M)`` is exactly the ratio of their |R_k| -- the
PSD- and mass/distance-dependent factors cancel.  :func:`dominant_detector`
therefore picks ``k0 = argmax_k |R_k|`` at a single fiducial point (the
injection, or the maximum-likelihood point) directly, which is equivalent to
or `DetectorNetworkGeometry`'s: pass real SNRs and sort, without a costly
per-candidate matched-filter integral.
"""

from __future__ import annotations

import numpy as np

from ._ellipse import _complex_response, ideal_triangle_tensors

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
        ``(n_det, 3, 3)`` detector tensors, e.g. from
        :func:`nessai_gw._ellipse.ideal_triangle_tensors`.
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
