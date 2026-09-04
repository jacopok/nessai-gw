"""Small shared geocentric-geometry helpers.

A dependency-light leaf module (numpy only) so both
:mod:`nessai_gw.group_mixture` and
:mod:`nessai_gw.reparameterisations.time` can use the geocenter-to-detector
light-travel geometry without an import cycle through the reparameterisations
package ``__init__``.
"""

from __future__ import annotations

import numpy as np

#: Speed of light in m / s (CODATA / bilby ``speed_of_light``).
SPEED_OF_LIGHT = 299792458.0

_TWO_PI = 2.0 * np.pi


def greenwich_mean_sidereal_time(gps_time: float) -> float:
    """GMST in radians at ``gps_time`` (via bilby if it is available)."""
    try:
        from bilby_cython.time import greenwich_mean_sidereal_time as _gmst

        return float(_gmst(gps_time)) % _TWO_PI
    except Exception:  # pragma: no cover - fallback only
        # IAU 1982 GMST, adequate (<~1 arcmin) for a discrete-mode proposal.
        jd = gps_time / 86400.0 + 2444244.5
        t = (jd - 2451545.0) / 36525.0
        gmst_seconds = (
            67310.54841
            + (876600.0 * 3600.0 + 8640184.812866) * t
            + 0.093104 * t**2
            - 6.2e-6 * t**3
        )
        return (gmst_seconds * np.pi / 43200.0) % _TWO_PI


def geocenter_time_delay(vertex, gmst, ra, dec):
    """Geocenter-to-detector light-travel delay ``t_det - geocent_time``.

    Equal to ``-(n . r_det) / c`` with ``n`` the unit vector towards the
    source, matching bilby's ``time_delay_from_geocenter`` convention.

    Parameters
    ----------
    vertex : array_like
        Earth-fixed detector position in metres, shape ``(3,)``.
    gmst : float
        Greenwich mean sidereal time in radians (see
        :func:`greenwich_mean_sidereal_time`).
    ra, dec : array_like
        Source right ascension and declination in radians.
    """
    vertex = np.asarray(vertex, dtype=float)
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    phi = ra - gmst
    theta = 0.5 * np.pi - dec
    st = np.sin(theta)
    n = np.stack(
        [st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1
    )
    return -(n * vertex).sum(-1) / SPEED_OF_LIGHT
