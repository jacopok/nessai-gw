"""Tests for the Roulet et al. (arXiv:2207.03508) chirp-distance
reparameterisation."""

import numpy as np
import pytest

from nessai_gw._effective_distance import (
    chirp_distance,
    dominant_detector,
    luminosity_distance_from_chirp_distance,
)
from nessai_gw._ellipse import ideal_triangle_tensors
from nessai_gw._geometry import greenwich_mean_sidereal_time
from nessai_gw.group_mixture import ET_EMR_PLANE_NORMAL
from nessai_gw.reparameterisations import (
    ChirpDistanceReparameterisation,
    known_reparameterisations,
)

REFERENCE_TIME = 1187008882.4
FIDUCIAL = dict(ra=0.9694, dec=-1.1491, psi=1.57, theta_jn=1.0472)
NAMES = ("luminosity_distance", "chirp_mass", "theta_jn", "ra", "dec", "psi")


@pytest.fixture
def tensors():
    return ideal_triangle_tensors(ET_EMR_PLANE_NORMAL)


@pytest.fixture
def gmst():
    return greenwich_mean_sidereal_time(REFERENCE_TIME)


@pytest.fixture
def reparam(tensors, gmst):
    return ChirpDistanceReparameterisation(
        parameters="luminosity_distance",
        tensors=tensors,
        gmst=gmst,
        fiducial=FIDUCIAL,
    )


def structured(d_l, mchirp, theta_jn, ra, dec, psi):
    x = np.zeros(len(d_l), dtype=[(n, "f8") for n in NAMES])
    x["luminosity_distance"], x["chirp_mass"] = d_l, mchirp
    x["theta_jn"], x["ra"], x["dec"], x["psi"] = theta_jn, ra, dec, psi
    return x


def random_points(n, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(50.0, 500.0, n),
        rng.uniform(15.0, 60.0, n),
        np.arccos(rng.uniform(-1, 1, n)),
        rng.uniform(0, 2 * np.pi, n),
        np.arcsin(rng.uniform(-1, 1, n)),
        rng.uniform(0, np.pi, n),
    )


def test_registered():
    assert "chirp-distance" in known_reparameterisations


def test_rejects_the_wrong_parameter(tensors, gmst):
    with pytest.raises(RuntimeError, match="luminosity_distance"):
        ChirpDistanceReparameterisation(
            parameters="mass_ratio", tensors=tensors, gmst=gmst,
            fiducial=FIDUCIAL,
        )


def test_requires_k0_or_fiducial(tensors, gmst):
    with pytest.raises(ValueError, match="`k0`.*`fiducial`"):
        ChirpDistanceReparameterisation(
            parameters="luminosity_distance", tensors=tensors, gmst=gmst,
        )


def test_requires_tensors_or_geometry():
    with pytest.raises(ValueError, match="tensors.*gmst.*plane_normal"):
        ChirpDistanceReparameterisation(
            parameters="luminosity_distance", fiducial=FIDUCIAL,
        )


def test_k0_matches_dominant_detector(tensors, gmst, reparam):
    k0, _ = dominant_detector(
        tensors, FIDUCIAL["ra"], FIDUCIAL["dec"], FIDUCIAL["psi"],
        FIDUCIAL["theta_jn"], gmst,
    )
    assert reparam.k0 == k0


def test_explicit_k0_bypasses_fiducial(tensors, gmst):
    r = ChirpDistanceReparameterisation(
        parameters="luminosity_distance", tensors=tensors, gmst=gmst, k0=2,
    )
    assert r.k0 == 2


def test_matches_effective_distance_module(reparam, tensors, gmst):
    """The reparameterisation must agree exactly with the standalone
    :mod:`nessai_gw._effective_distance` functions it wraps."""
    d_l, mchirp, theta_jn, ra, dec, psi = random_points(500, seed=1)
    x = structured(d_l, mchirp, theta_jn, ra, dec, psi)
    x_prime = np.zeros(len(x), dtype=[("chirp_distance", "f8")])
    _, x_prime, log_j = reparam.reparameterise(x, x_prime, np.zeros(len(x)))

    expected = chirp_distance(
        d_l, mchirp, theta_jn, ra, dec, psi, tensors, gmst, reparam.k0,
    )
    np.testing.assert_allclose(x_prime["chirp_distance"], expected, rtol=1e-12)


def test_round_trip(reparam):
    d_l, mchirp, theta_jn, ra, dec, psi = random_points(1000, seed=5)
    x = structured(d_l, mchirp, theta_jn, ra, dec, psi)
    x_prime = np.zeros(len(x), dtype=[("chirp_distance", "f8")])
    log_j = np.zeros(len(x))
    x, x_prime, log_j = reparam.reparameterise(x, x_prime, log_j)

    back = structured(
        np.zeros_like(d_l), mchirp, theta_jn, ra, dec, psi
    )
    log_j_inv = np.zeros(len(x))
    back, _, log_j_inv = reparam.inverse_reparameterise(
        back, x_prime, log_j_inv
    )
    assert np.abs(back["luminosity_distance"] - d_l).max() < 1e-8
    # the inverse Jacobian is minus the forward one at the same point
    assert np.abs(log_j + log_j_inv).max() < 1e-8


def test_round_trip_matches_inverse_function(reparam, tensors, gmst):
    d_l, mchirp, theta_jn, ra, dec, psi = random_points(200, seed=6)
    x = structured(d_l, mchirp, theta_jn, ra, dec, psi)
    x_prime = np.zeros(len(x), dtype=[("chirp_distance", "f8")])
    _, x_prime, _ = reparam.reparameterise(x, x_prime, np.zeros(len(x)))

    expected = luminosity_distance_from_chirp_distance(
        x_prime["chirp_distance"], mchirp, theta_jn, ra, dec, psi, tensors,
        gmst, reparam.k0,
    )
    np.testing.assert_allclose(expected, d_l, rtol=1e-10)


def test_jacobian_matches_numerical_derivative(reparam):
    _, mchirp, theta_jn, ra, dec, psi = random_points(200, seed=7)
    d_l = np.random.default_rng(8).uniform(50.0, 500.0, 200)
    eps = 1e-3

    def prime(d):
        x = structured(d, mchirp, theta_jn, ra, dec, psi)
        xp = np.zeros(len(x), dtype=[("chirp_distance", "f8")])
        _, xp, _ = reparam.reparameterise(x, xp, np.zeros(len(x)))
        return xp["chirp_distance"]

    numerical = np.log(
        np.abs((prime(d_l + eps) - prime(d_l - eps)) / (2 * eps))
    )
    x = structured(d_l, mchirp, theta_jn, ra, dec, psi)
    xp = np.zeros(len(x), dtype=[("chirp_distance", "f8")])
    _, _, log_j = reparam.reparameterise(x, xp, np.zeros(len(x)))
    assert np.abs(log_j - numerical).max() < 1e-6


def test_reduces_inclination_correlation(reparam):
    """The point of the reparameterisation: for a (nearly) fixed sky/psi --
    as in a converged single-site posterior, where the sky is tightly
    constrained -- `luminosity_distance` at fixed SNR (fixed measured
    amplitude `a_k0`) is forced to trace `|R_k0(theta_jn)|`, so it correlates
    strongly with `cos(theta_jn)`. `chirp_distance` divides that factor back
    out and should not."""
    rng = np.random.default_rng(42)
    n = 5000
    # |R_k0| is an even function of cos(theta_jn) (Eq. 9 only involves
    # cos^2(iota) and cos(iota) squared terms), so it is only correlated
    # with cos(theta_jn) itself -- rather than |cos(theta_jn)| -- once the
    # posterior support breaks the theta_jn <-> pi - theta_jn symmetry, as
    # real (single-mode) live points do; restrict to one branch here.
    theta_jn = rng.uniform(0.2, 1.3, n)
    ra = np.full(n, FIDUCIAL["ra"])
    dec = np.full(n, FIDUCIAL["dec"])
    psi = np.full(n, FIDUCIAL["psi"])
    mchirp = np.full(n, 28.0)
    # Distance chosen so that the *measured* amplitude a_k0 is exactly
    # constant, i.e. exactly the degeneracy the coordinate is designed to
    # remove.
    r_k0 = reparam._abs_r(
        structured(np.zeros(n), mchirp, theta_jn, ra, dec, psi)
    )
    d_l = 100.0 * mchirp ** (5.0 / 6.0) * r_k0

    x = structured(d_l, mchirp, theta_jn, ra, dec, psi)
    x_prime = np.zeros(n, dtype=[("chirp_distance", "f8")])
    _, x_prime, _ = reparam.reparameterise(x, x_prime, np.zeros(n))

    corr_before = np.corrcoef(np.cos(theta_jn), d_l)[0, 1]
    corr_after = np.corrcoef(np.cos(theta_jn), x_prime["chirp_distance"])[0, 1]
    assert np.abs(corr_before) > 0.5
    assert x_prime["chirp_distance"].std() < 1e-6
    assert np.abs(corr_after) < np.abs(corr_before)
