"""Tests for the effective-tidal-deformability and Doppler chirp-mass reparameterisations."""

import warnings

import numpy as np
import pytest
from scipy import stats

from nessai_gw.group_mixture import (
    _prime_parameter_names,
    triangular_group_reparameterisations,
)
from nessai_gw.reparameterisations import (
    DopplerChirpMassReparameterisation,
    EffectiveTidalDeformabilityReparameterisation,
    known_reparameterisations,
)
from nessai_gw.reparameterisations.mass import doppler_factor
from nessai_gw.reparameterisations.tides import (
    lambda_tilde,
    lambda_tilde_ratio,
)

L1, L2 = (5.0, 2000.0), (0.0, 1500.0)
NAMES = ("lambda_1", "lambda_2", "mass_ratio", "chirp_mass", "ra", "dec")
#: roughly the Earth's orbital velocity over c
K = np.array([-6.1e-5, 7.3e-5, 3.2e-5])
MC = (1.183, 1.184)


def _points(n, seed=0, q=(0.3, 1.0)):
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=[(name, "f8") for name in NAMES])
    x["lambda_1"] = rng.uniform(*L1, n)
    x["lambda_2"] = rng.uniform(*L2, n)
    x["mass_ratio"] = rng.uniform(*q, n)
    x["chirp_mass"] = rng.uniform(*MC, n)
    x["ra"] = rng.uniform(0, 2 * np.pi, n)
    x["dec"] = np.arcsin(rng.uniform(-1, 1, n))
    return x


def _prime(reparam, n):
    return np.zeros(n, dtype=[(p, "f8") for p in reparam.prime_parameters])


@pytest.fixture
def tides():
    return EffectiveTidalDeformabilityReparameterisation(
        parameters=["lambda_1", "lambda_2"],
        prior_bounds={"lambda_1": L1, "lambda_2": L2},
    )


@pytest.fixture
def doppler():
    return DopplerChirpMassReparameterisation(
        parameters="chirp_mass", prior_bounds={"chirp_mass": MC},
        doppler_vector=K,
    )


def test_registered():
    assert "effective-tidal-deformability" in known_reparameterisations
    assert "doppler-chirp-mass" in known_reparameterisations


# ---------------------------------------------------------------------------
# tides
# ---------------------------------------------------------------------------
def test_lambda_tilde_equal_mass():
    """At q = 1, lambda_tilde is the mean deformability."""
    np.testing.assert_allclose(lambda_tilde(300.0, 500.0, 1.0), 400.0)
    np.testing.assert_allclose(lambda_tilde_ratio(1.0), 1.0)


def test_tides_init(tides):
    assert tides.prime_parameters == ["lambda_tilde_prime", "lambda_diff_prime"]
    assert "mass_ratio" in tides.inverse_input_parameters
    with pytest.raises(RuntimeError, match="exactly two"):
        EffectiveTidalDeformabilityReparameterisation(
            parameters=["lambda_1"], prior_bounds={"lambda_1": L1}
        )


def test_tides_u_is_a_function_of_lambda_tilde(tides):
    """At fixed q, lambda_tilde_prime is monotonic in lambda_tilde alone."""
    x = _points(2000, seed=1, q=(0.8, 0.8))
    xp = _prime(tides, len(x))
    tides.reparameterise(x, xp, np.zeros(len(x)))
    lt = lambda_tilde(x["lambda_1"], x["lambda_2"], x["mass_ratio"])
    order = np.argsort(lt)
    assert np.all(np.diff(xp["lambda_tilde_prime"][order]) >= -1e-12)


def test_tides_prior_maps_to_standard_normal(tides):
    x = _points(20000, seed=2)
    xp = _prime(tides, len(x))
    tides.reparameterise(x, xp, np.zeros(len(x)))
    for name in tides.prime_parameters:
        assert stats.kstest(xp[name], "norm").pvalue > 1e-3
    assert abs(np.corrcoef(xp["lambda_tilde_prime"],
                           xp["lambda_diff_prime"])[0, 1]) < 0.03


def test_tides_round_trip(tides):
    x = _points(5000, seed=3)
    xp = _prime(tides, len(x))
    _, xp, lj = tides.reparameterise(x.copy(), xp, np.zeros(len(x)))
    back = x.copy()
    back["lambda_1"] = back["lambda_2"] = 0.0
    back, _, lj_inv = tides.inverse_reparameterise(back, xp, np.zeros(len(x)))
    np.testing.assert_allclose(back["lambda_1"], x["lambda_1"], rtol=1e-9,
                               atol=1e-8)
    np.testing.assert_allclose(back["lambda_2"], x["lambda_2"], rtol=1e-9,
                               atol=1e-7)
    np.testing.assert_allclose(lj, -lj_inv, atol=1e-10)


def test_tides_round_trip_in_the_tails(tides):
    u = np.array([-7.0, -5.0, 0.0, 0.0, 5.0, 7.0, -1.0, 1.0])
    w = np.array([0.0, 6.0, -7.0, 7.0, -6.0, 0.0, 0.5, -0.5])
    x = _points(len(u), seed=4)
    x["mass_ratio"] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5]
    xp = _prime(tides, len(u))
    xp["lambda_tilde_prime"], xp["lambda_diff_prime"] = u, w
    x, _, _ = tides.inverse_reparameterise(x, xp, np.zeros(len(u)))
    assert np.all((x["lambda_1"] >= L1[0]) & (x["lambda_1"] <= L1[1]))
    assert np.all((x["lambda_2"] >= L2[0]) & (x["lambda_2"] <= L2[1]))
    xp2 = _prime(tides, len(u))
    tides.reparameterise(x, xp2, np.zeros(len(u)))
    np.testing.assert_allclose(xp2["lambda_tilde_prime"], u, atol=1e-6)
    # |w| = 7 puts lambda_1 within ~1e-9 of a prior edge, which float64
    # holds only to ~1e-13 of lambda_1: ~1e-5 in w (prior mass ~1e-12)
    np.testing.assert_allclose(xp2["lambda_diff_prime"], w, atol=3e-5)


def test_tides_jacobian_matches_finite_difference(tides):
    x = _points(300, seed=5)
    h = 1e-6

    def uw(d1, d2):
        y = x.copy()
        y["lambda_1"] += d1
        y["lambda_2"] += d2
        yp = _prime(tides, len(y))
        tides.reparameterise(y, yp, np.zeros(len(y)))
        return np.stack([yp[p] for p in tides.prime_parameters], axis=-1)

    jac = np.stack(
        [(uw(h, 0) - uw(-h, 0)) / (2 * h), (uw(0, h) - uw(0, -h)) / (2 * h)],
        axis=-1,
    )
    xp = _prime(tides, len(x))
    _, _, lj = tides.reparameterise(x.copy(), xp, np.zeros(len(x)))
    np.testing.assert_allclose(
        np.log(np.abs(np.linalg.det(jac))), lj, atol=1e-5
    )


def test_tides_inverse_skips_invalid_mass_ratio(tides):
    x = _points(6, seed=6)
    xp = _prime(tides, 6)
    tides.reparameterise(x.copy(), xp, np.zeros(6))
    x["mass_ratio"][:3] = [0.0, 1e-310, np.nan]
    with np.errstate(all="raise"):
        x, _, lj = tides.inverse_reparameterise(x, xp, np.zeros(6))
    assert np.all(np.isnan(x["lambda_1"][:3]))
    assert np.all(np.isnan(lj[:3]))
    assert np.all(np.isfinite(lj[3:]))


# ---------------------------------------------------------------------------
# Doppler chirp mass
# ---------------------------------------------------------------------------
def test_doppler_factor_matches_bilby_xg():
    motion = pytest.importorskip("bilby_xG.motion")
    t = 1187008882.4
    _, v = motion.earth_barycentric_position_velocity(t)
    k = motion.precession_matrix(t) @ v / motion.speed_of_light
    for ra, dec in [(0.3, -0.4), (3.4462, -0.4081), (5.0, 1.2)]:
        np.testing.assert_allclose(
            doppler_factor(ra, dec, k), motion.doppler_factor(ra, dec, t),
            rtol=1e-15,
        )


def test_doppler_init_errors():
    with pytest.raises(RuntimeError, match="needs doppler_vector"):
        DopplerChirpMassReparameterisation(
            parameters="chirp_mass", prior_bounds={"chirp_mass": MC}
        )
    with pytest.raises(RuntimeError, match="not v/c"):
        DopplerChirpMassReparameterisation(
            parameters="chirp_mass", prior_bounds={"chirp_mass": MC},
            doppler_vector=[3e4, 0, 0],
        )


def test_doppler_round_trip_and_jacobian(doppler):
    x = _points(2000, seed=7)
    doppler.update(x)
    xp = _prime(doppler, len(x))
    _, xp, lj = doppler.reparameterise(x.copy(), xp, np.zeros(len(x)))
    assert np.all(np.abs(xp["chirp_mass_prime"]) <= 1.0)
    back = x.copy()
    back["chirp_mass"] = 0.0
    back, _, lj_inv = doppler.inverse_reparameterise(back, xp, np.zeros(len(x)))
    np.testing.assert_allclose(back["chirp_mass"], x["chirp_mass"], rtol=1e-14)
    np.testing.assert_allclose(lj, -lj_inv, atol=1e-12)
    # d chirp_mass_prime / d chirp_mass
    h = 1e-9
    y = x.copy()
    y["chirp_mass"] += h
    yp = _prime(doppler, len(y))
    doppler.reparameterise(y, yp, np.zeros(len(y)))
    np.testing.assert_allclose(
        np.log((yp["chirp_mass_prime"] - xp["chirp_mass_prime"]) / h), lj,
        atol=1e-5,
    )


def test_doppler_removes_the_sky_dependence(doppler):
    """A barycentric chirp mass M / D(sky) maps to one prime value."""
    x = _points(500, seed=8)
    x["chirp_mass"] = 1.1835 / doppler_factor(x["ra"], x["dec"], K)
    xp = _prime(doppler, len(x))
    doppler.reparameterise(x, xp, np.zeros(len(x)))
    assert np.ptp(xp["chirp_mass_prime"]) < 1e-9


def test_doppler_bounds_reset(doppler):
    initial = doppler.bounds
    assert initial[0] < MC[0] and initial[1] > MC[1]
    doppler.update(_points(100, seed=9))
    assert doppler.bounds != initial
    doppler.reset()
    assert doppler.bounds == initial


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------
WIRING = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "lambda_1",
          "lambda_2", "luminosity_distance", "theta_jn", "psi", "phase", "ra",
          "dec", "geocent_time"]
T_REF = 1187008882.4


def test_triangular_wiring():
    reps = triangular_group_reparameterisations(
        WIRING, T_REF, effective_spin=True, effective_tidal_deformability=True,
        doppler_vector=K,
    )
    assert reps["lambda_1"] == {
        "reparameterisation": "effective-tidal-deformability",
        "parameters": ["lambda_1", "lambda_2"],
    }
    assert "lambda_2" not in reps
    assert reps["chirp_mass"]["reparameterisation"] == "doppler-chirp-mass"
    np.testing.assert_array_equal(reps["chirp_mass"]["doppler_vector"], K)
    default = triangular_group_reparameterisations(WIRING, T_REF)
    assert default["lambda_1"]["reparameterisation"] == "logit"
    assert "chirp_mass" not in default


def test_wiring_errors():
    with pytest.raises(ValueError, match="mass_ratio"):
        triangular_group_reparameterisations(
            [n for n in WIRING if n != "mass_ratio"], T_REF,
            effective_tidal_deformability=True,
        )
    with pytest.raises(ValueError, match="'ra' and 'dec'"):
        triangular_group_reparameterisations(
            [n for n in WIRING if n not in ("ra", "dec")], T_REF,
            doppler_vector=K,
        )


@pytest.mark.parametrize("sky_2d", [False, True])
def test_prime_names_from_nessai(sky_2d):
    """nessai accepts the order (the probe builds the real
    ``CombinedReparameterisation``, whose ``check_order`` checks every
    inverse input) and reports the new prime names."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        prime = _prime_parameter_names(
            WIRING, T_REF, sky_2d=sky_2d, effective_spin=True,
            effective_tidal_deformability=True, doppler_vector=K,
        )
    assert {"lambda_tilde_prime", "lambda_diff_prime",
            "chirp_mass_prime"} <= set(prime)
    assert "lambda_1_prime" not in prime
