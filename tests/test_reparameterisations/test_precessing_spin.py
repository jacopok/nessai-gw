"""Tests for the precessing-spin reparameterisation: an exact bijection
whose prior is approximately N(0, I_6)."""

import numpy as np
import pytest
from scipy import stats

from nessai_gw.reparameterisations import PrecessingSpinReparameterisation
from nessai_gw.reparameterisations.spin import (
    _in_plane_forward,
    _in_plane_inverse,
)

A1, A2 = 0.05, 0.04
NAMES = ["a_1", "tilt_1", "a_2", "tilt_2", "phi_12", "phi_jl"]
PRIMES = ["chi_eff_prime", "chi_diff_prime", "s1_x", "s1_y", "s2_x", "s2_y"]


@pytest.fixture(scope="module")
def reparam():
    bounds = {
        "a_1": [0, A1], "a_2": [0, A2], "tilt_1": [0, np.pi],
        "tilt_2": [0, np.pi], "phi_12": [0, 2 * np.pi],
        "phi_jl": [0, 2 * np.pi],
    }
    return PrecessingSpinReparameterisation(parameters=NAMES, prior_bounds=bounds)


def _prior(n, seed=0, q_min=0.3):
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=[(k, "f8") for k in NAMES + ["mass_ratio"]])
    x["a_1"] = rng.uniform(0, A1, n)
    x["a_2"] = rng.uniform(0, A2, n)
    x["tilt_1"] = np.arccos(rng.uniform(-1, 1, n))
    x["tilt_2"] = np.arccos(rng.uniform(-1, 1, n))
    x["phi_12"] = rng.uniform(0, 2 * np.pi, n)
    x["phi_jl"] = rng.uniform(0, 2 * np.pi, n)
    x["mass_ratio"] = rng.uniform(q_min, 1.0, n)
    return x


def _forward(reparam, x):
    x_prime = np.zeros(len(x), dtype=[(k, "f8") for k in PRIMES])
    x, x_prime, log_j = reparam.reparameterise(x, x_prime, np.zeros(len(x)))
    return x_prime, log_j


def _inverse(reparam, x_prime, mass_ratio):
    x = np.zeros(len(x_prime), dtype=[(k, "f8") for k in NAMES + ["mass_ratio"]])
    x["mass_ratio"] = mass_ratio
    x, _, log_j = reparam.inverse_reparameterise(
        x, x_prime, np.zeros(len(x_prime))
    )
    return x, log_j


def test_round_trip(reparam):
    x = _prior(5000, seed=1, q_min=0.05)
    x_prime, log_j = _forward(reparam, x)
    back, log_j_back = _inverse(reparam, x_prime, x["mass_ratio"])
    for name in NAMES:
        np.testing.assert_allclose(back[name], x[name], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(log_j_back, -log_j, atol=1e-7)


def test_in_plane_round_trip_near_the_edges():
    """Accurate as the spin approaches the orbital angular momentum, the
    orbital plane and the magnitude bound."""
    a = np.array([1e-9, 0.01, 0.03, 0.049999, 0.02, 0.02, 0.02, 0.02])
    tilt = np.array([1.0, 1e-7, np.pi - 1e-7, 0.5, np.pi / 2 - 1e-9,
                     np.pi / 2 + 1e-9, 0.3, 2.9])
    rho, log_j = _in_plane_forward(a, tilt, A1)
    a_back, tilt_back, log_j_back = _in_plane_inverse(a * np.cos(tilt), rho, A1)
    np.testing.assert_allclose(a_back, a, rtol=1e-9)
    np.testing.assert_allclose(tilt_back, tilt, rtol=1e-9)
    np.testing.assert_allclose(log_j_back, log_j, rtol=1e-8)


def test_jacobian_is_that_of_the_map(reparam):
    """The in-plane part: finite differences of (chi_z, s_x, s_y) in
    (a, tilt, phi)."""
    rng = np.random.default_rng(3)
    n = 300
    a = rng.uniform(0.002, A1 * 0.98, n)
    tilt = np.arccos(rng.uniform(-0.98, 0.98, n))
    phi = rng.uniform(0, 2 * np.pi, n)

    def f(a, tilt, phi):
        rho, _ = _in_plane_forward(a, tilt, A1)
        return np.stack([a * np.cos(tilt), rho * np.cos(phi), rho * np.sin(phi)], -1)

    h = 1e-7
    jac = np.stack([
        (f(a + h, tilt, phi) - f(a - h, tilt, phi)) / (2 * h),
        (f(a, tilt + h, phi) - f(a, tilt - h, phi)) / (2 * h),
        (f(a, tilt, phi + h) - f(a, tilt, phi - h)) / (2 * h),
    ], axis=-1)
    _, log_j = _in_plane_forward(a, tilt, A1)
    np.testing.assert_allclose(np.log(np.abs(np.linalg.det(jac))), log_j, atol=1e-5)


def test_prior_is_approximately_standard_normal(reparam):
    x = _prior(20000, seed=4)
    x_prime, _ = _forward(reparam, x)
    for name in PRIMES:
        y = x_prime[name]
        assert abs(y.mean()) < 0.05 and abs(y.std() - 1) < 0.05, name
        assert stats.kstest(y, "norm").statistic < 0.02, name
    # the in-plane radii are chi(2)
    for sx, sy in (("s1_x", "s1_y"), ("s2_x", "s2_y")):
        r2 = x_prime[sx] ** 2 + x_prime[sy] ** 2
        assert stats.kstest(r2, "chi2", args=(2,)).statistic < 0.02


def test_log_jacobian_is_the_density_ratio(reparam):
    """log p_prior(x) - log|dx'/dx| = log N(x'; 0, I) but for the tabulated
    effective-spin map's own (documented) departure from normality: the
    in-plane part is exact."""
    x = _prior(2000, seed=5)
    x_prime, log_j = _forward(reparam, x)
    log_prior = (
        -np.log(A1) - np.log(A2)
        + np.log(np.sin(x["tilt_1"]) / 2) + np.log(np.sin(x["tilt_2"]) / 2)
        - 2 * np.log(2 * np.pi)
    )
    log_normal = np.sum(
        [stats.norm.logpdf(x_prime[name]) for name in PRIMES], axis=0
    )
    residual = log_prior - log_j - log_normal

    def log_aligned(chi, a_max):
        return np.log(-np.log(np.abs(chi) / a_max)) - np.log(2 * a_max)

    chi_1z = x["a_1"] * np.cos(x["tilt_1"])
    chi_2z = x["a_2"] * np.cos(x["tilt_2"])
    u, w, lj = reparam._transform.forward(chi_1z, chi_2z, x["mass_ratio"])
    aligned_residual = (
        log_aligned(chi_1z, A1) + log_aligned(chi_2z, A2) - lj
        - stats.norm.logpdf(u) - stats.norm.logpdf(w)
    )
    np.testing.assert_allclose(residual, aligned_residual, atol=1e-8)


def test_reflection_is_a_sign_flip_of_the_first_spin(reparam):
    """phi_jl -> phi_jl + pi is (s1_x, s1_y) -> -(s1_x, s1_y), the rest fixed."""
    x = _prior(100, seed=6)
    x_prime, log_j = _forward(reparam, x)
    flipped = x.copy()
    flipped["phi_jl"] = np.remainder(x["phi_jl"] + np.pi, 2 * np.pi)
    f_prime, f_log_j = _forward(reparam, flipped)
    np.testing.assert_allclose(f_prime["s1_x"], -x_prime["s1_x"], atol=1e-12)
    np.testing.assert_allclose(f_prime["s1_y"], -x_prime["s1_y"], atol=1e-12)
    for name in ("chi_eff_prime", "chi_diff_prime", "s2_x", "s2_y"):
        np.testing.assert_array_equal(f_prime[name], x_prime[name])
    np.testing.assert_array_equal(f_log_j, log_j)


def test_inverse_drops_invalid_points(reparam):
    rng = np.random.default_rng(7)
    n = 1000
    x_prime = np.zeros(n, dtype=[(k, "f8") for k in PRIMES])
    for name in PRIMES:
        x_prime[name] = 3 * rng.standard_normal(n)
    x_prime["s2_x"][0] = np.nan
    q = rng.uniform(0.3, 1.0, n)
    q[1] = 0.0
    with np.errstate(all="raise"):
        x, log_j = _inverse(reparam, x_prime, q)
    assert np.all(np.isnan(x["a_2"][:2])) and np.all(np.isnan(log_j[:2]))
    kept = np.isfinite(log_j)
    assert kept[2:].mean() > 0.99
    assert np.all((x["a_1"][kept] >= 0) & (x["a_1"][kept] <= A1))
    assert np.all((x["tilt_2"][kept] >= 0) & (x["tilt_2"][kept] <= np.pi))


def test_bounds_must_start_at_zero():
    bounds = {
        "a_1": [0.01, A1], "a_2": [0, A2], "tilt_1": [0, np.pi],
        "tilt_2": [0, np.pi], "phi_12": [0, 2 * np.pi],
        "phi_jl": [0, 2 * np.pi],
    }
    with pytest.raises(RuntimeError):
        PrecessingSpinReparameterisation(parameters=NAMES, prior_bounds=bounds)
