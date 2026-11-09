"""Tests for the tabulated effective-spin transform: an exact bijection with
an exact Jacobian whose prior is only approximately N(0, I)."""

import pickle

import numpy as np
import pytest
from scipy import stats

from nessai_gw._effective_spin import EffectiveSpinTransform
from nessai_gw._effective_spin_table import (
    SpinOrbitChart,
    TabulatedEffectiveSpinTransform,
    spin_orbit_coefficients,
)
from nessai_gw.reparameterisations import EffectiveSpinReparameterisation

A1, A2 = 0.05, 0.04


def _points(n, seed=0, q_min=0.3):
    rng = np.random.default_rng(seed)
    return (
        A1 * rng.uniform(0, 1, n) * rng.uniform(-1, 1, n),
        A2 * rng.uniform(0, 1, n) * rng.uniform(-1, 1, n),
        rng.uniform(q_min, 1.0, n),
    )


@pytest.fixture(scope="module")
def transform():
    return TabulatedEffectiveSpinTransform(A1, A2)


def test_round_trip_is_exact(transform):
    c1, c2, q = _points(5000, seed=1, q_min=0.05)
    u, w, lj = transform.forward(c1, c2, q)
    b1, b2, lj_back = transform.inverse(u, w, q)
    np.testing.assert_allclose(b1, c1, rtol=0, atol=1e-15)
    np.testing.assert_allclose(b2, c2, rtol=0, atol=1e-15)
    np.testing.assert_allclose(lj_back, lj, atol=1e-9)


def test_inverse_then_forward_is_exact_away_from_the_far_tails(transform):
    rng = np.random.default_rng(2)
    u, w = rng.uniform(-5, 5, (2, 5000))
    q = rng.uniform(0.05, 1.0, 5000)
    c1, c2, lj = transform.inverse(u, w, q)
    assert np.all(np.abs(c1) <= A1) and np.all(np.abs(c2) <= A2)
    u2, w2, lj2 = transform.forward(c1, c2, q)
    np.testing.assert_allclose(u2, u, atol=1e-9)
    np.testing.assert_allclose(w2, w, atol=1e-9)
    np.testing.assert_allclose(lj2, lj, atol=1e-6)


def test_jacobian_is_that_of_the_map(transform):
    c1, c2, q = _points(300, seed=3)
    h = 1e-8

    def uw(a, b):
        u, w, _ = transform.forward(a, b, q)
        return np.stack([u, w], axis=-1)

    jac = np.stack(
        [(uw(c1 + h, c2) - uw(c1 - h, c2)) / (2 * h),
         (uw(c1, c2 + h) - uw(c1, c2 - h)) / (2 * h)],
        axis=-1,
    )
    _, _, lj = transform.forward(c1, c2, q)
    err = np.abs(np.log(np.abs(np.linalg.det(jac))) - lj)
    # a difference straddling a table knot sees two slopes: allow a few
    assert np.median(err) < 1e-5
    assert np.mean(err > 1e-3) < 0.02


def test_prior_is_approximately_standard_normal(transform):
    c1, c2, q = _points(20000, seed=4)
    u, w, _ = transform.forward(c1, c2, q)
    for y in (u, w):
        assert abs(y.mean()) < 0.05 and abs(y.std() - 1) < 0.05
        assert stats.kstest(y, "norm").statistic < 0.02
    exact = EffectiveSpinTransform(A1, A2)
    ue, we, _ = exact.forward(c1[:2000], c2[:2000], q[:2000])
    assert np.sqrt(np.mean((u[:2000] - ue) ** 2)) < 0.02
    assert np.sqrt(np.mean((w[:2000] - we) ** 2)) < 0.1


def test_pickle_drops_the_tables(transform):
    data = pickle.dumps(transform)
    assert len(data) < 2000
    back = pickle.loads(data)
    c1, c2, q = _points(10, seed=5)
    np.testing.assert_array_equal(back.forward(c1, c2, q)[0],
                                  transform.forward(c1, c2, q)[0])


def test_reparameterisation_option():
    bounds = {"chi_1": [-A1, A1], "chi_2": [-A2, A2]}
    r = EffectiveSpinReparameterisation(prior_bounds=bounds, tabulated=False)
    assert isinstance(r._transform, EffectiveSpinTransform)
    r = EffectiveSpinReparameterisation(prior_bounds=bounds)
    assert isinstance(r._transform, TabulatedEffectiveSpinTransform)


def test_reparameterisation_drops_unresolved_tails():
    """Far beyond the tables' resolution the log-Jacobian can be non-finite;
    those points come back as NaN spins (out of bounds), without warnings."""
    bounds = {"chi_1": [-A1, A1], "chi_2": [-A2, A2]}
    r = EffectiveSpinReparameterisation(prior_bounds=bounds, tabulated=True)
    rng = np.random.default_rng(1)
    n = 100_000
    x = np.zeros(n, dtype=[(k, "f8") for k in ("chi_1", "chi_2", "mass_ratio")])
    x["mass_ratio"] = rng.uniform(0.05, 1.0, n)
    x_prime = np.zeros(
        n, dtype=[("chi_eff_prime", "f8"), ("chi_diff_prime", "f8")]
    )
    x_prime["chi_eff_prime"] = 6 * rng.standard_normal(n)
    x_prime["chi_diff_prime"] = 6 * rng.standard_normal(n)
    with np.errstate(all="raise"):
        x, _, log_j = r.inverse_reparameterise(x, x_prime, np.zeros(n))
    kept = np.isfinite(x["chi_1"]) & np.isfinite(x["chi_2"])
    assert not kept.all()
    assert np.all(np.isfinite(log_j[kept]))
    assert np.all(np.isnan(x["chi_2"][~np.isfinite(log_j)]))


# -- the spin-orbit chart ------------------------------------------------------

MSUN_S = 4.925490947641267e-06


@pytest.fixture(scope="module")
def chart(transform):
    return SpinOrbitChart(transform)


def test_spin_orbit_y_is_the_phase_coefficient(chart):
    """``4 a_max_1 y - 16 pi`` is the 1.5PN coefficient of the phase in units
    of its chirp-mass factor, at every mass ratio."""
    c1, c2, q = _points(2000, seed=6, q_min=0.1)
    y, _, _ = chart.forward(c1, c2, q)
    eta = q / (1 + q) ** 2
    m1, m2 = 1 / (1 + q), q / (1 + q)
    beta = (c1 * (113 * m1**2 + 75 * eta) + c2 * (113 * m2**2 + 75 * eta)) / 12
    total = 1.4 * eta**-0.6 * MSUN_S  # any chirp mass: it factors out
    c3 = 3 / (128 * eta) * (np.pi * total) ** (-2 / 3) * (4 * beta - 16 * np.pi)
    ratio = (4 * A1 * y - 16 * np.pi) / c3
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
    a1, r, kappa = spin_orbit_coefficients(np.array([1.0]))
    assert kappa[0] == 1 and r[0] == 1 and np.isclose(a1[0], 47 / 12)


def test_spin_orbit_round_trip_is_exact(chart):
    c1, c2, q = _points(5000, seed=7, q_min=0.1)
    y, w, lj = chart.forward(c1, c2, q)
    b1, b2, lj_back = chart.inverse(y, w, q)
    np.testing.assert_allclose(b1, c1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(b2, c2, rtol=0, atol=1e-14)
    np.testing.assert_allclose(lj_back, lj, atol=1e-8)


def test_spin_orbit_jacobian_is_that_of_the_map(chart):
    c1, c2, q = _points(300, seed=8)
    h = 1e-8

    def yw(a, b):
        y, w, _ = chart.forward(a, b, q)
        return np.stack([y, w], axis=-1)

    jac = np.stack(
        [(yw(c1 + h, c2) - yw(c1 - h, c2)) / (2 * h),
         (yw(c1, c2 + h) - yw(c1, c2 - h)) / (2 * h)],
        axis=-1,
    )
    _, _, lj = chart.forward(c1, c2, q)
    err = np.abs(np.log(np.abs(np.linalg.det(jac))) - lj)
    assert np.median(err) < 1e-5
    assert np.mean(err > 1e-3) < 0.02


def test_spin_orbit_w_is_along_constant_beta(chart):
    """At fixed ``q`` and ``y``, moving along ``w`` keeps beta fixed."""
    q = np.full(50, 0.6)
    y = np.full(50, chart.forward(0.01, -0.02, 0.6)[0][0])
    w = np.linspace(-3, 3, 50)
    c1, c2, _ = chart.inverse(y, w, q)
    assert np.all(np.isfinite(c1))
    eta = 0.6 / 1.6**2
    beta = c1 * (113 / 1.6**2 + 75 * eta) + c2 * (113 * 0.36 / 1.6**2 + 75 * eta)
    np.testing.assert_allclose(beta, beta[0], rtol=1e-12)
    assert np.all(np.diff(c1) > 0)


def test_spin_orbit_inverse_outside_the_prior_is_nan(chart):
    q = np.array([0.5, 0.5, 1.0])
    y, w, lj = chart.forward(np.zeros(3), np.zeros(3), q)
    y = y + np.array([0.0, 1e3, -1e3])
    c1, c2, lj = chart.inverse(y, w, q)
    assert np.isfinite(c1[0]) and np.isfinite(lj[0])
    assert np.all(np.isnan(c1[1:])) and np.all(np.isnan(c2[1:]))
    assert np.all(np.isnan(lj[1:]))


def _spin_x(n, seed):
    c1, c2, q = _points(n, seed=seed)
    x = np.zeros(n, dtype=[(k, "f8") for k in ("chi_1", "chi_2", "mass_ratio")])
    x["chi_1"], x["chi_2"], x["mass_ratio"] = c1, c2, q
    x_prime = np.zeros(n, dtype=[("chi_eff_prime", "f8"), ("chi_diff_prime", "f8")])
    return x, x_prime


def test_reparameterisation_spin_chart():
    bounds = {"chi_1": [-A1, A1], "chi_2": [-A2, A2]}
    r = EffectiveSpinReparameterisation(prior_bounds=bounds, spin_chart="spin-orbit")
    assert r.spin_chart == "spin-orbit"
    assert isinstance(r._transform, SpinOrbitChart)
    x, x_prime = _spin_x(1000, seed=9)
    _, x_prime, lj = r.reparameterise(x.copy(), x_prime, np.zeros(len(x)))
    y, w, lj_chart = r._transform.forward(x["chi_1"], x["chi_2"], x["mass_ratio"])
    np.testing.assert_array_equal(x_prime["chi_eff_prime"], y)
    np.testing.assert_array_equal(lj, lj_chart)
    back = x.copy()
    back["chi_1"] = back["chi_2"] = np.nan
    back, _, lj_back = r.inverse_reparameterise(back, x_prime, np.zeros(len(x)))
    np.testing.assert_allclose(back["chi_1"], x["chi_1"], atol=1e-14)
    np.testing.assert_allclose(back["chi_2"], x["chi_2"], atol=1e-14)
    np.testing.assert_allclose(lj_back, -lj, atol=1e-8)
    # pickled and back, still the chart; switched on an existing instance
    r2 = pickle.loads(pickle.dumps(r))
    assert r2.spin_chart == "spin-orbit"
    r2.spin_chart = "effective"
    assert type(r2._transform) is TabulatedEffectiveSpinTransform
    _, xp_eff, _ = r2.reparameterise(x.copy(), x_prime.copy(), np.zeros(len(x)))
    u, _, _ = TabulatedEffectiveSpinTransform(A1, A2).forward(
        x["chi_1"], x["chi_2"], x["mass_ratio"])
    np.testing.assert_array_equal(xp_eff["chi_eff_prime"], u)
    r2.spin_chart = "spin-orbit"
    r2.spin_chart = "spin-orbit"  # not wrapped twice
    assert type(r2._transform.inner) is TabulatedEffectiveSpinTransform


def test_reparameterisation_spin_chart_errors():
    bounds = {"chi_1": [-A1, A1], "chi_2": [-A2, A2]}
    assert EffectiveSpinReparameterisation(prior_bounds=bounds).spin_chart == "effective"
    with pytest.raises(ValueError, match="tabulated"):
        EffectiveSpinReparameterisation(prior_bounds=bounds, tabulated=False,
                                        spin_chart="spin-orbit")
    with pytest.raises(ValueError, match="spin_chart"):
        EffectiveSpinReparameterisation(prior_bounds=bounds, spin_chart="beta")
