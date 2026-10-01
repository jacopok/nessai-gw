"""Tests for the tabulated effective-spin transform: an exact bijection with
an exact Jacobian whose prior is only approximately N(0, I)."""

import pickle

import numpy as np
import pytest
from scipy import stats

from nessai_gw._effective_spin import EffectiveSpinTransform
from nessai_gw._effective_spin_table import TabulatedEffectiveSpinTransform
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
    r = EffectiveSpinReparameterisation(prior_bounds=bounds, tabulated=True)
    assert isinstance(r._transform, TabulatedEffectiveSpinTransform)
    r = EffectiveSpinReparameterisation(prior_bounds=bounds)
    assert isinstance(r._transform, EffectiveSpinTransform)
