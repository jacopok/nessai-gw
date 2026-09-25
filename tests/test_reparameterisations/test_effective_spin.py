"""Tests for the effective-spin reparameterisation (Roulet et al. 2022,
arXiv:2207.03508, Sec. IV, for bilby's AlignedSpin prior)."""

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
from scipy.special import ndtr

from nessai_gw._effective_spin import (
    EffectiveSpinTransform,
    _aligned_spin_pdf,
    aligned_spin_cdf,
)
from nessai_gw.group_mixture import triangular_group_reparameterisations
from nessai_gw.reparameterisations import (
    EffectiveSpinReparameterisation,
    known_reparameterisations,
)

A1, A2 = 0.05, 0.04
NAMES = ("chi_1", "chi_2", "mass_ratio")


def _aligned_spin(a, n, rng):
    """Draws from bilby's AlignedSpin: uniform magnitude, isotropic tilt."""
    return a * rng.uniform(0, 1, n) * rng.uniform(-1, 1, n)


def _points(n, seed=0):
    rng = np.random.default_rng(seed)
    return (
        _aligned_spin(A1, n, rng),
        _aligned_spin(A2, n, rng),
        rng.uniform(0.3, 1.0, n),
    )


@pytest.fixture(scope="module")
def transform():
    return EffectiveSpinTransform(A1, A2)


@pytest.fixture
def reparam():
    return EffectiveSpinReparameterisation(
        parameters=["chi_1", "chi_2"],
        prior_bounds={"chi_1": [-A1, A1], "chi_2": [-A2, A2]},
    )


def _structured(c1, c2, q):
    x = np.zeros(len(c1), dtype=[(n, "f8") for n in NAMES])
    x["chi_1"], x["chi_2"], x["mass_ratio"] = c1, c2, q
    return x


# ---------------------------------------------------------------------------
# the transform
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("s", [-0.06, -0.02, -1e-3, 0.0, 3e-3, 0.03, 0.07])
def test_cdf_of_s_matches_quad(transform, s):
    q = 0.7

    def integrand(x):
        return _aligned_spin_pdf(x, A1) * aligned_spin_cdf((s - x) / q, A2)

    ref = quad(
        integrand, -A1, A1, points=sorted({0.0, s, s - q * A2, s + q * A2}),
        limit=200, epsabs=1e-14,
    )[0]
    u, _ = transform.u_of_s(np.array([s]), np.array([q]))
    assert ndtr(u[0]) == pytest.approx(ref, abs=1e-9)


def test_conditional_cdf_matches_quad(transform):
    q, s = 0.6, 0.01
    lo, hi = max(-A1, s - q * A2), min(A1, s + q * A2)

    def joint(x):
        return _aligned_spin_pdf(x, A1) * _aligned_spin_pdf((s - x) / q, A2)

    pts = [p for p in (0.0, s) if lo < p < hi]
    z = quad(joint, lo, hi, points=pts, limit=200)[0]
    for chi1 in (lo + 1e-4, -0.01, 0.0, 0.004, s, 0.03):
        ref = quad(joint, lo, chi1, points=[p for p in pts if p < chi1],
                   limit=200)[0] / z
        w, _ = transform.w_of_chi1(np.array([chi1]), np.array([s]),
                                   np.array([q]))
        assert ndtr(w[0]) == pytest.approx(ref, abs=1e-8)


def test_prior_maps_to_standard_normal(transform):
    """Under the AlignedSpin prior (at any q) the pair is exactly N(0, I)."""
    c1, c2, q = _points(20000, seed=1)
    u, w, _ = transform.forward(c1, c2, q)
    for y in (u, w):
        assert stats.kstest(y, "norm").pvalue > 1e-3
    assert abs(np.corrcoef(u, w)[0, 1]) < 0.03
    # independent of q as well
    assert abs(np.corrcoef(u, q)[0, 1]) < 0.03
    assert abs(np.corrcoef(w, q)[0, 1]) < 0.03


def test_u_is_a_function_of_chi_eff(transform):
    """At fixed q, points with the same chi_eff share u."""
    rng = np.random.default_rng(2)
    q = np.full(50, 0.8)
    chi_eff = 0.01
    c1 = rng.uniform(chi_eff * 1.8 - 0.8 * A2, A1, 50)
    c2 = ((1 + q) * chi_eff - c1) / q
    u, w, _ = transform.forward(c1, c2, q)
    assert np.ptp(u) < 1e-9
    assert np.all(np.diff(w[np.argsort(c1)]) > 0)


def test_round_trip(transform):
    c1, c2, q = _points(5000, seed=3)
    u, w, lj = transform.forward(c1, c2, q)
    b1, b2, lj_back = transform.inverse(u, w, q)
    np.testing.assert_allclose(b1, c1, atol=1e-12)
    np.testing.assert_allclose(b2, c2, atol=1e-12)
    np.testing.assert_allclose(lj_back, lj, atol=1e-7)


def test_round_trip_in_the_tails(transform):
    q = np.full(6, 0.9)
    u = np.array([-7.0, -5.0, 0.0, 0.0, 5.0, 7.0])
    w = np.array([0.0, 6.0, -7.0, 7.0, -6.0, 0.0])
    c1, c2, _ = transform.inverse(u, w, q)
    assert np.all(np.abs(c1) < A1) and np.all(np.abs(c2) < A2)
    u2, w2, _ = transform.forward(c1, c2, q)
    np.testing.assert_allclose(u2, u, atol=1e-8)
    np.testing.assert_allclose(w2, w, atol=1e-8)


def test_jacobian_matches_finite_difference(transform):
    c1, c2, q = _points(200, seed=4)
    # stay away from the cusps where finite differences are poor
    keep = (np.abs(c1) > 2e-3) & (np.abs(c2) > 2e-3)
    c1, c2, q = c1[keep], c2[keep], q[keep]
    h = 1e-7

    def uw(a, b):
        u, w, _ = transform.forward(a, b, q)
        return np.stack([u, w], axis=-1)

    jac = np.stack(
        [
            (uw(c1 + h, c2) - uw(c1 - h, c2)) / (2 * h),
            (uw(c1, c2 + h) - uw(c1, c2 - h)) / (2 * h),
        ],
        axis=-1,
    )
    _, _, lj = transform.forward(c1, c2, q)
    np.testing.assert_allclose(
        np.log(np.abs(np.linalg.det(jac))), lj, atol=1e-5
    )


# ---------------------------------------------------------------------------
# the reparameterisation
# ---------------------------------------------------------------------------
def test_registered():
    assert "effective-spin" in known_reparameterisations


def test_init(reparam):
    assert reparam.parameters == ["chi_1", "chi_2"]
    assert reparam.prime_parameters == ["chi_eff_prime", "chi_diff_prime"]
    assert "mass_ratio" in reparam.inverse_input_parameters
    assert reparam._transform.a1 == A1 and reparam._transform.a2 == A2


def test_init_errors():
    with pytest.raises(RuntimeError, match="exactly two"):
        EffectiveSpinReparameterisation(
            parameters=["chi_1"], prior_bounds={"chi_1": [-A1, A1]}
        )
    with pytest.raises(RuntimeError, match="symmetric"):
        EffectiveSpinReparameterisation(
            parameters=["chi_1", "chi_2"],
            prior_bounds={"chi_1": [-A1, A1], "chi_2": [0.0, A2]},
        )


def test_reparameterisation_round_trip(reparam):
    c1, c2, q = _points(1000, seed=5)
    x = _structured(c1, c2, q)
    x_prime = np.zeros(len(x), dtype=[(n, "f8") for n in reparam.prime_parameters])
    x, x_prime, log_j = reparam.reparameterise(x, x_prime, np.zeros(len(x)))

    back = _structured(np.zeros_like(c1), np.zeros_like(c2), q)
    back, _, log_j_inv = reparam.inverse_reparameterise(
        back, x_prime, np.zeros(len(x))
    )
    np.testing.assert_allclose(back["chi_1"], c1, atol=1e-12)
    np.testing.assert_allclose(back["chi_2"], c2, atol=1e-12)
    np.testing.assert_allclose(log_j, -log_j_inv, atol=1e-7)


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------
WIRING = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance",
          "theta_jn", "psi", "phase", "ra", "dec", "geocent_time"]


def test_triangular_wiring():
    reps = triangular_group_reparameterisations(
        WIRING, 1187008882.4, effective_spin=True
    )
    assert reps["chi_1"] == {
        "reparameterisation": "effective-spin",
        "parameters": ["chi_1", "chi_2"],
    }
    assert "chi_2" not in reps
    assert next(iter(reps)) == "chi_1"
    default = triangular_group_reparameterisations(WIRING, 1187008882.4)
    assert default["chi_1"] == {"reparameterisation": "aligned-spin"}


def test_triangular_wiring_needs_mass_ratio():
    with pytest.raises(ValueError, match="mass_ratio"):
        triangular_group_reparameterisations(
            [n for n in WIRING if n != "mass_ratio"], 1187008882.4,
            effective_spin=True,
        )


def test_triangular_wiring_without_spins_is_a_no_op():
    names = [n for n in WIRING if n != "chi_2"]
    assert triangular_group_reparameterisations(
        names, 1187008882.4, effective_spin=True
    ) == triangular_group_reparameterisations(names, 1187008882.4)
