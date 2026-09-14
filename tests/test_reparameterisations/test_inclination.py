"""Tests for the polarisation-ellipse inclination reparameterisation."""

import numpy as np
import pytest

from nessai_gw._ellipse import (
    PolarisationEllipse,
    detector_frame_basis,
    ideal_triangle_tensors,
)
from nessai_gw.group_mixture import ET_EMR_PLANE_NORMAL
from nessai_gw.reparameterisations import (
    PolarisationEllipseReparameterisation,
    known_reparameterisations,
)

REFERENCE_TIME = 1187008882.4
FIDUCIAL = dict(ra=0.9694, dec=-1.1491, psi=1.57, theta_jn=1.0472)


@pytest.fixture
def ellipse():
    return PolarisationEllipse(ET_EMR_PLANE_NORMAL, REFERENCE_TIME, FIDUCIAL)


@pytest.fixture
def reparam(ellipse):
    return PolarisationEllipseReparameterisation(
        parameters="theta_jn", prior_bounds=[0.0, np.pi], ellipse=ellipse
    )


def random_sky(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n))


def structured(theta_jn, ra, dec):
    x = np.zeros(len(theta_jn), dtype=[(n, "f8") for n in
                                       ("theta_jn", "ra", "dec")])
    x["theta_jn"], x["ra"], x["dec"] = theta_jn, ra, dec
    return x


# ---------------------------------------------------------------------------
# the map itself
# ---------------------------------------------------------------------------
def test_ideal_tensors_sum_to_zero():
    """The null stream is what makes the whole construction exact."""
    tensors = ideal_triangle_tensors(ET_EMR_PLANE_NORMAL)
    assert np.abs(tensors.sum(axis=0)).max() < 1e-14


def test_recovers_the_fiducial_inclination(ellipse):
    got = ellipse.cos_iota(np.array([FIDUCIAL["ra"]]), np.array([FIDUCIAL["dec"]]))
    assert got[0] == pytest.approx(np.cos(FIDUCIAL["theta_jn"]), abs=1e-10)


def test_cos_iota_in_range(ellipse):
    ra, dec = random_sky(2000, seed=3)
    c = ellipse.cos_iota(ra, dec)
    assert np.all(np.abs(c) <= 1.0)
    assert np.all(np.isfinite(c))


def _rotate_sky(ra, dec, gmst, normal, angle, reflect=False):
    """Rotate/reflect the source direction in the detector frame."""
    basis = detector_frame_basis(normal)
    theta, phi = 0.5 * np.pi - dec, ra - gmst
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                  np.cos(theta)], axis=-1)
    v = n @ basis.T
    if reflect:
        v = v * np.array([1.0, 1.0, -1.0])
    c, s = np.cos(angle), np.sin(angle)
    v = np.stack([c * v[..., 0] - s * v[..., 1],
                  s * v[..., 0] + c * v[..., 1], v[..., 2]], axis=-1)
    m = v @ basis
    return ((np.arctan2(m[..., 1], m[..., 0]) + gmst) % (2 * np.pi),
            np.arcsin(np.clip(m[..., 2], -1, 1)))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_quarter_turn_leaves_cos_iota_invariant(ellipse, k):
    """The Z4 generator of the group acts trivially on the residual's centre."""
    ra, dec = random_sky(500, seed=1)
    base = ellipse.cos_iota(ra, dec)
    r, d = _rotate_sky(ra, dec, ellipse.gmst, ET_EMR_PLANE_NORMAL,
                       k * np.pi / 2)
    assert np.abs(ellipse.cos_iota(r, d) - base).max() < 1e-10


def test_reflection_flips_cos_iota(ellipse):
    """The Z2 generator flips its sign -- exactly as cos(theta_jn) does."""
    ra, dec = random_sky(500, seed=2)
    base = ellipse.cos_iota(ra, dec)
    r, d = _rotate_sky(ra, dec, ellipse.gmst, ET_EMR_PLANE_NORMAL, 0.0,
                       reflect=True)
    assert np.abs(ellipse.cos_iota(r, d) + base).max() < 1e-10


def test_third_turn_is_not_a_symmetry(ellipse):
    """The triangle's own C3 is *not* the relevant group; guards the Z4 tests."""
    ra, dec = random_sky(500, seed=1)
    r, d = _rotate_sky(ra, dec, ellipse.gmst, ET_EMR_PLANE_NORMAL,
                       2 * np.pi / 3)
    assert np.abs(ellipse.cos_iota(r, d) - ellipse.cos_iota(ra, dec)).max() > 0.1


def test_torch_matches_numpy(ellipse):
    torch = pytest.importorskip("torch")
    ra, dec = random_sky(300, seed=4)
    got = ellipse.cos_iota_torch(
        torch.as_tensor(ra), torch.as_tensor(np.sin(dec))
    ).numpy()
    assert np.abs(got - ellipse.cos_iota(ra, dec)).max() < 1e-10


# ---------------------------------------------------------------------------
# the reparameterisation
# ---------------------------------------------------------------------------
def test_registered():
    assert "polarisation-ellipse" in known_reparameterisations


def test_rejects_the_wrong_parameter(ellipse):
    with pytest.raises(RuntimeError, match="must act on 'theta_jn'"):
        PolarisationEllipseReparameterisation(
            parameters="psi", prior_bounds=[0.0, np.pi], ellipse=ellipse
        )


def test_requires_an_ellipse_or_its_ingredients():
    with pytest.raises(ValueError, match="requires either"):
        PolarisationEllipseReparameterisation(
            parameters="theta_jn", prior_bounds=[0.0, np.pi]
        )


def test_builds_its_own_ellipse():
    r = PolarisationEllipseReparameterisation(
        parameters="theta_jn", prior_bounds=[0.0, np.pi],
        plane_normal=ET_EMR_PLANE_NORMAL, reference_time=REFERENCE_TIME,
        fiducial=FIDUCIAL,
    )
    assert isinstance(r.ellipse, PolarisationEllipse)


def test_round_trip(reparam):
    ra, dec = random_sky(1000, seed=5)
    theta = np.arccos(np.random.default_rng(6).uniform(-1, 1, 1000))
    x = structured(theta, ra, dec)
    x_prime = np.zeros(len(x), dtype=[("theta_jn_prime", "f8")])
    log_j = np.zeros(len(x))
    x, x_prime, log_j = reparam.reparameterise(x, x_prime, log_j)

    back = structured(np.zeros_like(theta), ra, dec)
    log_j_inv = np.zeros(len(x))
    back, _, log_j_inv = reparam.inverse_reparameterise(
        back, x_prime, log_j_inv
    )
    assert np.abs(back["theta_jn"] - theta).max() < 1e-10
    # the inverse Jacobian is minus the forward one at the same point
    assert np.abs(log_j + log_j_inv).max() < 1e-8


def test_jacobian_matches_numerical_derivative(reparam):
    ra, dec = random_sky(200, seed=7)
    theta = np.random.default_rng(8).uniform(0.2, np.pi - 0.2, 200)
    eps = 1e-6

    def prime(t):
        x = structured(t, ra, dec)
        xp = np.zeros(len(x), dtype=[("theta_jn_prime", "f8")])
        _, xp, _ = reparam.reparameterise(x, xp, np.zeros(len(x)))
        return xp["theta_jn_prime"]

    numerical = np.log(np.abs((prime(theta + eps) - prime(theta - eps))
                              / (2 * eps)))
    x = structured(theta, ra, dec)
    xp = np.zeros(len(x), dtype=[("theta_jn_prime", "f8")])
    _, _, log_j = reparam.reparameterise(x, xp, np.zeros(len(x)))
    assert np.abs(log_j - numerical).max() < 1e-5


def test_out_of_range_prime_is_rejected(reparam):
    """A prime point implying |cos theta_jn| > 1 must leave the prior range."""
    ra, dec = random_sky(50, seed=9)
    x = structured(np.zeros(50), ra, dec)
    x_prime = np.zeros(50, dtype=[("theta_jn_prime", "f8")])
    x_prime["theta_jn_prime"] = 25.0
    x, _, _ = reparam.inverse_reparameterise(x, x_prime, np.zeros(50))
    assert np.all((x["theta_jn"] < 0.0) | (x["theta_jn"] > np.pi))


def test_residual_is_odd_under_the_reflection(reparam, ellipse):
    """The prime coordinate transforms exactly as the group action assumes."""
    ra, dec = random_sky(400, seed=10)
    theta = np.arccos(np.random.default_rng(11).uniform(-1, 1, 400))

    def prime(t, r, d):
        x = structured(t, r, d)
        xp = np.zeros(len(x), dtype=[("theta_jn_prime", "f8")])
        _, xp, _ = reparam.reparameterise(x, xp, np.zeros(len(x)))
        return xp["theta_jn_prime"].copy()

    r, d = _rotate_sky(ra, dec, ellipse.gmst, ET_EMR_PLANE_NORMAL, 0.0,
                       reflect=True)
    assert np.abs(prime(np.pi - theta, r, d) + prime(theta, ra, dec)).max() < 1e-10


# ---------------------------------------------------------------------------
# both residual coordinates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("coordinate", ["angle", "cos"])
def test_round_trip_both_coordinates(ellipse, coordinate):
    r = PolarisationEllipseReparameterisation(
        parameters="theta_jn", prior_bounds=[0.0, np.pi], ellipse=ellipse,
        coordinate=coordinate,
    )
    ra, dec = random_sky(500, seed=12)
    theta = np.random.default_rng(13).uniform(0.05, np.pi - 0.05, 500)
    x = structured(theta, ra, dec)
    xp = np.zeros(len(x), dtype=[("theta_jn_prime", "f8")])
    log_j = np.zeros(len(x))
    x, xp, log_j = r.reparameterise(x, xp, log_j)
    back = structured(np.zeros_like(theta), ra, dec)
    back, _, log_j_inv = r.inverse_reparameterise(back, xp, np.zeros(len(x)))
    keep = back["theta_jn"] >= 0.0
    assert keep.mean() > 0.5
    assert np.abs(back["theta_jn"][keep] - theta[keep]).max() < 1e-10
    assert np.abs((log_j + log_j_inv)[keep]).max() < 1e-8


@pytest.mark.parametrize("coordinate", ["angle", "cos"])
def test_jacobian_both_coordinates(ellipse, coordinate):
    r = PolarisationEllipseReparameterisation(
        parameters="theta_jn", prior_bounds=[0.0, np.pi], ellipse=ellipse,
        coordinate=coordinate,
    )
    ra, dec = random_sky(200, seed=14)
    theta = np.random.default_rng(15).uniform(0.2, np.pi - 0.2, 200)
    eps = 1e-6

    def prime(t):
        x = structured(t, ra, dec)
        xp = np.zeros(len(x), dtype=[("theta_jn_prime", "f8")])
        _, xp, _ = r.reparameterise(x, xp, np.zeros(len(x)))
        return xp["theta_jn_prime"].copy()

    numerical = np.log(np.abs((prime(theta + eps) - prime(theta - eps)) / (2 * eps)))
    x = structured(theta, ra, dec)
    xp = np.zeros(len(x), dtype=[("theta_jn_prime", "f8")])
    _, _, log_j = r.reparameterise(x, xp, np.zeros(len(x)))
    assert np.abs(log_j - numerical).max() < 1e-5


def test_rejects_an_unknown_coordinate(ellipse):
    with pytest.raises(ValueError, match="must be 'angle' or 'cos'"):
        PolarisationEllipseReparameterisation(
            parameters="theta_jn", prior_bounds=[0.0, np.pi], ellipse=ellipse,
            coordinate="linear",
        )


# ---------------------------------------------------------------------------
# adaptive (heteroskedastic) residual width
# ---------------------------------------------------------------------------
def _hetero_sheet(ellipse, n=6000, seed=0):
    """A fake sheet: cos theta_jn = cos iota* + noise that grows with
    |cos iota*| (the fan), so a correct width fit whitens it."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 2 * np.pi, n)
    dec = np.arcsin(rng.uniform(-1, 1, n))
    cs = ellipse.cos_iota(ra, dec)
    ct = np.clip(cs + rng.normal(0.0, 0.05 + 0.45 * np.abs(cs), n),
                 -0.999, 0.999)
    return ra, dec, np.arccos(ct), cs


def test_residual_width_defaults_to_one():
    e = PolarisationEllipse(ET_EMR_PLANE_NORMAL, REFERENCE_TIME, FIDUCIAL)
    assert np.allclose(e.residual_width(np.linspace(0, 1, 5)), 1.0)


def _width_is_trivial(e):
    return np.allclose(e.residual_width(np.linspace(0, 1, 9)), 1.0)


@pytest.mark.parametrize("coordinate", ["cos", "angle"])
def test_adaptive_width_fits_and_whitens(coordinate):
    e = PolarisationEllipse(ET_EMR_PLANE_NORMAL, REFERENCE_TIME, FIDUCIAL)
    r = PolarisationEllipseReparameterisation(
        parameters="theta_jn", prior_bounds=[0.0, np.pi], ellipse=e,
        coordinate=coordinate, adaptive_width=True, width_min_points=200,
    )
    ra, dec, theta, cs = _hetero_sheet(e)
    x = structured(theta, ra, dec)
    r.update(x)
    # the width picked up the fan: wider towards the face-on points
    assert e.residual_width(0.8) > 1.8 * e.residual_width(0.1)

    a = np.abs(cs)

    def spread_ratio(rep):
        xp = np.zeros(len(x), dtype=[("theta_jn_prime", "f8")])
        _, xp, _ = rep.reparameterise(x.copy(), xp, np.zeros(len(x)))
        v = xp["theta_jn_prime"]
        return v[a < 0.25].std() / v[a > 0.55].std()

    plain = PolarisationEllipseReparameterisation(
        parameters="theta_jn", prior_bounds=[0.0, np.pi], ellipse=e,
        coordinate=coordinate,  # adaptive_width off -> constant scale
    )
    # the constant-scale lock leaves a large spread mismatch across the sky;
    # the fitted width brings it much closer to flat
    assert abs(np.log(spread_ratio(plain))) > 0.8
    assert abs(np.log(spread_ratio(r))) < 0.5 * abs(np.log(spread_ratio(plain)))


def test_adaptive_width_round_trip_and_jacobian():
    e = PolarisationEllipse(ET_EMR_PLANE_NORMAL, REFERENCE_TIME, FIDUCIAL)
    r = PolarisationEllipseReparameterisation(
        parameters="theta_jn", prior_bounds=[0.0, np.pi], ellipse=e,
        coordinate="cos", adaptive_width=True, width_min_points=200,
    )
    ra, dec, theta, _ = _hetero_sheet(e, seed=1)
    x = structured(theta, ra, dec)
    r.update(x)
    xp = np.zeros(len(x), dtype=[("theta_jn_prime", "f8")])
    log_j = np.zeros(len(x))
    _, xp, log_j = r.reparameterise(x, xp, log_j)
    back = structured(np.zeros_like(theta), ra, dec)
    back, _, log_j_inv = r.inverse_reparameterise(back, xp, np.zeros(len(x)))
    ok = back["theta_jn"] >= 0.0
    assert np.abs(back["theta_jn"][ok] - theta[ok]).max() < 1e-9
    assert np.abs((log_j + log_j_inv)[ok]).max() < 1e-8

    eps = 1e-6

    def prime(t):
        xx = structured(t, ra, dec)
        pp = np.zeros(len(xx), dtype=[("theta_jn_prime", "f8")])
        _, pp, _ = r.reparameterise(xx, pp, np.zeros(len(xx)))
        return pp["theta_jn_prime"]

    num = np.log(np.abs((prime(theta + eps) - prime(theta - eps)) / (2 * eps)))
    assert np.abs(log_j - num).max() < 1e-5


def test_adaptive_width_is_a_noop_without_the_flag():
    e = PolarisationEllipse(ET_EMR_PLANE_NORMAL, REFERENCE_TIME, FIDUCIAL)
    r = PolarisationEllipseReparameterisation(
        parameters="theta_jn", prior_bounds=[0.0, np.pi], ellipse=e,
    )
    ra, dec, theta, _ = _hetero_sheet(e, seed=2)
    r.update(structured(theta, ra, dec))
    assert _width_is_trivial(e)


def test_adaptive_width_waits_for_enough_points():
    e = PolarisationEllipse(ET_EMR_PLANE_NORMAL, REFERENCE_TIME, FIDUCIAL)
    r = PolarisationEllipseReparameterisation(
        parameters="theta_jn", prior_bounds=[0.0, np.pi], ellipse=e,
        adaptive_width=True, width_min_points=5000,
    )
    ra, dec, theta, _ = _hetero_sheet(e, n=1000, seed=3)
    r.update(structured(theta, ra, dec))
    assert _width_is_trivial(e)
