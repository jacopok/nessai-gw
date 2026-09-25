"""Tests for the Roulet et al. (arXiv:2207.03508) chirp-distance
reparameterisation."""

import numpy as np
import pytest

from nessai_gw._effective_distance import (
    chirp_distance,
    dominant_detector,
    luminosity_distance_from_chirp_distance,
    response_R,
)
from nessai_gw._ellipse import ideal_triangle_tensors
from nessai_gw._geometry import greenwich_mean_sidereal_time
from nessai_gw.group_mixture import (
    CHIRP_DISTANCE_REQUIRES,
    ET_EMR_DETECTOR_TENSORS,
    ET_EMR_PLANE_NORMAL,
    ET_EMR_VERTEX,
    TriangularDetectorGroupAction,
    detector_tensors,
    triangular_group_reparameterisations,
)
from nessai_gw.reparameterisations import (
    ChirpDistanceReparameterisation,
    known_reparameterisations,
)

REFERENCE_TIME = 1187008882.4
FIDUCIAL = dict(ra=0.9694, dec=-1.1491, psi=1.57, theta_jn=1.0472)
NAMES = ("luminosity_distance", "chirp_mass", "theta_jn", "ra", "dec", "psi")


@pytest.fixture
def tensors():
    return ET_EMR_DETECTOR_TENSORS


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


def test_requires_tensors():
    with pytest.raises(ValueError, match="tensors"):
        ChirpDistanceReparameterisation(
            parameters="luminosity_distance", fiducial=FIDUCIAL,
            reference_time=REFERENCE_TIME,
        )


@pytest.mark.parametrize(
    "time_kwargs",
    [{}, {"gmst": 1.0, "reference_time": REFERENCE_TIME}],
)
def test_requires_exactly_one_of_gmst_and_reference_time(tensors, time_kwargs):
    with pytest.raises(ValueError, match="exactly one of `gmst`"):
        ChirpDistanceReparameterisation(
            parameters="luminosity_distance", tensors=tensors,
            fiducial=FIDUCIAL, **time_kwargs,
        )


def test_reference_time_matches_gmst(tensors, gmst, reparam):
    r = ChirpDistanceReparameterisation(
        parameters="luminosity_distance", tensors=tensors,
        reference_time=REFERENCE_TIME, fiducial=FIDUCIAL,
    )
    assert r.gmst == pytest.approx(gmst)
    assert r.k0 == reparam.k0


def test_single_tensor_and_k0_range(tensors, gmst):
    r = ChirpDistanceReparameterisation(
        parameters="luminosity_distance", tensors=tensors[1], gmst=gmst, k0=0,
    )
    assert r.tensors.shape == (1, 3, 3)
    with pytest.raises(ValueError, match="out of range"):
        ChirpDistanceReparameterisation(
            parameters="luminosity_distance", tensors=tensors, gmst=gmst,
            k0=3,
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


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_et_emr_tensors_match_the_plane_normal():
    """The stored tensors are traceless, lie in the ET-EMR plane and (nearly)
    sum to zero, as three nested interferometers of a triangle must."""
    for d in ET_EMR_DETECTOR_TENSORS:
        np.testing.assert_allclose(d, d.T)
        assert abs(np.trace(d)) < 1e-12
        assert np.linalg.norm(d @ ET_EMR_PLANE_NORMAL) < 5e-3
    assert np.abs(ET_EMR_DETECTOR_TENSORS.sum(0)).max() < 1e-3


def test_detector_tensors_reads_interferometers():
    class _Geom:
        def __init__(self, d):
            self.detector_tensor = d

    class _Ifo:
        def __init__(self, d):
            self.geometry = _Geom(d)

    class _IfoDirect:
        def __init__(self, d):
            self.detector_tensor = d

    ifos = [_Ifo(ET_EMR_DETECTOR_TENSORS[0]), _IfoDirect(ET_EMR_DETECTOR_TENSORS[1])]
    np.testing.assert_array_equal(
        detector_tensors(ifos), ET_EMR_DETECTOR_TENSORS[:2]
    )


def test_ideal_triangle_misses_the_real_arm_orientation(gmst):
    """Why the reparameterisation must be built on the real tensors: the
    idealised triangle only fixes the plane, and a single sub-detector's
    ``|R_k|`` depends on the in-plane arm orientation."""
    _, _, theta_jn, ra, dec, psi = random_points(2000, seed=11)
    real = np.abs(response_R(ET_EMR_DETECTOR_TENSORS, ra, dec, psi, theta_jn, gmst))
    ideal = np.abs(response_R(
        ideal_triangle_tensors(ET_EMR_PLANE_NORMAL), ra, dec, psi, theta_jn, gmst
    ))
    best = min(
        np.abs(ideal[:, k] - real[:, j]).max()
        for k in range(3) for j in range(3)
    )
    assert best > 0.05


@pytest.mark.parametrize(
    "tensors, tol",
    [
        (ideal_triangle_tensors(ET_EMR_PLANE_NORMAL, 0.3), 1e-10),
        (ET_EMR_DETECTOR_TENSORS, 1e-2),
    ],
)
def test_group_invariant(tensors, tol):
    """``|R_k|`` -- hence ``chirp_distance`` -- is invariant under every
    element of the 32-element triangular group: exactly for a planar
    triangle, to ~1e-3 for the real ET-EMR tensors.  This is what lets the
    prime-space action pass the coordinate through untouched."""
    torch = pytest.importorskip("torch")
    action = TriangularDetectorGroupAction(
        REFERENCE_TIME, ET_EMR_PLANE_NORMAL, ET_EMR_VERTEX,
        polarisation_quarter=True, azimuth_offset=0.3,
    )
    rng = np.random.default_rng(12)
    n = 500
    pts = {
        "ra": rng.uniform(0, 2 * np.pi, n),
        "sin_dec": rng.uniform(-1, 1, n),
        "cos_theta_jn": rng.uniform(-1, 1, n),
        "psi": rng.uniform(0, np.pi, n),
        "phase": rng.uniform(0, 2 * np.pi, n),
        "geocent_time": np.full(n, REFERENCE_TIME),
    }

    def abs_r(p):
        return np.abs(response_R(
            tensors, p["ra"], np.arcsin(p["sin_dec"]), p["psi"],
            np.arccos(p["cos_theta_jn"]), action.gmst,
        ))

    r0 = abs_r(pts)
    pts_t = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in pts.items()}
    for mode in range(action.group_size):
        out = action(pts_t, torch.full((n,), mode))
        r = abs_r({k: v.numpy() for k, v in out.items()})
        assert np.median(np.abs(np.log(r / r0))) < tol


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------
WIRING_PARAMETERS = [
    "chirp_mass", "mass_ratio", "luminosity_distance", "theta_jn", "psi",
    "phase", "ra", "dec", "geocent_time",
]


def test_triangular_wiring_defaults_to_et_emr_tensors():
    reps = triangular_group_reparameterisations(
        WIRING_PARAMETERS, REFERENCE_TIME, chirp_distance=True,
        chirp_distance_fiducial=FIDUCIAL,
    )
    spec = reps["luminosity_distance"]
    assert spec["reparameterisation"] == "chirp-distance"
    np.testing.assert_array_equal(spec["tensors"], ET_EMR_DETECTOR_TENSORS)
    assert spec["reference_time"] == REFERENCE_TIME
    assert "plane_normal" not in spec and "azimuth_offset" not in spec
    assert next(iter(reps)) == "luminosity_distance"


def test_triangular_wiring_needs_k0_or_fiducial():
    with pytest.raises(ValueError, match="chirp_distance_k0"):
        triangular_group_reparameterisations(
            WIRING_PARAMETERS, REFERENCE_TIME, chirp_distance=True,
        )


@pytest.mark.parametrize("drop", CHIRP_DISTANCE_REQUIRES)
def test_triangular_wiring_needs_the_inputs(drop):
    names = [n for n in WIRING_PARAMETERS if n != drop]
    with pytest.raises(ValueError, match=drop):
        triangular_group_reparameterisations(
            names, REFERENCE_TIME, chirp_distance=True, chirp_distance_k0=0,
        )


def test_triangular_wiring_without_distance_is_a_no_op():
    names = [n for n in WIRING_PARAMETERS if n != "luminosity_distance"]
    assert triangular_group_reparameterisations(
        names, REFERENCE_TIME, chirp_distance=True,
    ) == triangular_group_reparameterisations(names, REFERENCE_TIME)
