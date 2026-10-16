"""Tests for the single-interferometer (D4h) group action."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
bilby = pytest.importorskip("bilby")

from nessai_gw.group_mixture import (  # noqa: E402
    AdaptiveFundamentalDomain,
    PrimeSpaceTriangularGroupAction,
    _prime_parameter_names,
)
from nessai_gw.single_detector import (  # noqa: E402
    SINGLE_INTERFEROMETER_SKY_GROUP_SIZE,
    SKY_INVERSE_TABLE,
    SKY_PRODUCT_TABLE,
    SingleInterferometerGroupAction,
    arms_from_geodetic,
    effective_arm_basis,
)

REFERENCE_TIME = 1187008882.4
LAT, LON, BISECTOR = 0.7, 0.2, 0.4
BNS_PARAMETERS = [
    "chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance",
    "dec", "ra", "theta_jn", "psi", "phase", "geocent_time",
]


def _interferometer(opening_angle):
    half = np.degrees(0.5 * opening_angle)
    return bilby.gw.detector.Interferometer(
        "X", None, minimum_frequency=10, maximum_frequency=100, length=4,
        latitude=np.degrees(LAT), longitude=np.degrees(LON), elevation=0.0,
        xarm_azimuth=np.degrees(BISECTOR) - half,
        yarm_azimuth=np.degrees(BISECTOR) + half,
    )


def _points(n=300, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "ra": torch.as_tensor(rng.uniform(0, 2 * np.pi, n)),
        "sin_dec": torch.as_tensor(rng.uniform(-1, 1, n)),
        "cos_theta_jn": torch.as_tensor(rng.uniform(-1, 1, n)),
        "psi": torch.as_tensor(rng.uniform(0, np.pi, n)),
        "phase": torch.as_tensor(rng.uniform(0, 2 * np.pi, n)),
        "geocent_time": torch.full((n,), REFERENCE_TIME, dtype=torch.float64),
    }


def _response_22(ifo, p):
    """Complex (2, 2) response ``(F+ (1+c^2)/2 - i Fx c) e^{-2 i phase}``."""
    out = []
    for i in range(len(p["ra"])):
        ra = float(p["ra"][i])
        dec = float(np.arcsin(float(p["sin_dec"][i])))
        psi = float(p["psi"][i])
        fp = ifo.antenna_response(ra, dec, REFERENCE_TIME, psi, "plus")
        fc = ifo.antenna_response(ra, dec, REFERENCE_TIME, psi, "cross")
        c = float(p["cos_theta_jn"][i])
        out.append((fp * (1 + c * c) / 2 - 1j * fc * c)
                   * np.exp(-2j * float(p["phase"][i])))
    return np.array(out)


def test_sky_tables_form_a_group():
    n = SINGLE_INTERFEROMETER_SKY_GROUP_SIZE
    assert np.all(SKY_PRODUCT_TABLE[0] == np.arange(n))
    for row in SKY_PRODUCT_TABLE:
        assert sorted(row) == list(range(n))
    for g in range(n):
        assert SKY_PRODUCT_TABLE[g, SKY_INVERSE_TABLE[g]] == 0
    # D4h is not abelian
    assert not np.array_equal(SKY_PRODUCT_TABLE, SKY_PRODUCT_TABLE.T)


@pytest.mark.parametrize("alpha", [0.5 * np.pi, np.pi / 3, 1.2])
def test_effective_arms_diagonalise_the_tensor(alpha):
    x, y = arms_from_geodetic(LAT, LON, BISECTOR, alpha)
    D = 0.5 * (np.outer(x, x) - np.outer(y, y))
    e1, e2, _ = effective_arm_basis(x, y)
    expected = 0.5 * np.sin(alpha) * (np.outer(e1, e1) - np.outer(e2, e2))
    assert np.allclose(D, expected, atol=1e-12)


@pytest.mark.parametrize("alpha", [0.5 * np.pi, np.pi / 3])
def test_matches_bilby_arms(alpha):
    ifo = _interferometer(alpha)
    x, y = arms_from_geodetic(LAT, LON, BISECTOR, alpha)
    assert np.allclose(ifo.geometry.x, x) and np.allclose(ifo.geometry.y, y)


@pytest.mark.parametrize("alpha", [0.5 * np.pi, np.pi / 3])
def test_response_invariant_and_inverse(alpha):
    ifo = _interferometer(alpha)
    action = SingleInterferometerGroupAction.from_interferometer(
        ifo, REFERENCE_TIME, polarisation_quarter=True
    )
    assert action.group_size == 64
    assert np.isclose(action.opening_angle, alpha)
    p = _points()
    n = len(p["ra"])
    ref = _response_22(ifo, p)
    t_det = p["geocent_time"] + action._geocenter_delay(
        p["ra"], torch.asin(p["sin_dec"]))
    count = torch.zeros(n)
    for g in range(action.group_size):
        modes = torch.full((n,), g)
        q = action(p, modes)
        assert np.allclose(_response_22(ifo, q), ref, atol=1e-6), g
        q_t_det = q["geocent_time"] + action._geocenter_delay(
            q["ra"], torch.asin(q["sin_dec"]))
        assert torch.allclose(q_t_det, t_det, atol=1e-9, rtol=0)
        back = action(q, modes, inverse=True)
        for k in ("ra", "sin_dec", "cos_theta_jn", "psi", "phase"):
            assert torch.allclose(back[k], p[k], atol=1e-8), (g, k)
        count += action.in_fundamental_domain(q).float()
    assert torch.all(count == 1)


@pytest.fixture(scope="module")
def names():
    return _prime_parameter_names(
        BNS_PARAMETERS, REFERENCE_TIME, sky_2d=True, psi_single=True
    )


@pytest.fixture(scope="module")
def prime_action(names):
    base = SingleInterferometerGroupAction.from_geodetic(
        REFERENCE_TIME, LAT, LON, BISECTOR, polarisation_quarter=True
    )
    return PrimeSpaceTriangularGroupAction(base, names)


def _as_dict(t, names):
    return {n: t[:, i] for i, n in enumerate(names)}


def _stack(d, names):
    return torch.stack([d[n] for n in names], dim=1)


@pytest.fixture(scope="module")
def prime_points(names):
    rng = np.random.default_rng(3)
    n = 1500
    cols = [
        rng.uniform(0.0, 1.0, n) if nm in ("sky_u", "sky_v")
        else rng.uniform(-1.0, 1.0, n)
        for nm in names
    ]
    return torch.as_tensor(np.stack(cols, axis=1), dtype=torch.float64)


def test_prime_space_round_trip_and_partition(prime_action, names, prime_points):
    x = prime_points
    n = len(x)
    count = torch.zeros(n)
    for g in range(prime_action._action.group_size):
        modes = torch.full((n,), g)
        img = prime_action(_as_dict(x, names), modes)
        back = _stack(prime_action(img, modes, inverse=True), names)
        assert torch.allclose(back, x, atol=1e-6), g
        count += prime_action.in_fundamental_domain(img).float()
    assert torch.all(count == 1)


@pytest.mark.parametrize("seams", [(0.0, 0.37, 0.62, 0.0), (0.0, 0.9, 0.05, 0.0)])
def test_adaptive_domain_round_trip(prime_action, names, prime_points, seams):
    x = prime_points
    n = len(x)
    canon = torch.full_like(x, float("nan"))
    found = torch.zeros(n, dtype=torch.bool)
    for g in range(prime_action._action.group_size):
        img = _stack(prime_action(_as_dict(x, names), torch.full((n,), g)), names)
        ok = prime_action.in_fundamental_domain(_as_dict(img, names)) & ~found
        canon[ok] = img[ok]
        found |= ok
    assert found.all()
    afd = AdaptiveFundamentalDomain(prime_action, names)
    afd._seams.copy_(torch.tensor(seams, dtype=torch.float64))
    afd._seen.fill_(True)
    y = afd.forward(canon)
    assert afd._box_valid(y, False).all()
    assert torch.allclose(afd.inverse(y), canon, atol=1e-9)
    # the sky seam of a mirror-bounded domain never moves
    afd.update(canon)
    assert float(afd._seams[0]) == 0.0


@pytest.mark.parametrize("shift_phi_jl", [True, False])
def test_precessing_response_invariance(shift_phi_jl):
    """The response to a precessing binary (IMRPhenomPv2) is invariant under
    every element, the polarisation/phase quarter turn included, with the
    improper ones shifting phi_jl by pi -- and not without the shift."""
    pytest.importorskip("lalsimulation")
    ifo = _interferometer(0.5 * np.pi)
    action = SingleInterferometerGroupAction.from_interferometer(
        ifo, REFERENCE_TIME, polarisation_quarter=True
    )
    frequencies = np.arange(20.0, 200.0, 0.25)
    intrinsic = dict(mass_1=30.0, mass_2=20.0, a_1=0.8, a_2=0.5, tilt_1=1.3,
                     tilt_2=2.0, phi_12=1.0, luminosity_distance=500.0)

    def strain(ra, dec, psi, theta_jn, phase, phi_jl):
        wf = bilby.gw.source.lal_binary_black_hole(
            frequencies, **intrinsic, theta_jn=theta_jn, phase=phase,
            phi_jl=phi_jl, waveform_approximant="IMRPhenomPv2",
            reference_frequency=20.0, minimum_frequency=20.0)
        return sum(
            ifo.antenna_response(ra, dec, REFERENCE_TIME, psi, m) * wf[m]
            for m in ("plus", "cross")
        )

    points = dict(_points(n=4, seed=3))
    points["phi_jl"] = torch.as_tensor(
        np.random.default_rng(4).uniform(0, 2 * np.pi, 4))
    worst_improper = 0.0
    for i in range(4):
        pt = {k: v[i:i + 1] for k, v in points.items()}
        p = {k: float(v) for k, v in pt.items()}
        h0 = strain(p["ra"], np.arcsin(p["sin_dec"]), p["psi"],
                    np.arccos(p["cos_theta_jn"]), p["phase"], p["phi_jl"])
        for g in range(action.group_size):
            out = action(pt, torch.tensor([g]))
            h1 = strain(
                float(out["ra"]), float(np.arcsin(float(out["sin_dec"]))),
                float(out["psi"]), float(np.arccos(float(out["cos_theta_jn"]))),
                float(out["phase"]),
                float(out["phi_jl"]) if shift_phi_jl else p["phi_jl"],
            )
            error = np.linalg.norm(h1 - h0) / np.linalg.norm(h0)
            _, improper, _ = action.decode_modes(torch.tensor([g]))
            if bool(improper[0]):
                worst_improper = max(worst_improper, error)
            else:
                assert error < 1e-6, g
    if shift_phi_jl:
        assert worst_improper < 1e-6
    else:
        assert worst_improper > 0.2
