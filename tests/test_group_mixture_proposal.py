"""Tests for the triangular-detector group-mixture proposal wiring in :mod:`nessai_gw.group_mixture`.

These exercise the prime-space adapter, the reparameterisation profile, the
prime-name probe and the proposal factory.  Only the proposal-factory tests
need a version of nessai that ships ``nessai.flowmodel.group_mixture``; they
skip otherwise.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nessai_gw.group_mixture import (  # noqa: E402
    ETTriangleGroupAction,
    PrimeSpaceTriangularGroupAction,
    TriangularDetectorGroupAction,
    _prime_parameter_names,
    recommended_sky_azimuth_offset,
    triangular_group_reparameterisations,
    make_et_group_flow_proposal,
)

REFERENCE_TIME = 1187008882.4

BNS_PARAMETERS = [
    "chirp_mass",
    "mass_ratio",
    "chi_1",
    "chi_2",
    "lambda_1",
    "lambda_2",
    "luminosity_distance",
    "theta_jn",
    "psi",
    "phase",
    "ra",
    "dec",
    "geocent_time",
]

PRIME_NAMES = [
    "ra_dec_x",
    "ra_dec_y",
    "ra_dec_z",
    "psi_x",
    "psi_y",
    "delta_phase_x",
    "delta_phase_y",
    "theta_jn_prime",
    "geocent_time_prime",
]

def _has_group_mixture():
    try:
        import nessai.flowmodel.group_mixture  # noqa: F401

        return True
    except ImportError:
        return False


requires_group_mixture = pytest.mark.skipif(
    not _has_group_mixture(),
    reason="nessai.flowmodel.group_mixture not available",
)


def test_triangular_group_reparameterisations():
    reps = triangular_group_reparameterisations(BNS_PARAMETERS, REFERENCE_TIME)
    assert reps["chi_1"] == {"reparameterisation": "aligned-spin"}
    assert reps["chi_2"] == {"reparameterisation": "aligned-spin"}
    assert reps["lambda_1"]["reparameterisation"] == "logit"
    assert reps["lambda_1"]["update_bounds"] is False
    assert reps["theta_jn"] == {
        "reparameterisation": "angle-sine",
        "update_bounds": False,
    }
    assert reps["geocent_time"]["reparameterisation"] == "scaleandshift"
    assert reps["geocent_time"]["shift"] == pytest.approx(REFERENCE_TIME)
    assert reps["phase"] == {"reparameterisation": "polarisation-phase"}
    # parameters that get the default GW reparameterisation are not listed
    assert "ra" not in reps
    assert "psi" not in reps
    assert "chirp_mass" not in reps


@pytest.fixture(scope="module")
def prime_points():
    rng = np.random.default_rng(2024)
    n = 4000
    ra = rng.uniform(0, 2 * np.pi, n)
    dec = np.arcsin(rng.uniform(-1, 1, n))
    r_sky = rng.uniform(0.5, 1.5, n)
    cos_dec = np.cos(dec)
    psi = rng.uniform(0, np.pi, n)
    r_psi = rng.uniform(0.5, 1.5, n)
    delta_phase = rng.uniform(0, 2 * np.pi, n)
    r_dphase = rng.uniform(0.5, 1.5, n)
    return {
        "ra_dec_x": torch.as_tensor(r_sky * cos_dec * np.cos(ra)),
        "ra_dec_y": torch.as_tensor(r_sky * cos_dec * np.sin(ra)),
        "ra_dec_z": torch.as_tensor(r_sky * np.sin(dec)),
        "psi_x": torch.as_tensor(r_psi * np.cos(2.0 * psi)),
        "psi_y": torch.as_tensor(r_psi * np.sin(2.0 * psi)),
        "delta_phase_x": torch.as_tensor(r_dphase * np.cos(delta_phase)),
        "delta_phase_y": torch.as_tensor(r_dphase * np.sin(delta_phase)),
        "theta_jn_prime": torch.as_tensor(rng.uniform(-1, 1, n)),
        "geocent_time_prime": torch.as_tensor(rng.uniform(-20, 20, n)),
    }


def test_et_action_is_triangular_subclass():
    assert issubclass(ETTriangleGroupAction, TriangularDetectorGroupAction)


@pytest.fixture(scope="module")
def prime_action():
    base = ETTriangleGroupAction(reference_time=REFERENCE_TIME)
    return PrimeSpaceTriangularGroupAction(base, PRIME_NAMES)


@pytest.fixture(scope="module")
def prime_action_16():
    base = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, phase_reflection=True
    )
    return PrimeSpaceTriangularGroupAction(base, PRIME_NAMES)


def test_prime_space_action_missing_coordinate():
    base = ETTriangleGroupAction(reference_time=REFERENCE_TIME)
    with pytest.raises(RuntimeError, match="prime space is missing"):
        PrimeSpaceTriangularGroupAction(base, ["ra_dec_x", "ra_dec_y", "ra_dec_z"])


@pytest.mark.parametrize("g", range(ETTriangleGroupAction.group_size))
def test_prime_space_action_inverse_round_trip(prime_action, prime_points, g):
    n = len(prime_points["ra_dec_x"])
    modes = torch.full((n,), g, dtype=torch.long)
    mapped = prime_action(prime_points, modes)
    back = prime_action(mapped, modes, inverse=True)
    for name in PRIME_NAMES:
        assert torch.allclose(
            back[name], prime_points[name], atol=1e-6, rtol=1e-6
        ), name


def test_prime_space_action_identity(prime_action, prime_points):
    n = len(prime_points["ra_dec_x"])
    modes = torch.zeros(n, dtype=torch.long)
    mapped = prime_action(prime_points, modes)
    for name in PRIME_NAMES:
        assert torch.allclose(
            mapped[name], prime_points[name], atol=1e-6, rtol=1e-6
        ), name


def _decode_pair(px, py, scale):
    r = torch.hypot(px, py)
    ang = torch.remainder(torch.atan2(py, px), 2 * np.pi) / scale
    return ang, r


def _to_physical(points):
    """prime-space dict -> physical (ra, sin_dec, cos_theta_jn, psi, phase)."""
    # Flow sky coordinates are detector-frame; rotate back to equatorial to
    # match PrimeSpaceTriangularGroupAction._decode.
    _R = torch.as_tensor(
        ETTriangleGroupAction(
            reference_time=REFERENCE_TIME
        ).sky_frame_rotation
    )
    _d = points
    _dx, _dy, _dz = _d["ra_dec_x"], _d["ra_dec_y"], _d["ra_dec_z"]
    sx = _R[0, 0] * _dx + _R[1, 0] * _dy + _R[2, 0] * _dz
    sy = _R[0, 1] * _dx + _R[1, 1] * _dy + _R[2, 1] * _dz
    sz = _R[0, 2] * _dx + _R[1, 2] * _dy + _R[2, 2] * _dz
    r_sky = torch.sqrt(sx * sx + sy * sy + sz * sz)
    ra = torch.remainder(torch.atan2(sy, sx), 2 * np.pi)
    sin_dec = torch.clamp(sz / r_sky, -1.0, 1.0)
    psi, _ = _decode_pair(points["psi_x"], points["psi_y"], 2.0)
    dphase, _ = _decode_pair(
        points["delta_phase_x"], points["delta_phase_y"], 1.0
    )
    u = points["theta_jn_prime"]
    # angle-sine: u = 2 theta_jn / pi - 1  ->  cos(theta_jn) sign = -sign(u)
    cos_theta_jn = torch.cos(0.5 * np.pi * (u + 1.0))
    sign_ct = -torch.sign(u)
    phase = torch.remainder(dphase - sign_ct * psi, 2 * np.pi)
    return {
        "ra": ra,
        "sin_dec": sin_dec,
        "cos_theta_jn": cos_theta_jn,
        "psi": psi,
        "phase": phase,
        "geocent_time": points["geocent_time_prime"] * 0.0
        + ETTriangleGroupAction(reference_time=REFERENCE_TIME).reference_time,
    }


def _delta_phase_from_physical(phys):
    return torch.remainder(
        phys["phase"] + torch.sign(phys["cos_theta_jn"]) * phys["psi"],
        2 * np.pi,
    )


def test_prime_space_action_physical_phase_invariant(prime_action, prime_points):
    """The group-invariant physical ``phase`` is unchanged by every element."""
    n = len(prime_points["ra_dec_x"])
    phase0 = _to_physical(prime_points)["phase"]
    for g in range(ETTriangleGroupAction.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        mapped = prime_action(prime_points, modes)
        d = torch.remainder(_to_physical(mapped)["phase"] - phase0, 2 * np.pi)
        d = torch.minimum(d, 2 * np.pi - d)
        assert d.max() < 1e-5, g


def test_prime_space_action_preserves_radii(prime_action, prime_points):
    n = len(prime_points["ra_dec_x"])
    r_sky = torch.hypot(
        torch.hypot(prime_points["ra_dec_x"], prime_points["ra_dec_y"]),
        prime_points["ra_dec_z"],
    )
    r_psi = torch.hypot(prime_points["psi_x"], prime_points["psi_y"])
    r_dphase = torch.hypot(
        prime_points["delta_phase_x"], prime_points["delta_phase_y"]
    )
    for g in range(ETTriangleGroupAction.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        mapped = prime_action(prime_points, modes)
        r_sky_t = torch.hypot(
            torch.hypot(mapped["ra_dec_x"], mapped["ra_dec_y"]),
            mapped["ra_dec_z"],
        )
        r_psi_t = torch.hypot(mapped["psi_x"], mapped["psi_y"])
        r_dphase_t = torch.hypot(
            mapped["delta_phase_x"], mapped["delta_phase_y"]
        )
        assert torch.allclose(r_sky_t, r_sky, atol=1e-6)
        assert torch.allclose(r_psi_t, r_psi, atol=1e-6)
        assert torch.allclose(r_dphase_t, r_dphase, atol=1e-6)


def test_prime_space_action_delta_phase_matches_physical(
    prime_action, prime_points
):
    """The adapter's delta_phase == (decode fully -> run the physical action ->
    apply ``phase' + sign(cos theta_jn') * psi'``)."""
    n = len(prime_points["ra_dec_x"])
    base = prime_action._action
    phys0 = _to_physical(prime_points)
    for g in range(ETTriangleGroupAction.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        phys = base(phys0, modes)
        want = _delta_phase_from_physical(phys)
        mapped = prime_action(prime_points, modes)
        got, _ = _decode_pair(
            mapped["delta_phase_x"], mapped["delta_phase_y"], 1.0
        )
        d = torch.remainder(got - want, 2 * np.pi)
        d = torch.minimum(d, 2 * np.pi - d)
        assert d.max() < 1e-5, g


@pytest.mark.parametrize("g", range(16))
def test_prime_space_action_16_inverse_round_trip(
    prime_action_16, prime_points, g
):
    n = len(prime_points["ra_dec_x"])
    modes = torch.full((n,), g, dtype=torch.long)
    mapped = prime_action_16(prime_points, modes)
    back = prime_action_16(mapped, modes, inverse=True)
    for name in PRIME_NAMES:
        assert torch.allclose(
            back[name], prime_points[name], atol=1e-6, rtol=1e-6
        ), (g, name)


def test_prime_space_action_16_physical_phase(prime_action_16, prime_points):
    """Physical phase is invariant mod 2pi for modes 0..7 and shifted by pi
    for modes 8..15 (invariant mod pi across all 16)."""
    n = len(prime_points["ra_dec_x"])
    phase0 = _to_physical(prime_points)["phase"]
    for g in range(16):
        modes = torch.full((n,), g, dtype=torch.long)
        mapped = prime_action_16(prime_points, modes)
        got = _to_physical(mapped)["phase"]
        shift = np.pi if g >= 8 else 0.0
        d = torch.remainder(got - phase0 - shift, 2 * np.pi)
        d = torch.minimum(d, 2 * np.pi - d)
        assert d.max() < 1e-5, g
        # always invariant modulo pi
        dpi = torch.remainder(got - phase0, np.pi)
        dpi = torch.minimum(dpi, np.pi - dpi)
        assert dpi.max() < 1e-5, g


def test_prime_space_action_16_delta_phase_matches_physical(
    prime_action_16, prime_points
):
    n = len(prime_points["ra_dec_x"])
    base = prime_action_16._action
    phys0 = _to_physical(prime_points)
    for g in range(16):
        modes = torch.full((n,), g, dtype=torch.long)
        phys = base(phys0, modes)
        want = _delta_phase_from_physical(phys)
        mapped = prime_action_16(prime_points, modes)
        got, _ = _decode_pair(
            mapped["delta_phase_x"], mapped["delta_phase_y"], 1.0
        )
        d = torch.remainder(got - want, 2 * np.pi)
        d = torch.minimum(d, 2 * np.pi - d)
        assert d.max() < 1e-5, g


def _in_domain_count(action, prime_points, group_size):
    """For each point, how many of its ``group_size`` orbit images the adapter
    reports as canonical."""
    n = len(prime_points["ra_dec_x"])
    count = torch.zeros(n)
    for g in range(group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        image = action(prime_points, modes)
        count += action.in_fundamental_domain(image).float()
    return count


def test_prime_space_fundamental_domain_partitions_orbit(
    prime_action, prime_points
):
    """Exactly one of the 8 orbit images is canonical (generic points)."""
    count = _in_domain_count(prime_action, prime_points, 8)
    assert torch.all(count == 1), count.unique(return_counts=True)


def test_prime_space_16_fundamental_domain_partitions_orbit(
    prime_action_16, prime_points
):
    """With ``phase_reflection`` exactly one of the 16 orbit images is
    canonical, and it is the one with ``delta_phase in [0, pi)`` -- i.e. the
    Z2 is folded on the flow coordinate, not the (unconstrained) raw phase."""
    count = _in_domain_count(prime_action_16, prime_points, 16)
    assert torch.all(count == 1), count.unique(return_counts=True)

    n = len(prime_points["ra_dec_x"])
    for g in range(16):
        modes = torch.full((n,), g, dtype=torch.long)
        image = prime_action_16(prime_points, modes)
        canon = prime_action_16.in_fundamental_domain(image)
        dphase, _ = _decode_pair(
            image["delta_phase_x"], image["delta_phase_y"], 1.0
        )
        assert torch.all(dphase[canon] < np.pi + 1e-6), g


def test_prime_parameter_names():
    prime = _prime_parameter_names(BNS_PARAMETERS, REFERENCE_TIME)
    if not any(n.startswith("ra_dec") for n in prime):
        pytest.skip("probe fell back to the name-order heuristic")
    assert {"ra_dec_x", "ra_dec_y", "ra_dec_z"} <= set(prime)
    assert "psi_x" in prime and "psi_y" in prime
    assert "delta_phase_x" in prime and "delta_phase_y" in prime
    assert "phase_x" not in prime and "phase_y" not in prime
    # every physical parameter contributes at least one prime coordinate
    assert len(prime) >= len(BNS_PARAMETERS)


@requires_group_mixture
@pytest.mark.parametrize("prime_space", [True, False])
@pytest.mark.parametrize("phase_reflection", [False, True])
def test_make_et_group_flow_proposal(prime_space, phase_reflection):
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS,
        REFERENCE_TIME,
        prime_space=prime_space,
        phase_reflection=phase_reflection,
    )
    from nessai.proposal import FlowProposal
    from nessai.flowmodel.group_mixture import GroupFlowProposalMixin
    from nessai_gw.proposals import GWReparamMixin

    assert cls._FlowModelClass.group_size == (16 if phase_reflection else 8)
    assert cls.__qualname__ == "TriangularGroupFlowProposal"
    assert cls.__module__ == "nessai_gw.group_mixture"
    assert issubclass(cls, (GWReparamMixin, GroupFlowProposalMixin, FlowProposal))
    mro = cls.__mro__
    assert mro.index(GWReparamMixin) < mro.index(GroupFlowProposalMixin)
    assert mro.index(GroupFlowProposalMixin) < mro.index(FlowProposal)
    assert cls._FlowModelClass is not None
    # class is resolvable by module.qualname (nessai checkpoints via pickle)
    import nessai_gw.group_mixture as gm

    assert getattr(gm, "TriangularGroupFlowProposal") is cls


def test_rotated_anglepair_matches_anglepair_rotated():
    """RotatedAnglePair's prime output is AnglePair's output rotated by R,
    and the forward/inverse round trip recovers ra/dec."""
    from nessai.reparameterisations import AnglePair
    from nessai_gw.reparameterisations.sky import RotatedAnglePair

    R = ETTriangleGroupAction(reference_time=REFERENCE_TIME).sky_frame_rotation
    bounds = {"ra": np.array([0.0, 2 * np.pi]), "dec": np.array([-np.pi / 2, np.pi / 2])}
    rng = np.random.default_rng(0)
    plain = AnglePair(
        parameters=["ra", "dec"], prior_bounds=dict(bounds),
        convention="ra-dec", rng=np.random.default_rng(1),
    )
    rot = RotatedAnglePair(
        parameters=["ra", "dec"], prior_bounds=dict(bounds),
        convention="ra-dec", rotation=R, rng=np.random.default_rng(1),
    )
    n = 500
    x = np.zeros(n, dtype=[(p, "f8") for p in plain.parameters])
    x["ra"] = rng.uniform(0, 2 * np.pi, n)
    x["dec"] = np.arcsin(rng.uniform(-1, 1, n))
    x[plain.parameters[2]] = rng.uniform(0.5, 1.5, n)
    dt = [(p, "f8") for p in plain.prime_parameters]
    xp_a = np.zeros(n, dtype=dt)
    xp_b = np.zeros(n, dtype=dt)
    _, xp_a, _ = plain.reparameterise(x.copy(), xp_a, np.zeros(n))
    _, xp_b, _ = rot.reparameterise(x.copy(), xp_b, np.zeros(n))
    va = np.stack([xp_a[p] for p in plain.prime_parameters], axis=-1)
    vb = np.stack([xp_b[p] for p in rot.prime_parameters], axis=-1)
    np.testing.assert_allclose(vb, va @ R.T, atol=1e-9)

    back = np.zeros(n, dtype=[(p, "f8") for p in rot.parameters])
    back, _, _ = rot.inverse_reparameterise(back, xp_b, np.zeros(n))
    np.testing.assert_allclose(np.cos(back["ra"]), np.cos(x["ra"]), atol=1e-6)
    np.testing.assert_allclose(np.sin(back["ra"]), np.sin(x["ra"]), atol=1e-6)
    np.testing.assert_allclose(back["dec"], x["dec"], atol=1e-6)


def test_fundamental_domain_is_positive_sky_octant(prime_action, prime_points):
    """With the detector-frame rotation, the canonical image of each orbit has
    all three sky prime coordinates non-negative (the axis-aligned octant)."""
    n = len(prime_points["ra_dec_x"])
    seen = torch.zeros(n, dtype=torch.bool)
    for g in range(8):
        modes = torch.full((n,), g, dtype=torch.long)
        image = prime_action(prime_points, modes)
        canon = prime_action.in_fundamental_domain(image)
        for c in ("ra_dec_x", "ra_dec_y", "ra_dec_z"):
            assert torch.all(image[c][canon] > -1e-9), (g, c)
        seen |= canon
    assert torch.all(seen)


def test_azimuth_offset_rotates_frame_and_moves_seam():
    a0 = ETTriangleGroupAction(reference_time=REFERENCE_TIME)
    off = 0.6
    a1 = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, azimuth_offset=off
    )
    # e3 (normal) unchanged; e1/e2 rotated by `off` about e3
    np.testing.assert_allclose(a1._basis_np[2], a0._basis_np[2], atol=1e-12)
    c, s = np.cos(off), np.sin(off)
    np.testing.assert_allclose(
        a1._basis_np[0], c * a0._basis_np[0] + s * a0._basis_np[1], atol=1e-12
    )
    # sky_frame_rotation follows, still orthogonal
    np.testing.assert_allclose(
        a1.sky_frame_rotation @ a1.sky_frame_rotation.T, np.eye(3), atol=1e-10
    )


def test_recommended_sky_azimuth_offset_centres_the_wedge():
    """A localised sky blob: the recommended offset moves its folded azimuth
    circular mean to pi/4, and re-measuring with that offset confirms it."""
    rng = np.random.default_rng(0)
    ra = rng.normal(0.9694, 0.05, 4000)
    dec = rng.normal(-1.1491, 0.05, 4000)
    a0 = ETTriangleGroupAction(reference_time=REFERENCE_TIME)
    d0 = recommended_sky_azimuth_offset(a0, ra, dec)
    assert d0["concentration"] > 0.8  # tight blob -> well localised
    a1 = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME,
        azimuth_offset=d0["recommended_azimuth_offset"],
    )
    d1 = recommended_sky_azimuth_offset(a1, ra, dec)
    assert d1["circ_mean_lambda"] == pytest.approx(np.pi / 4, abs=0.02)
    assert d1["recommended_azimuth_offset"] == pytest.approx(0.0, abs=0.02)


@requires_group_mixture
@pytest.mark.parametrize("boundary_reflection", [False, True])
def test_make_et_group_flow_proposal_reflection_and_offset(boundary_reflection):
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS,
        REFERENCE_TIME,
        azimuth_offset=0.3,
        boundary_reflection=boundary_reflection,
    )
    expect = (
        ["ra_dec_x", "ra_dec_y", "ra_dec_z"] if boundary_reflection else None
    )
    assert (
        getattr(cls._FlowModelClass, "reflect_parameters", None) == expect
    )


@requires_group_mixture
def test_make_et_group_flow_proposal_missing_parameter():
    with pytest.raises(RuntimeError, match="missing"):
        make_et_group_flow_proposal(["ra", "dec", "psi"], REFERENCE_TIME)
