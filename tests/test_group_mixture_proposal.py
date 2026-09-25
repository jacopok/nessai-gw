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
    recommended_polarisation_offset,
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
    "delta_phase",
    "theta_jn_prime",
    "t_det",
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
    assert (
        reps["geocent_time"]["reparameterisation"] == "detector-center-time"
    )
    assert reps["geocent_time"]["reference_time"] == pytest.approx(
        REFERENCE_TIME
    )
    assert len(reps["geocent_time"]["vertex"]) == 3
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
    # single [-1, 1) polarisation-phase coordinate
    delta_phase = rng.uniform(-1.0, 1.0, n)
    return {
        "ra_dec_x": torch.as_tensor(r_sky * cos_dec * np.cos(ra)),
        "ra_dec_y": torch.as_tensor(r_sky * cos_dec * np.sin(ra)),
        "ra_dec_z": torch.as_tensor(r_sky * np.sin(dec)),
        "psi_x": torch.as_tensor(r_psi * np.cos(2.0 * psi)),
        "psi_y": torch.as_tensor(r_psi * np.sin(2.0 * psi)),
        "delta_phase": torch.as_tensor(delta_phase),
        "theta_jn_prime": torch.as_tensor(rng.uniform(-1, 1, n)),
        # detector-centre arrival time (invariant under the group)
        "t_det": torch.as_tensor(rng.uniform(-20, 20, n)),
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


def test_prime_space_action_t_det_invariant(prime_action, prime_points):
    """The detector-centre arrival time is carried through untouched by every
    group element (Tissino et al. 2026, arXiv:2606.04918)."""
    n = len(prime_points["ra_dec_x"])
    for g in range(ETTriangleGroupAction.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        mapped = prime_action(prime_points, modes)
        assert torch.equal(mapped["t_det"], prime_points["t_det"]), g


def _decode_pair(px, py, scale):
    r = torch.hypot(px, py)
    ang = torch.remainder(torch.atan2(py, px), 2 * np.pi) / scale
    return ang, r


def _decode_delta_phase(prime, scale=1.0):
    """Single ``delta_phase`` coordinate in ``[-1, 1)`` -> physical angle."""
    return torch.remainder((prime + 1.0) * np.pi / scale, 2 * np.pi)


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
    dphase = _decode_delta_phase(points["delta_phase"])
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
    for g in range(ETTriangleGroupAction.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        mapped = prime_action(prime_points, modes)
        r_sky_t = torch.hypot(
            torch.hypot(mapped["ra_dec_x"], mapped["ra_dec_y"]),
            mapped["ra_dec_z"],
        )
        r_psi_t = torch.hypot(mapped["psi_x"], mapped["psi_y"])
        assert torch.allclose(r_sky_t, r_sky, atol=1e-6)
        assert torch.allclose(r_psi_t, r_psi, atol=1e-6)
        # single delta_phase coordinate stays in [-1, 1)
        assert torch.all(mapped["delta_phase"] >= -1.0 - 1e-9)
        assert torch.all(mapped["delta_phase"] < 1.0 + 1e-9)


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
        got = _decode_delta_phase(mapped["delta_phase"])
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
        got = _decode_delta_phase(mapped["delta_phase"])
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
        dphase = _decode_delta_phase(image["delta_phase"])
        assert torch.all(dphase[canon] < np.pi + 1e-6), g


# ---------------------------------------------------------------------------
# arg-alpha-beta phase coordinates
# ---------------------------------------------------------------------------
ARG_AB_PRIME_NAMES = [
    "ra_dec_x",
    "ra_dec_y",
    "ra_dec_z",
    "arg_alpha",
    "arg_beta",
    "theta_jn_prime",
    "t_det",
]


@pytest.fixture(scope="module")
def arg_ab_prime_points():
    rng = np.random.default_rng(7)
    n = 4000
    ra = rng.uniform(0, 2 * np.pi, n)
    dec = np.arcsin(rng.uniform(-1, 1, n))
    r_sky = rng.uniform(0.5, 1.5, n)
    cd = np.cos(dec)
    return {
        "ra_dec_x": torch.as_tensor(r_sky * cd * np.cos(ra)),
        "ra_dec_y": torch.as_tensor(r_sky * cd * np.sin(ra)),
        "ra_dec_z": torch.as_tensor(r_sky * np.sin(dec)),
        "arg_alpha": torch.as_tensor(rng.uniform(-1.0, 1.0, n)),
        "arg_beta": torch.as_tensor(rng.uniform(-1.0, 1.0, n)),
        "theta_jn_prime": torch.as_tensor(rng.uniform(-1, 1, n)),
        "t_det": torch.as_tensor(rng.uniform(-20, 20, n)),
    }


def _arg_ab_of(psi, phase):
    return (
        torch.remainder(phase - psi, 2 * np.pi) / np.pi - 1.0,
        torch.remainder(phase + psi, 2 * np.pi) / np.pi - 1.0,
    )


@pytest.mark.parametrize("polarisation_quarter", [False, True])
def test_triangular_group_reparameterisations_arg_alpha_beta(
    polarisation_quarter,
):
    reps = triangular_group_reparameterisations(
        BNS_PARAMETERS, REFERENCE_TIME, phase_coordinates="arg-alpha-beta"
    )
    assert "phase" not in reps
    assert reps["arg-alpha-beta"] == {"parameters": ["psi", "phase"]}
    # the polarisation-phase / psi entries are gone
    assert "psi" not in reps


def test_triangular_group_reparameterisations_bad_phase_coordinates():
    with pytest.raises(ValueError, match="phase_coordinates"):
        triangular_group_reparameterisations(
            BNS_PARAMETERS, REFERENCE_TIME, phase_coordinates="nope"
        )


def test_triangular_group_reparameterisations_independent():
    reps = triangular_group_reparameterisations(
        BNS_PARAMETERS, REFERENCE_TIME, phase_coordinates="independent"
    )
    assert reps["phase"] == {"reparameterisation": "single-angle", "scale": 2.0}
    # psi is untouched here -- left to the default psi_single handling
    assert "psi" not in reps
    assert "arg-alpha-beta" not in reps


@requires_group_mixture
def test_make_et_group_flow_proposal_independent_needs_physical_space():
    with pytest.raises(RuntimeError, match="prime_space=False"):
        make_et_group_flow_proposal(
            BNS_PARAMETERS, REFERENCE_TIME, phase_coordinates="independent",
            prime_space=True,
        )


@requires_group_mixture
def test_make_et_group_flow_proposal_independent_physical_space_builds():
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, phase_coordinates="independent",
        prime_space=False,
    )
    assert cls is not None


def test_prime_parameter_names_arg_alpha_beta():
    prime = _prime_parameter_names(
        BNS_PARAMETERS, REFERENCE_TIME, phase_coordinates="arg-alpha-beta"
    )
    assert "arg_alpha" in prime and "arg_beta" in prime
    assert "delta_phase" not in prime
    assert "psi_x" not in prime and "psi_prime" not in prime


@pytest.fixture(scope="module", params=[False, True],
                ids=["group8", "group32"])
def arg_ab_action(request):
    base = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME,
        polarisation_quarter=request.param,
        phase_reflection=request.param,
    )
    return PrimeSpaceTriangularGroupAction(base, ARG_AB_PRIME_NAMES)


def test_arg_ab_action_detected(arg_ab_action):
    assert arg_ab_action._arg_ab == ("arg_alpha", "arg_beta")


def test_arg_ab_action_inverse_round_trip(arg_ab_action, arg_ab_prime_points):
    n = len(arg_ab_prime_points["ra_dec_x"])
    for g in range(arg_ab_action._action.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        fwd = arg_ab_action(arg_ab_prime_points, modes)
        back = arg_ab_action(fwd, modes, inverse=True)
        for name in ARG_AB_PRIME_NAMES:
            d = (back[name] - arg_ab_prime_points[name]).abs().max()
            assert d < 1e-9, (g, name, float(d))


def test_arg_ab_action_t_det_and_theta_jn(arg_ab_action, arg_ab_prime_points):
    n = len(arg_ab_prime_points["ra_dec_x"])
    for g in range(arg_ab_action._action.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        fwd = arg_ab_action(arg_ab_prime_points, modes)
        assert torch.allclose(fwd["t_det"], arg_ab_prime_points["t_det"])
        assert torch.allclose(
            fwd["theta_jn_prime"].abs(),
            arg_ab_prime_points["theta_jn_prime"].abs(),
        )
        for k in ("arg_alpha", "arg_beta"):
            assert fwd[k].min() >= -1 - 1e-9 and fwd[k].max() < 1 + 1e-9


def test_arg_ab_action_fundamental_domain_partitions_orbit(
    arg_ab_action, arg_ab_prime_points
):
    count = _in_domain_count(
        arg_ab_action, arg_ab_prime_points, arg_ab_action._action.group_size
    )
    assert torch.all(count == 1), count.unique(return_counts=True)


def test_arg_ab_action_matches_physical(arg_ab_action, arg_ab_prime_points):
    """The prime (arg_alpha, arg_beta) action equals the physical action on
    (psi, phase) mapped through arg-alpha-beta."""
    n = len(arg_ab_prime_points["ra_dec_x"])
    acted, _ = arg_ab_action._decode(arg_ab_prime_points)
    phys0 = {
        "ra": acted["ra"],
        "sin_dec": acted["sin_dec"],
        "cos_theta_jn": torch.full_like(acted["ra"], 0.3),
        "psi": acted["psi"],
        "phase": acted["phase"],
    }
    base = arg_ab_action._action
    for g in range(base.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        phys = base(phys0, modes)
        aa, bb = _arg_ab_of(phys["psi"], phys["phase"])
        fwd = arg_ab_action(arg_ab_prime_points, modes)
        for got, want in ((fwd["arg_alpha"], aa), (fwd["arg_beta"], bb)):
            d = torch.remainder(got - want, 2.0)
            d = torch.minimum(d, 2.0 - d)
            assert d.max() < 1e-6, g


def test_prime_parameter_names():
    prime = _prime_parameter_names(BNS_PARAMETERS, REFERENCE_TIME)
    if not any(n.startswith("ra_dec") for n in prime):
        pytest.skip("probe fell back to the name-order heuristic")
    assert {"ra_dec_x", "ra_dec_y", "ra_dec_z"} <= set(prime)
    assert "psi_x" in prime and "psi_y" in prime
    assert "delta_phase" in prime
    assert "delta_phase_x" not in prime and "delta_phase_y" not in prime
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
        polarisation_quarter=False,
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
        convention="ra-dec", rotation=R, radial_sigma=None,
        rng=np.random.default_rng(1),
    )
    n = 500
    x = np.zeros(
        n, dtype=[(p, "f8") for p in plain.parameters + plain.auxiliary_parameters]
    )
    x["ra"] = rng.uniform(0, 2 * np.pi, n)
    x["dec"] = np.arcsin(rng.uniform(-1, 1, n))
    x[plain.auxiliary_parameters[0]] = rng.uniform(0.5, 1.5, n)
    dt = [(p, "f8") for p in plain.output_parameters]
    xp_a = np.zeros(n, dtype=dt)
    xp_b = np.zeros(n, dtype=dt)
    _, xp_a, _ = plain.reparameterise(x.copy(), xp_a, np.zeros(n))
    _, xp_b, _ = rot.reparameterise(x.copy(), xp_b, np.zeros(n))
    va = np.stack([xp_a[p] for p in plain.output_parameters], axis=-1)
    vb = np.stack([xp_b[p] for p in rot.output_parameters], axis=-1)
    np.testing.assert_allclose(vb, va @ R.T, atol=1e-9)

    back = np.zeros(
        n, dtype=[(p, "f8") for p in rot.parameters + rot.auxiliary_parameters]
    )
    back, _, _ = rot.inverse_reparameterise(back, xp_b, np.zeros(n))
    np.testing.assert_allclose(np.cos(back["ra"]), np.cos(x["ra"]), atol=1e-6)
    np.testing.assert_allclose(np.sin(back["ra"]), np.sin(x["ra"]), atol=1e-6)
    np.testing.assert_allclose(back["dec"], x["dec"], atol=1e-6)


def test_rotated_anglepair_radial_shell():
    """The default concentrated radius keeps the prime points in a unit shell
    (no mass tapering toward the origin); ``radial_sigma=None`` restores chi(3).
    """
    from nessai_gw.reparameterisations.sky import RotatedAnglePair

    R = ETTriangleGroupAction(reference_time=REFERENCE_TIME).sky_frame_rotation
    bounds = {
        "ra": np.array([0.0, 2 * np.pi]),
        "dec": np.array([-np.pi / 2, np.pi / 2]),
    }
    rng = np.random.default_rng(0)
    n = 20000
    x = np.zeros(n, dtype=[("ra", "f8"), ("dec", "f8"), ("ra_dec_radial", "f8")])
    x["ra"] = rng.uniform(0, 2 * np.pi, n)
    x["dec"] = np.arcsin(rng.uniform(-1, 1, n))

    def radii(radial_sigma):
        rep = RotatedAnglePair(
            parameters=["ra", "dec"], prior_bounds=dict(bounds),
            convention="ra-dec", rotation=R, radial_sigma=radial_sigma,
            rng=np.random.default_rng(1),
        )
        xp = np.zeros(n, dtype=[(p, "f8") for p in rep.output_parameters])
        _, xp, _ = rep.reparameterise(x.copy(), xp, np.zeros(n))
        v = np.stack([xp[p] for p in rep.output_parameters], axis=-1)
        return np.linalg.norm(v, axis=1)

    shell = radii(0.15)
    assert shell.mean() == pytest.approx(1.0, abs=0.02)
    assert shell.std() == pytest.approx(0.15, abs=0.02)
    assert shell.min() > 0.4  # nothing near the origin

    chi3 = radii(None)
    assert chi3.std() > 0.4  # the wide chi(3) spread
    assert (chi3 < 0.7).mean() > 0.05  # real mass tapering toward r=0


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


def test_recommended_polarisation_offset_centres_the_seam():
    """A psi blob straddling the pi/2 seam: the recommended offset moves its
    folded circular mean to pi/4, and re-measuring confirms it."""
    rng = np.random.default_rng(1)
    # centred near the pi/2 seam itself -- the worst case for the default
    # (unshifted) fold, exactly what motivates this offset.
    psi = np.mod(rng.normal(0.5 * np.pi, 0.08, 4000), np.pi)
    d0 = recommended_polarisation_offset(psi)
    assert d0["concentration"] > 0.8
    offset = d0["recommended_azimuth_offset"]
    psi_shifted = np.mod(psi - offset, np.pi)
    d1 = recommended_polarisation_offset(psi_shifted)
    assert d1["circ_mean_lambda"] == pytest.approx(np.pi / 4, abs=0.02)
    assert d1["recommended_azimuth_offset"] == pytest.approx(0.0, abs=0.02)


def test_recommended_polarisation_offset_low_concentration_when_spread_out():
    rng = np.random.default_rng(2)
    psi = rng.uniform(0, np.pi, 4000)
    d = recommended_polarisation_offset(psi)
    assert d["concentration"] < 0.2


def test_polarisation_offset_moves_the_seam_not_the_group_action():
    """Every physical point still has exactly one canonical image among the
    32, for any polarisation_offset -- the offset only relabels which image
    that is, it must not break exclusivity/coverage of the fold."""
    rng = np.random.default_rng(3)
    n = 500
    ra = torch.as_tensor(rng.uniform(0, 2 * np.pi, n))
    dec = torch.asin(torch.as_tensor(rng.uniform(-1, 1, n)))
    psi = torch.as_tensor(rng.uniform(0, np.pi, n))
    cos_theta_jn = torch.as_tensor(rng.uniform(-1, 1, n))
    phase = torch.as_tensor(rng.uniform(0, 2 * np.pi, n))
    point = dict(ra=ra, sin_dec=torch.sin(dec), psi=psi,
                cos_theta_jn=cos_theta_jn, phase=phase)

    for offset in (0.0, 0.4, -0.9, np.pi / 2, 2.3):
        action = ETTriangleGroupAction(
            reference_time=REFERENCE_TIME, polarisation_quarter=True,
            polarisation_offset=offset,
        )
        seen = torch.zeros(n, dtype=torch.bool)
        for g in range(action.group_size):
            modes = torch.full((n,), g, dtype=torch.long)
            image = action(point, modes)
            canon = action.in_fundamental_domain(image)
            # exactly one image per point should be canonical
            assert not torch.any(canon & seen), offset
            seen |= canon
        assert torch.all(seen), offset


def test_polarisation_offset_default_matches_unshifted_behaviour():
    action0 = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, polarisation_quarter=True,
    )
    action1 = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, polarisation_quarter=True,
        polarisation_offset=0.0,
    )
    assert action0.polarisation_offset == action1.polarisation_offset == 0.0


@requires_group_mixture
@pytest.mark.parametrize("boundary_reflection", [False, True])
def test_make_et_group_flow_proposal_reflection_and_offset(boundary_reflection):
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS,
        REFERENCE_TIME,
        azimuth_offset=0.3,
        boundary_reflection=boundary_reflection,
        gaussianise_sky=False,  # tested in isolation; it supersedes reflection
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


# --------------------------------------------------------------------------
# SkyOctantGaussianiser (canonical_transform)

from nessai_gw.group_mixture import SkyOctantGaussianiser  # noqa: E402


def _octant_canon(n, seed=7):
    """``[n, len(PRIME_NAMES)]`` canon tensor, sky on the +++ octant shell."""
    rng = np.random.default_rng(seed)
    v = np.abs(rng.normal(size=(n, 3)))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rng.uniform(0.7, 1.3, n)
    v *= r[:, None]
    cols = {k: torch.zeros(n, dtype=torch.float64) for k in PRIME_NAMES}
    cols["ra_dec_x"] = torch.as_tensor(v[:, 0])
    cols["ra_dec_y"] = torch.as_tensor(v[:, 1])
    cols["ra_dec_z"] = torch.as_tensor(v[:, 2])
    for k in ("psi_x", "psi_y", "delta_phase", "theta_jn_prime"):
        cols[k] = torch.as_tensor(rng.normal(size=n))
    return torch.stack([cols[k] for k in PRIME_NAMES], dim=-1)


def test_sky_octant_gaussianiser_roundtrip():
    t = SkyOctantGaussianiser(PRIME_NAMES)
    canon = _octant_canon(512)
    base, ljf = t.forward(canon)
    back, lji = t.inverse(base)
    assert torch.allclose(back, canon, atol=1e-6)
    assert torch.allclose(ljf, -lji, atol=1e-6)
    # non-sky columns are untouched
    for i, name in enumerate(PRIME_NAMES):
        if not name.startswith("ra_dec"):
            assert torch.allclose(base[:, i], canon[:, i])


def test_sky_octant_gaussianiser_jacobian_numeric():
    t = SkyOctantGaussianiser(PRIME_NAMES)
    canon = _octant_canon(64, seed=1)
    _, ljf = t.forward(canon)
    idx = [PRIME_NAMES.index(f"ra_dec_{c}") for c in ("x", "y", "z")]
    eps = 1e-6
    jac = torch.zeros(canon.shape[0], 3, 3, dtype=torch.float64)
    for j, col in enumerate(idx):
        for s in (+1, -1):
            pert = canon.clone()
            pert[:, col] += s * eps
            fp, _ = t.forward(pert)
            jac[:, :, j] += s * fp[:, idx] / (2 * eps)
    logdet = torch.log(torch.linalg.det(jac).abs())
    assert torch.allclose(ljf, logdet, atol=1e-4)


def test_sky_octant_gaussianiser_gaussianises_uniform_prior():
    rng = np.random.default_rng(0)
    v = np.abs(rng.normal(size=(200_000, 3)))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    canon = torch.zeros(v.shape[0], len(PRIME_NAMES), dtype=torch.float64)
    canon[:, 0] = torch.as_tensor(v[:, 0])
    canon[:, 1] = torch.as_tensor(v[:, 1])
    canon[:, 2] = torch.as_tensor(v[:, 2])
    base, _ = SkyOctantGaussianiser(PRIME_NAMES).forward(canon)
    a, b = base[:, 0].numpy(), base[:, 1].numpy()
    assert abs(a.mean()) < 0.02 and abs(b.mean()) < 0.02
    assert abs(a.std() - 1.0) < 0.02 and abs(b.std() - 1.0) < 0.02
    # independent
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.02


@requires_group_mixture
def test_gaussianise_sky_on_by_default_for_prime_space():
    # default sky_2d="auto" resolves on -> SkyOctantProbit
    on = make_et_group_flow_proposal(BNS_PARAMETERS, REFERENCE_TIME)
    assert isinstance(
        getattr(on._FlowModelClass, "canonical_transform", None),
        SkyOctantProbit,
    )
    # 3-coordinate sky still uses the Gaussianiser
    on3 = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, sky_2d=False
    )
    assert isinstance(
        getattr(on3._FlowModelClass, "canonical_transform", None),
        SkyOctantGaussianiser,
    )
    off = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, prime_space=False
    )
    assert getattr(off._FlowModelClass, "canonical_transform", None) is None
    opt_out = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, gaussianise_sky=False
    )
    assert getattr(opt_out._FlowModelClass, "canonical_transform", None) is None


@requires_group_mixture
def test_make_et_group_flow_proposal_gaussianise_sky():
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, gaussianise_sky=True, sky_2d=False
    )
    ct = getattr(cls._FlowModelClass, "canonical_transform", None)
    assert isinstance(ct, SkyOctantGaussianiser)
    # supersedes boundary_reflection for the sky
    cls2 = make_et_group_flow_proposal(
        BNS_PARAMETERS,
        REFERENCE_TIME,
        gaussianise_sky=True,
        sky_2d=False,
        boundary_reflection=True,
    )
    assert getattr(cls2._FlowModelClass, "reflect_parameters", None) is None


@requires_group_mixture
def test_gaussianise_sky_requires_prime_space():
    with pytest.raises(RuntimeError, match="prime-space"):
        make_et_group_flow_proposal(
            BNS_PARAMETERS,
            REFERENCE_TIME,
            prime_space=False,
            gaussianise_sky=True,
        )


# ---------------------------------------------------------------------------
# 2-D equal-area sky path (EqualAreaSky + SkyOctantProbit + sky_2d wiring)
# ---------------------------------------------------------------------------
from nessai_gw.group_mixture import (  # noqa: E402
    SkyOctantProbit,
    SkyOctantPullback,
)
from nessai_gw.reparameterisations.sky import EqualAreaSky  # noqa: E402

PRIME_NAMES_2D = [
    "sky_u",
    "sky_v",
    "psi_x",
    "psi_y",
    "delta_phase",
    "theta_jn_prime",
    "t_det",
]


def _et_action(phase_reflection=False):
    return ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, phase_reflection=phase_reflection
    )


def _sky_rotation():
    return _et_action().sky_frame_rotation


def test_equal_area_sky_round_trip_and_uniform_prior():
    from nessai.livepoint import empty_structured_array

    rep = EqualAreaSky(
        parameters=["ra", "dec"],
        prior_bounds={"ra": [0.0, 2 * np.pi], "dec": [-np.pi / 2, np.pi / 2]},
        rotation=_sky_rotation(),
    )
    assert rep.prime_parameters == ["sky_u", "sky_v"]

    rng = np.random.default_rng(0)
    n = 20000
    x = empty_structured_array(n, names=["ra", "dec"])
    x["ra"] = rng.uniform(0, 2 * np.pi, n)
    x["dec"] = np.arcsin(rng.uniform(-1, 1, n))
    xp = empty_structured_array(n, names=["sky_u", "sky_v"])
    x, xp, lj = rep.reparameterise(x, xp, np.zeros(n))

    # uniform-on-sphere prior -> (u, v) uniform on the unit square, independent
    for c in ("sky_u", "sky_v"):
        assert xp[c].min() >= 0 and xp[c].max() <= 1
        assert abs(xp[c].mean() - 0.5) < 0.02
        assert abs(xp[c].std() - np.sqrt(1 / 12)) < 0.01
    assert abs(np.corrcoef(xp["sky_u"], xp["sky_v"])[0, 1]) < 0.03

    x2 = empty_structured_array(n, names=["ra", "dec"])
    x2, xp, lj2 = rep.inverse_reparameterise(x2, xp.copy(), np.zeros(n))
    dra = np.abs(x2["ra"] - x["ra"])
    dra = np.minimum(dra, 2 * np.pi - dra)
    assert dra.max() < 1e-9
    assert np.abs(x2["dec"] - x["dec"]).max() < 1e-9
    assert np.allclose(lj, -lj2, atol=1e-9)


def _folded_sub_square(n, seed=3):
    """canon tensor with sky_u in [0, 1/4), sky_v in [0, 1/2) (fundamental)."""
    rng = np.random.default_rng(seed)
    cols = {k: torch.as_tensor(rng.normal(size=n)) for k in PRIME_NAMES_2D}
    cols["sky_u"] = torch.as_tensor(rng.uniform(0, 0.25, n))
    cols["sky_v"] = torch.as_tensor(rng.uniform(0, 0.5, n))
    return torch.stack([cols[k] for k in PRIME_NAMES_2D], dim=-1)


def test_sky_octant_probit_round_trip_and_jacobian():
    t = SkyOctantProbit(PRIME_NAMES_2D)
    canon = _folded_sub_square(256)
    base, ljf = t.forward(canon)
    back, lji = t.inverse(base)
    assert torch.allclose(back, canon, atol=1e-6)
    assert torch.allclose(ljf, -lji, atol=1e-6)
    for i, name in enumerate(PRIME_NAMES_2D):
        if name not in ("sky_u", "sky_v"):
            assert torch.allclose(base[:, i], canon[:, i])
    # numeric Jacobian of the 2x2 sky block
    iu, iv = PRIME_NAMES_2D.index("sky_u"), PRIME_NAMES_2D.index("sky_v")
    eps = 1e-6
    jac = torch.zeros(canon.shape[0], 2, 2, dtype=torch.float64)
    for j, col in enumerate((iu, iv)):
        for s in (+1, -1):
            pert = canon.clone()
            pert[:, col] += s * eps
            fp, _ = t.forward(pert)
            jac[:, :, j] += s * fp[:, [iu, iv]] / (2 * eps)
    logdet = torch.log(torch.linalg.det(jac).abs())
    assert torch.allclose(ljf, logdet, atol=1e-4)


def _bumpy_q_grid(nu=48, nv=48):
    u = (np.arange(nu) + 0.5) / nu
    v = (np.arange(nv) + 0.5) / nv
    U, V = np.meshgrid(u, v, indexing="ij")
    return 0.3 + np.exp(-((U - 0.6) ** 2 + (V - 0.35) ** 2) / 0.02)


def test_sky_octant_pullback_round_trip_and_jacobian():
    t = SkyOctantPullback(_bumpy_q_grid(), PRIME_NAMES_2D)
    canon = _folded_sub_square(256)
    base, ljf = t.forward(canon)
    back, lji = t.inverse(base)
    assert torch.allclose(back, canon, atol=1e-6)
    assert torch.allclose(ljf, -lji, atol=1e-6)
    for i, name in enumerate(PRIME_NAMES_2D):
        if name not in ("sky_u", "sky_v"):
            assert torch.allclose(base[:, i], canon[:, i])
    iu, iv = PRIME_NAMES_2D.index("sky_u"), PRIME_NAMES_2D.index("sky_v")
    eps = 1e-6
    jac = torch.zeros(canon.shape[0], 2, 2, dtype=torch.float64)
    for j, col in enumerate((iu, iv)):
        for s in (+1, -1):
            pert = canon.clone()
            pert[:, col] += s * eps
            fp, _ = t.forward(pert)
            jac[:, :, j] += s * fp[:, [iu, iv]] / (2 * eps)
    logdet = torch.log(torch.linalg.det(jac).abs())
    # piecewise-constant density vs finite-diff of a piecewise-linear map:
    # agree except for the ~1/nu fraction of points straddling a cell wall.
    assert (torch.abs(ljf - logdet) < 1e-3).float().mean() > 0.9


def test_sky_octant_pullback_uniform_q_matches_probit():
    """A flat q_grid must reduce to SkyOctantProbit."""
    canon = _folded_sub_square(512)
    flat = SkyOctantPullback(np.ones((32, 32)), PRIME_NAMES_2D)
    ref = SkyOctantProbit(PRIME_NAMES_2D)
    bf, ljf = flat.forward(canon.clone())
    br, ljr = ref.forward(canon.clone())
    assert torch.allclose(bf, br, atol=1e-6)
    assert torch.allclose(ljf, ljr, atol=1e-6)


def test_sky_octant_pullback_gaussianises_its_own_q():
    """Sampling proportional to q_grid, the pullback yields ~N(0, 1)^2."""
    q = _bumpy_q_grid(64, 64)
    nu, nv = q.shape
    rng = np.random.default_rng(1)
    flat_p = (q / q.sum()).ravel()
    idx = rng.choice(flat_p.size, size=150_000, p=flat_p)
    iu, iv = np.unravel_index(idx, (nu, nv))
    su = (iu + rng.uniform(size=idx.size)) * 0.25 / nu
    sv = (iv + rng.uniform(size=idx.size)) * 0.5 / nv
    canon = torch.zeros(idx.size, len(PRIME_NAMES_2D), dtype=torch.float64)
    canon[:, 0] = torch.as_tensor(su)
    canon[:, 1] = torch.as_tensor(sv)
    base, _ = SkyOctantPullback(q, PRIME_NAMES_2D).forward(canon)
    a, b = base[:, 0].numpy(), base[:, 1].numpy()
    assert abs(a.mean()) < 0.03 and abs(b.mean()) < 0.03
    assert abs(a.std() - 1.0) < 0.03 and abs(b.std() - 1.0) < 0.03


def test_sky_octant_probit_gaussianises_folded_prior():
    rng = np.random.default_rng(0)
    n = 200_000
    canon = torch.zeros(n, len(PRIME_NAMES_2D), dtype=torch.float64)
    canon[:, 0] = torch.as_tensor(rng.uniform(0, 0.25, n))  # sky_u
    canon[:, 1] = torch.as_tensor(rng.uniform(0, 0.5, n))   # sky_v
    base, _ = SkyOctantProbit(PRIME_NAMES_2D).forward(canon)
    a, b = base[:, 0].numpy(), base[:, 1].numpy()
    assert abs(a.mean()) < 0.02 and abs(b.mean()) < 0.02
    assert abs(a.std() - 1.0) < 0.02 and abs(b.std() - 1.0) < 0.02
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.02


def test_prime_space_action_2d_sky_matches_3d(prime_points):
    """The 2-D equal-area sky action must be physically identical to the 3-D one."""
    R = _sky_rotation()
    base3 = _et_action(phase_reflection=True)
    a3 = PrimeSpaceTriangularGroupAction(base3, PRIME_NAMES)
    a2 = PrimeSpaceTriangularGroupAction(base3, PRIME_NAMES_2D)
    assert a2._sky2d == ("sky_u", "sky_v")

    # recover (ra, sin_dec) that prime_points encodes, then build the 2-D input
    acted3, _ = a3._decode(prime_points)
    ra, sin_dec = acted3["ra"], acted3["sin_dec"]
    dec = torch.asin(torch.clamp(sin_dec, -1.0, 1.0))
    cd = torch.cos(dec)
    we = torch.stack([cd * torch.cos(ra), cd * torch.sin(ra), sin_dec])
    Rt = torch.as_tensor(R)
    wd = Rt @ we
    lam = torch.remainder(torch.atan2(wd[1], wd[0]), 2 * np.pi)
    z = torch.clamp(wd[2], -1.0, 1.0)
    pts2 = {k: prime_points[k] for k in PRIME_NAMES_2D if k in prime_points}
    pts2["sky_u"] = lam / (2 * np.pi)
    pts2["sky_v"] = 0.5 * (1.0 - z)

    for g in range(16):
        modes = torch.full((ra.shape[0],), g)
        m3 = a3(prime_points, modes)
        m2 = a2(pts2, modes)
        # compare the physical sky the two encodings imply
        d3, _ = a3._decode(m3)
        e2, _ = a2._decode(m2)
        dra = torch.remainder(d3["ra"] - e2["ra"], 2 * np.pi)
        dra = torch.minimum(dra, 2 * np.pi - dra)
        assert dra.abs().max() < 1e-6, g
        assert (d3["sin_dec"] - e2["sin_dec"]).abs().max() < 1e-6, g
        # delta_phase / psi / theta_jn coordinates must match exactly
        for c in ("delta_phase", "psi_x", "psi_y", "theta_jn_prime", "t_det"):
            assert torch.allclose(m3[c], m2[c], atol=1e-6), (g, c)


@requires_group_mixture
def test_make_et_group_flow_proposal_sky_2d():
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, sky_2d=True
    )
    ct = getattr(cls._FlowModelClass, "canonical_transform", None)
    assert isinstance(ct, SkyOctantProbit)
    names = list(cls._FlowModelClass.param_names)
    assert "sky_u" in names and "sky_v" in names
    assert not any(n.startswith("ra_dec") for n in names)
    # one fewer flow dimension than the 3-coordinate path
    base = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, sky_2d=False
    )
    assert any(n.startswith("ra_dec") for n in base._FlowModelClass.param_names)
    assert len(names) == len(base._FlowModelClass.param_names) - 1
    # on by default (prime-space + gaussianise-sky both active)
    dflt = make_et_group_flow_proposal(BNS_PARAMETERS, REFERENCE_TIME)
    assert isinstance(
        getattr(dflt._FlowModelClass, "canonical_transform", None),
        SkyOctantProbit,
    )


@requires_group_mixture
def test_sky_2d_requires_gaussianise_and_prime_space():
    with pytest.raises(RuntimeError, match="prime-space"):
        make_et_group_flow_proposal(
            BNS_PARAMETERS, REFERENCE_TIME, prime_space=False, sky_2d=True
        )
    with pytest.raises(RuntimeError, match="gaussianise_sky"):
        make_et_group_flow_proposal(
            BNS_PARAMETERS, REFERENCE_TIME, sky_2d=True, gaussianise_sky=False
        )


# ---------------------------------------------------------------------------
# single psi coordinate (SingleAngleReparameterisation + psi_single wiring)
# ---------------------------------------------------------------------------
from nessai_gw.reparameterisations.phase import (  # noqa: E402
    SingleAngleReparameterisation,
)

PRIME_NAMES_PSI1 = [
    "ra_dec_x",
    "ra_dec_y",
    "ra_dec_z",
    "psi_prime",
    "delta_phase",
    "theta_jn_prime",
    "t_det",
]


def test_single_angle_reparameterisation_round_trip():
    from nessai.livepoint import empty_structured_array

    rep = SingleAngleReparameterisation(
        parameters=["psi"], prior_bounds={"psi": [0.0, np.pi]}, scale=2.0
    )
    assert rep.prime_parameters == ["psi_prime"]
    rng = np.random.default_rng(0)
    n = 10000
    x = empty_structured_array(n, names=["psi"])
    x["psi"] = rng.uniform(0, np.pi, n)
    xp = empty_structured_array(n, names=["psi_prime"])
    x, xp, lj = rep.reparameterise(x, xp, np.zeros(n))
    assert xp["psi_prime"].min() >= -1 and xp["psi_prime"].max() < 1
    assert abs(xp["psi_prime"].mean()) < 0.03  # ~uniform
    assert np.allclose(lj, np.log(2.0 / np.pi))
    x2 = empty_structured_array(n, names=["psi"])
    x2, xp, lj2 = rep.inverse_reparameterise(x2, xp.copy(), np.zeros(n))
    dpsi = np.abs(x2["psi"] - x["psi"])
    dpsi = np.minimum(dpsi, np.pi - dpsi)
    assert dpsi.max() < 1e-9
    assert np.allclose(lj, -lj2)


def test_prime_space_action_single_psi_matches_pair(prime_points):
    base = _et_action(phase_reflection=True)
    a_pair = PrimeSpaceTriangularGroupAction(base, PRIME_NAMES)
    a_single = PrimeSpaceTriangularGroupAction(base, PRIME_NAMES_PSI1)
    assert a_single._psi_single and not a_pair._psi_single

    # recover physical psi from the pair, build the single-coord input
    acted, _ = a_pair._decode(prime_points)
    psi = acted["psi"]
    pts1 = {k: prime_points[k] for k in PRIME_NAMES_PSI1 if k in prime_points}
    pts1["psi_prime"] = (
        torch.remainder(psi * 2.0, 2 * np.pi) / np.pi - 1.0
    )
    for g in range(16):
        modes = torch.full((psi.shape[0],), g)
        m_pair = a_pair(prime_points, modes)
        m_single = a_single(pts1, modes)
        dp, _ = a_pair._decode(m_pair)
        ds, _ = a_single._decode(m_single)
        dpsi = torch.remainder(dp["psi"] - ds["psi"], np.pi)
        dpsi = torch.minimum(dpsi, np.pi - dpsi)
        assert dpsi.abs().max() < 1e-6, g
        assert torch.allclose(m_pair["delta_phase"], m_single["delta_phase"],
                              atol=1e-6), g


@requires_group_mixture
def test_make_et_group_flow_proposal_psi_single():
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, psi_single=True, sky_2d=False
    )
    names = list(cls._FlowModelClass.param_names)
    assert "psi_prime" in names
    assert "psi_x" not in names and "psi_y" not in names
    base = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, psi_single=False, sky_2d=False
    )
    assert "psi_x" in base._FlowModelClass.param_names
    assert len(names) == len(base._FlowModelClass.param_names) - 1
    # on by default
    dflt = list(
        make_et_group_flow_proposal(
            BNS_PARAMETERS, REFERENCE_TIME
        )._FlowModelClass.param_names
    )
    assert "psi_prime" in dflt
    # combines with sky_2d
    both = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, psi_single=True, sky_2d=True
    )
    bn = list(both._FlowModelClass.param_names)
    assert "psi_prime" in bn and "sky_u" in bn
    assert len(bn) == len(base._FlowModelClass.param_names) - 2


# ---------------------------------------------------------------------------
# 32-element group: the {psi+pi/2, phase-pi/2} polarisation/phase quarter turn
# ---------------------------------------------------------------------------
def test_polarisation_quarter_group_size_and_decode():
    a = _et_action()
    assert a.group_size == 8
    a16 = _et_action(phase_reflection=True)
    assert a16.group_size == 16
    aq = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, polarisation_quarter=True
    )
    assert aq.group_size == 32 and aq.phase_reflection  # implies phase_reflection
    modes = torch.arange(32)
    k, refl, step = aq._decode(modes)
    assert torch.equal(step, torch.div(modes, 8, rounding_mode="floor"))
    # every element composed with its inverse is the identity, inverse is a perm
    inv = aq.invert_modes(modes)
    assert torch.equal(aq.invert_modes(inv), modes)
    assert sorted(inv.tolist()) == list(range(32))


def _pq_prime_action():
    base = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, polarisation_quarter=True
    )
    return PrimeSpaceTriangularGroupAction(base, PRIME_NAMES)


def test_polarisation_quarter_prime_action_round_trip(prime_points):
    a = _pq_prime_action()
    for g in range(32):
        modes = torch.full((prime_points["ra_dec_x"].shape[0],), g)
        fwd = a(prime_points, modes, inverse=False)
        back = a(fwd, modes, inverse=True)
        for c in PRIME_NAMES:
            assert torch.allclose(
                torch.as_tensor(back[c]), prime_points[c], atol=1e-6
            ), (g, c)


def test_polarisation_quarter_fundamental_domain_partitions_orbit(prime_points):
    a = _pq_prime_action()
    n = prime_points["ra_dec_x"].shape[0]
    counts = torch.zeros(n)
    for g in range(32):
        modes = torch.full((n,), g)
        img = a(prime_points, modes, inverse=True)
        counts += a.in_fundamental_domain(img).float()
    assert torch.all(counts == 1)


def test_polarisation_quarter_preserves_physical_symmetry(prime_points):
    """step==2 must reproduce the phase -> phase + pi flip (delta_phase + pi)."""
    a = _pq_prime_action()
    n = prime_points["ra_dec_x"].shape[0]
    m0 = a(prime_points, torch.zeros(n, dtype=torch.long))
    m2 = a(prime_points, torch.full((n,), 16))  # step = 16 // 8 = 2
    d = torch.remainder(
        (m2["delta_phase"] - m0["delta_phase"]) - 1.0, 2.0
    )  # delta_phase_prime is in [-1, 1); +pi -> +1 unit
    d = torch.minimum(d, 2.0 - d)
    assert d.abs().max() < 1e-6
    # psi unchanged by step 2 (psi -> psi + pi == psi)
    assert torch.allclose(m2["theta_jn_prime"], m0["theta_jn_prime"], atol=1e-6)


@requires_group_mixture
def test_make_et_group_flow_proposal_polarisation_quarter():
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, polarisation_quarter=True
    )
    fm = cls._FlowModelClass
    assert fm.group_size == 32
    # on by default
    dflt = make_et_group_flow_proposal(BNS_PARAMETERS, REFERENCE_TIME)
    assert dflt._FlowModelClass.group_size == 32
    off = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, polarisation_quarter=False,
        phase_reflection=True,
    )
    assert off._FlowModelClass.group_size == 16


@requires_group_mixture
def test_make_et_group_flow_proposal_clustered():
    """n_clusters_max > 1 wires the clustered flow model; default is off."""
    from nessai.flowmodel.group_mixture import (
        ClusteredGroupMixtureFlowModel,
        GroupMixtureFlowModel,
    )

    dflt = make_et_group_flow_proposal(BNS_PARAMETERS, REFERENCE_TIME)
    assert not issubclass(
        dflt._FlowModelClass, ClusteredGroupMixtureFlowModel
    )

    clustered = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, n_clusters_max=3,
        cluster_max_overlap=0.05, cluster_min_size=150,
    )
    fm = clustered._FlowModelClass
    assert issubclass(fm, ClusteredGroupMixtureFlowModel)
    assert issubclass(fm, GroupMixtureFlowModel)
    assert fm.n_clusters_max == 3
    assert fm.max_cluster_overlap == 0.05
    assert fm.min_cluster_size == 150
    # group geometry unchanged
    assert fm.group_size == 32


@requires_group_mixture
def test_reset_model_weights_realigns_prime_parameter_order(tmp_path):
    """Regression test for the ``--reset-flow`` prime-dimension mixup.

    ``TriangularGroupFlowProposal._FlowModelClass.param_names`` starts as
    ``_prime_parameter_names``'s throwaway-probe *prediction* of the real
    proposal's prime-parameter order (the probe uses a plain
    ``GWFlowProposal``, not ``TriangularGroupFlowProposal``, and so cannot
    see this proposal's own ``add_default_reparameterisations`` overrides,
    e.g. ``EqualAreaSky``/``RotatedAnglePair``/``SingleAngleReparameterisation``).
    ``initialise()`` corrects this once, binding the model (and the shared
    prime-space ``action``) to the real ``self.prime_parameters`` order.

    ``FlowModel.reset_model()`` (triggered on every ``--reset-flow`` round)
    rebuilds the model via ``get_model()``, which falls back to the
    *uncorrected* class-attribute order -- unless
    ``reset_model_weights`` re-applies the same correction. This builds a
    real, fully-initialised proposal, deliberately corrupts the predicted
    class-attribute order (simulating a probe/proposal mismatch), resets
    the flow, and asserts the model and action stay correctly bound to the
    live ``prime_parameters`` order rather than reverting to the corrupted
    one.
    """
    from nessai.model import Model

    bounds = {
        name: np.asarray(
            {
                "chirp_mass": (1.0, 2.0),
                "mass_ratio": (0.125, 1.0),
                "chi_1": (-0.99, 0.99),
                "chi_2": (-0.99, 0.99),
                "luminosity_distance": (10.0, 5000.0),
                "theta_jn": (0.0, np.pi),
                "psi": (0.0, np.pi),
                "phase": (0.0, 2 * np.pi),
                "ra": (0.0, 2 * np.pi),
                "dec": (-np.pi / 2, np.pi / 2),
                "geocent_time": (
                    REFERENCE_TIME - 0.1,
                    REFERENCE_TIME + 0.1,
                ),
            }[name],
            dtype=float,
        )
        for name in BNS_PARAMETERS
        if name not in ("lambda_1", "lambda_2")
    }
    names = list(bounds)

    class _StubModel(Model):
        def __init__(self):
            self.names = names
            self.bounds = bounds

        def log_prior(self, x):
            return np.zeros(len(np.atleast_1d(x)))

        def log_likelihood(self, x):
            return np.zeros(len(np.atleast_1d(x)))

    cls = make_et_group_flow_proposal(
        names, REFERENCE_TIME, prime_space=True, n_clusters_max=1,
    )
    from nessai_gw.group_mixture import triangular_group_reparameterisations

    proposal = cls(
        _StubModel(),
        output=str(tmp_path),
        poolsize=100,
        reparameterisations=triangular_group_reparameterisations(
            names, REFERENCE_TIME
        ),
        flow_config={"n_blocks": 1, "n_neurons": 4, "n_layers": 1},
    )
    proposal.initialise()

    model = proposal.flow.model
    prime = list(proposal.prime_parameters)
    assert model.param_names == prime

    # Simulate a probe/proposal order mismatch: corrupt the *predicted*
    # class-attribute order that get_model() falls back to on a reset.
    flow_model_cls = type(model)
    corrupted = list(reversed(prime))
    flow_model_cls.param_names = corrupted
    assert model.param_names == prime  # instance override unaffected yet

    proposal.reset_model_weights(weights=True, permutations=True)

    new_model = proposal.flow.model
    assert new_model is not model  # a fresh instance was built
    assert new_model.param_names == prime
    assert new_model.param_names != corrupted


# ---------------------------------------------------------------------------
# polarisation-ellipse inclination coordinate
# ---------------------------------------------------------------------------
ELLIPSE_FIDUCIAL = dict(ra=3.4462, dec=-0.4081, psi=1.57, theta_jn=0.3491)


@pytest.fixture(scope="module")
def ellipse():
    from nessai_gw._ellipse import PolarisationEllipse
    from nessai_gw.group_mixture import ET_EMR_PLANE_NORMAL

    return PolarisationEllipse(
        ET_EMR_PLANE_NORMAL, REFERENCE_TIME, ELLIPSE_FIDUCIAL
    )


def test_reparameterisations_use_the_ellipse_for_theta_jn(ellipse):
    reps = triangular_group_reparameterisations(
        BNS_PARAMETERS, REFERENCE_TIME, polarisation_ellipse=ellipse
    )
    assert reps["theta_jn"]["reparameterisation"] == "polarisation-ellipse"
    assert reps["theta_jn"]["ellipse"] is ellipse
    assert reps["theta_jn"]["adaptive_width"] is False
    reps_w = triangular_group_reparameterisations(
        BNS_PARAMETERS, REFERENCE_TIME, polarisation_ellipse=ellipse,
        polarisation_ellipse_adaptive_width=True,
    )
    assert reps_w["theta_jn"]["adaptive_width"] is True
    # theta_jn supplies what polarisation-phase consumes on the inverse, and
    # consumes ra/dec itself, so it must sit between phase and the sky.
    order = list(reps)
    assert order.index("phase") < order.index("theta_jn")
    assert order.index("theta_jn") < order.index("geocent_time")


def test_reparameterisations_default_to_angle_sine():
    reps = triangular_group_reparameterisations(BNS_PARAMETERS, REFERENCE_TIME)
    assert reps["theta_jn"]["reparameterisation"] == "angle-sine"


@pytest.mark.parametrize("coordinate", ["angle", "cos"])
def test_prime_action_recovers_sign_cos_theta_jn(ellipse, coordinate):
    """The fold needs sign(cos theta_jn); with a residual coordinate it can no
    longer read it off the sign of the prime, so the ellipse must supply it."""
    action = PrimeSpaceTriangularGroupAction(
        ETTriangleGroupAction(reference_time=REFERENCE_TIME),
        PRIME_NAMES,
        ellipse=ellipse,
        ellipse_coordinate=coordinate,
    )
    rng = np.random.default_rng(11)
    n = 3000
    ra = rng.uniform(0, 2 * np.pi, n)
    dec = np.arcsin(rng.uniform(-1, 1, n))
    theta_jn = np.arccos(rng.uniform(-1, 1, n))
    centre = ellipse.cos_iota(ra, dec)
    if coordinate == "cos":
        prime = np.cos(theta_jn) - centre
    else:
        prime = (theta_jn - np.arccos(np.clip(centre, -1, 1))) * 2.0 / np.pi
    cos_dec = np.cos(dec)
    # the flow's sky coordinates are detector-frame (RotatedAnglePair), so feed
    # v_det = R v_eq -- _decode rotates back with R^T.
    v_eq = np.stack(
        [cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)], axis=-1
    )
    v_det = v_eq @ np.asarray(action._action.sky_frame_rotation).T
    points = {
        "ra_dec_x": torch.as_tensor(v_det[:, 0].copy()),
        "ra_dec_y": torch.as_tensor(v_det[:, 1].copy()),
        "ra_dec_z": torch.as_tensor(v_det[:, 2].copy()),
        "psi_x": torch.as_tensor(np.cos(2.0 * rng.uniform(0, np.pi, n))),
        "psi_y": torch.as_tensor(np.sin(2.0 * rng.uniform(0, np.pi, n))),
        "delta_phase": torch.as_tensor(rng.uniform(-1.0, 1.0, n)),
        "theta_jn_prime": torch.as_tensor(prime),
        "t_det": torch.as_tensor(rng.uniform(-20, 20, n)),
    }
    _, aux = action._decode(points)
    assert np.array_equal(
        aux["sign_ct"].numpy(), np.sign(np.cos(theta_jn))
    )


@pytest.mark.parametrize("coordinate", ["angle", "cos"])
def test_prime_action_sign_cos_theta_jn_with_adaptive_width(coordinate):
    """Sign recovery must survive the heteroskedastic residual: _decode has to
    undo the same s(|cos iota*|) the reparameterisation divided by."""
    from nessai_gw._ellipse import PolarisationEllipse
    from nessai_gw.group_mixture import ET_EMR_PLANE_NORMAL

    e = PolarisationEllipse(
        ET_EMR_PLANE_NORMAL, REFERENCE_TIME, ELLIPSE_FIDUCIAL
    )
    e.set_width([0.0, 0.2, 0.6, 1.0], [0.06, 0.09, 0.3, 0.55])
    action = PrimeSpaceTriangularGroupAction(
        ETTriangleGroupAction(reference_time=REFERENCE_TIME),
        PRIME_NAMES, ellipse=e, ellipse_coordinate=coordinate,
    )
    rng = np.random.default_rng(11)
    n = 3000
    ra = rng.uniform(0, 2 * np.pi, n)
    dec = np.arcsin(rng.uniform(-1, 1, n))
    theta_jn = np.arccos(rng.uniform(-1, 1, n))
    centre = e.cos_iota(ra, dec)
    s = e.residual_width(np.abs(np.clip(centre, -1, 1)))
    if coordinate == "cos":
        prime = (np.cos(theta_jn) - centre) / s
    else:
        prime = (theta_jn - np.arccos(np.clip(centre, -1, 1))) * (2.0 / np.pi) / s
    cos_dec = np.cos(dec)
    v_eq = np.stack(
        [cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)], axis=-1
    )
    v_det = v_eq @ np.asarray(action._action.sky_frame_rotation).T
    points = {
        "ra_dec_x": torch.as_tensor(v_det[:, 0].copy()),
        "ra_dec_y": torch.as_tensor(v_det[:, 1].copy()),
        "ra_dec_z": torch.as_tensor(v_det[:, 2].copy()),
        "psi_x": torch.as_tensor(np.cos(2.0 * rng.uniform(0, np.pi, n))),
        "psi_y": torch.as_tensor(np.sin(2.0 * rng.uniform(0, np.pi, n))),
        "delta_phase": torch.as_tensor(rng.uniform(-1.0, 1.0, n)),
        "theta_jn_prime": torch.as_tensor(prime),
        "t_det": torch.as_tensor(rng.uniform(-20, 20, n)),
    }
    _, aux = action._decode(points)
    assert np.array_equal(
        aux["sign_ct"].numpy(), np.sign(np.cos(theta_jn))
    )


@pytest.mark.parametrize("g", range(ETTriangleGroupAction.group_size))
def test_prime_action_with_ellipse_width_round_trips(prime_points, g):
    """The group action round-trips with a non-trivial residual width set."""
    from nessai_gw._ellipse import PolarisationEllipse
    from nessai_gw.group_mixture import ET_EMR_PLANE_NORMAL

    e = PolarisationEllipse(
        ET_EMR_PLANE_NORMAL, REFERENCE_TIME, ELLIPSE_FIDUCIAL
    )
    e.set_width([0.0, 0.2, 0.6, 1.0], [0.1, 0.14, 0.35, 0.6])
    action = PrimeSpaceTriangularGroupAction(
        ETTriangleGroupAction(reference_time=REFERENCE_TIME),
        PRIME_NAMES, ellipse=e,
    )
    modes = torch.full(
        (prime_points["t_det"].shape[0],), g, dtype=torch.long
    )
    moved = action(prime_points, modes)
    back = action(moved, modes, inverse=True)
    for name in PRIME_NAMES:
        assert torch.allclose(back[name], prime_points[name], atol=1e-8), name


@pytest.mark.parametrize("g", range(ETTriangleGroupAction.group_size))
def test_prime_action_with_ellipse_round_trips(ellipse, prime_points, g):
    action = PrimeSpaceTriangularGroupAction(
        ETTriangleGroupAction(reference_time=REFERENCE_TIME),
        PRIME_NAMES,
        ellipse=ellipse,
    )
    modes = torch.full((prime_points["t_det"].shape[0],), g, dtype=torch.long)
    moved = action(prime_points, modes)
    back = action(moved, modes, inverse=True)
    for name in PRIME_NAMES:
        assert torch.allclose(
            back[name], prime_points[name], atol=1e-8
        ), name
