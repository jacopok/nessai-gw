"""Tests for the ET group-mixture proposal wiring in :mod:`nessai_gw.group_mixture`.

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
    PrimeSpaceETGroupAction,
    _prime_parameter_names,
    et_group_reparameterisations,
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


def test_et_group_reparameterisations():
    reps = et_group_reparameterisations(BNS_PARAMETERS, REFERENCE_TIME)
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


@pytest.fixture(scope="module")
def prime_action():
    base = ETTriangleGroupAction(reference_time=REFERENCE_TIME)
    return PrimeSpaceETGroupAction(base, PRIME_NAMES)


def test_prime_space_action_missing_coordinate():
    base = ETTriangleGroupAction(reference_time=REFERENCE_TIME)
    with pytest.raises(RuntimeError, match="prime space is missing"):
        PrimeSpaceETGroupAction(base, ["ra_dec_x", "ra_dec_y", "ra_dec_z"])


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


def _physical_phase(points):
    """phase = delta_phase - psi from a prime-space dict."""
    psi, _ = _decode_pair(points["psi_x"], points["psi_y"], 2.0)
    dphase, _ = _decode_pair(
        points["delta_phase_x"], points["delta_phase_y"], 1.0
    )
    return torch.remainder(dphase - psi, 2 * np.pi)


def test_prime_space_action_physical_phase_invariant(prime_action, prime_points):
    """The group-invariant physical ``phase`` is unchanged by every element."""
    n = len(prime_points["ra_dec_x"])
    phase0 = _physical_phase(prime_points)
    for g in range(ETTriangleGroupAction.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        mapped = prime_action(prime_points, modes)
        d = torch.remainder(_physical_phase(mapped) - phase0, 2 * np.pi)
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
    """delta_phase_out from the adapter == phase' + psi' from the raw action."""
    n = len(prime_points["ra_dec_x"])
    base = prime_action._action
    acted, _ = prime_action._decode(prime_points)
    for g in range(ETTriangleGroupAction.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        phys = base(acted, modes)
        want = torch.remainder(phys["phase"] + phys["psi"], 2 * np.pi)
        mapped = prime_action(prime_points, modes)
        got, _ = _decode_pair(
            mapped["delta_phase_x"], mapped["delta_phase_y"], 1.0
        )
        d = torch.remainder(got - want, 2 * np.pi)
        d = torch.minimum(d, 2 * np.pi - d)
        assert d.max() < 1e-5, g


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
def test_make_et_group_flow_proposal(prime_space):
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, prime_space=prime_space
    )
    from nessai.proposal import FlowProposal
    from nessai.flowmodel.group_mixture import GroupFlowProposalMixin
    from nessai_gw.proposals import GWReparamMixin

    assert cls.__qualname__ == "ETGroupFlowProposal"
    assert cls.__module__ == "nessai_gw.group_mixture"
    assert issubclass(cls, (GWReparamMixin, GroupFlowProposalMixin, FlowProposal))
    mro = cls.__mro__
    assert mro.index(GWReparamMixin) < mro.index(GroupFlowProposalMixin)
    assert mro.index(GroupFlowProposalMixin) < mro.index(FlowProposal)
    assert cls._FlowModelClass is not None
    # class is resolvable by module.qualname (nessai checkpoints via pickle)
    import nessai_gw.group_mixture as gm

    assert getattr(gm, "ETGroupFlowProposal") is cls


@requires_group_mixture
def test_make_et_group_flow_proposal_missing_parameter():
    with pytest.raises(RuntimeError, match="missing"):
        make_et_group_flow_proposal(["ra", "dec", "psi"], REFERENCE_TIME)
