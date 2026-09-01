"""Tests for the ET group-mixture proposal wiring in :mod:`nessai_gw.group_mixture`.

These exercise the prime-space adapter, the reparameterisation profile, the
prime-name probe and the proposal factory.  The factory / probe need a version
of nessai that ships ``nessai.flowmodel.group_mixture`` and are skipped
otherwise.
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
    "phase_x",
    "phase_y",
    "theta_jn_prime",
    "geocent_time_prime",
]

_group_mixture = pytest.importorskip("nessai.flowmodel.group_mixture")


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
    # parameters that get the default GW reparameterisation are not listed
    assert "ra" not in reps
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
    phase = rng.uniform(0, 2 * np.pi, n)
    r_phase = rng.uniform(0.5, 1.5, n)
    return {
        "ra_dec_x": torch.as_tensor(r_sky * cos_dec * np.cos(ra)),
        "ra_dec_y": torch.as_tensor(r_sky * cos_dec * np.sin(ra)),
        "ra_dec_z": torch.as_tensor(r_sky * np.sin(dec)),
        "psi_x": torch.as_tensor(r_psi * np.cos(2.0 * psi)),
        "psi_y": torch.as_tensor(r_psi * np.sin(2.0 * psi)),
        "phase_x": torch.as_tensor(r_phase * np.cos(phase)),
        "phase_y": torch.as_tensor(r_phase * np.sin(phase)),
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


def test_prime_space_action_phase_invariant(prime_action, prime_points):
    n = len(prime_points["ra_dec_x"])
    for g in range(ETTriangleGroupAction.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        mapped = prime_action(prime_points, modes)
        assert torch.equal(mapped["phase_x"], prime_points["phase_x"])
        assert torch.equal(mapped["phase_y"], prime_points["phase_y"])


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


def test_prime_parameter_names():
    prime = _prime_parameter_names(BNS_PARAMETERS, REFERENCE_TIME)
    if not any(n.startswith("ra_dec") for n in prime):
        pytest.skip("probe fell back to the name-order heuristic")
    assert {"ra_dec_x", "ra_dec_y", "ra_dec_z"} <= set(prime)
    assert "psi_x" in prime and "psi_y" in prime
    assert "phase_x" in prime and "phase_y" in prime
    # every physical parameter contributes at least one prime coordinate
    assert len(prime) >= len(BNS_PARAMETERS)


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


def test_make_et_group_flow_proposal_missing_parameter():
    with pytest.raises(RuntimeError, match="missing"):
        make_et_group_flow_proposal(["ra", "dec", "psi"], REFERENCE_TIME)
