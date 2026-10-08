"""Circular base-frame coordinates for the group-mixture proposals."""

import numpy as np
import pytest
import torch

from nessai_gw.group_mixture import (
    POLARISATION_PHASE_CIRCLE,
    AdaptiveFundamentalDomain,
    CircularCanonicalTransform,
    SkyOctantProbit,
    make_et_group_flow_proposal,
    triangular_group_reparameterisations,
)
from nessai_gw.network_group import (
    AdaptiveNetworkDomain,
    BaselineFlipSymmetry,
    PrimeSpacePolarisationPhaseAction,
)

PRIME = ["theta_jn_prime", "psi_prime", "delta_phase", "sky_u", "sky_v", "t_det"]
REFERENCE_TIME = 1187008882.4


def _has_circular_group_mixture():
    try:
        import inspect

        from nessai.flowmodel.group_mixture import make_group_mixture_flow
    except ImportError:
        return False
    return "circular_parameters" in inspect.signature(
        make_group_mixture_flow
    ).parameters


requires_circular = pytest.mark.skipif(
    not _has_circular_group_mixture(),
    reason="nessai without circular group-mixture coordinates",
)


@pytest.fixture(autouse=True)
def float64():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)


def _canonical_points(n, seed=1, u_hi=0.5):
    rng = np.random.default_rng(seed)
    cols = {
        "theta_jn_prime": rng.uniform(-1, 1, n),
        "psi_prime": rng.uniform(-1, 0, n),
        "delta_phase": rng.uniform(-1, 0, n),
        "sky_u": rng.uniform(0, u_hi, n),
        "sky_v": rng.uniform(0, 1, n),
        "t_det": rng.normal(size=n),
    }
    return torch.as_tensor(np.column_stack([cols[n] for n in PRIME]))


def _circle_transform(start_u=0.0):
    return CircularCanonicalTransform(
        {
            "psi_prime": POLARISATION_PHASE_CIRCLE,
            "delta_phase": POLARISATION_PHASE_CIRCLE,
            "sky_u": (start_u, 1.0, (0.0, 1.0)),
        },
        PRIME,
        inner=SkyOctantProbit(PRIME, u_scale=None, v_scale=1.0),
    )


def _wrapped(d):
    return torch.remainder(d + np.pi, 2 * np.pi) - np.pi


def test_circular_transform_round_trip_and_jacobian():
    tr = _circle_transform()
    canon = _canonical_points(500, u_hi=1.0)
    t, lj = tr.forward(canon)
    for name in ("psi_prime", "delta_phase", "sky_u"):
        i = PRIME.index(name)
        assert (t[:, i] >= -np.pi).all() and (t[:, i] < np.pi).all()
    back, lj_inv = tr.inverse(t)
    np.testing.assert_allclose(back, canon, atol=1e-10)
    np.testing.assert_allclose(lj + lj_inv, 0.0, atol=1e-12)
    # numeric Jacobian of the whole map
    x = canon[:4].clone()
    jac = torch.autograd.functional.jacobian(
        lambda v: tr.forward(v)[0].sum(0), x
    )
    num = torch.stack([torch.linalg.slogdet(jac[:, i, :])[1] for i in range(4)])
    np.testing.assert_allclose(num, lj[:4], atol=1e-8)


def test_circular_transform_glues_the_fold():
    """The ends of the fundamental-domain intervals meet on the circle."""
    tr = _circle_transform()
    lo = _canonical_points(1)
    hi = lo.clone()
    i = PRIME.index("psi_prime")
    lo[0, i], hi[0, i] = -1.0 + 1e-9, -1e-9
    t_lo, _ = tr.forward(lo)
    t_hi, _ = tr.forward(hi)
    assert float(_wrapped(t_hi[0, i] - t_lo[0, i]).abs()) < 1e-7
    # any angle maps back into the fundamental domain
    t = t_lo.repeat(100, 1)
    t[:, i] = torch.linspace(-np.pi, np.pi, 100)
    back, _ = tr.inverse(t)
    assert (back[:, i] >= -1).all() and (back[:, i] < 0).all()


def test_sky_probit_v_only():
    tr = SkyOctantProbit(PRIME, u_scale=None, v_scale=1.0)
    canon = _canonical_points(200, u_hi=1.0)
    t, lj = tr.forward(canon)
    iu = PRIME.index("sky_u")
    np.testing.assert_array_equal(t[:, iu], canon[:, iu])
    back, lj_inv = tr.inverse(t)
    np.testing.assert_allclose(back, canon, atol=1e-10)
    np.testing.assert_allclose(lj + lj_inv, 0.0, atol=1e-12)


@pytest.mark.parametrize("sky_u_offset", [0.0, 0.3])
def test_baseline_flip_symmetry_is_the_half_turn(sky_u_offset):
    """``h`` in the circular base frame is the prime-space half-turn."""
    action = PrimeSpacePolarisationPhaseAction(
        PRIME, baseline_flip=True, sky_u_offset=sky_u_offset
    )
    tr = _circle_transform(start_u=sky_u_offset)
    sym = BaselineFlipSymmetry(PRIME)
    canon = _canonical_points(400, u_hi=1.0)
    t, _ = tr.forward(canon)
    (h_t,) = sym.images(t)
    d = {n: canon[:, i] for i, n in enumerate(PRIME)}
    out = action(d, torch.full((400,), 4))
    t_ref, _ = tr.forward(torch.stack([out[n] for n in PRIME], dim=1))
    for name in PRIME:
        i = PRIME.index(name)
        diff = h_t[:, i] - t_ref[:, i]
        if name in ("psi_prime", "delta_phase", "sky_u"):
            diff = _wrapped(diff)
        np.testing.assert_allclose(diff, 0.0, atol=1e-9, err_msg=name)
    # involution, and the fold lands in the canonical half circle
    (hh_t,) = sym.images(h_t)
    np.testing.assert_allclose(_wrapped(hh_t - t), 0.0, atol=1e-12)
    folded = sym.fold(t)
    iu = PRIME.index("sky_u")
    assert (folded[:, iu] < 0).all()
    back, _ = tr.inverse(folded)
    assert action.in_fundamental_domain(
        {n: back[:, i] for i, n in enumerate(PRIME)}
    ).sum() > 0


def test_adaptive_domains_keep_circular_seams():
    action = PrimeSpacePolarisationPhaseAction(PRIME, baseline_flip=True)
    rng = np.random.default_rng(3)
    n = 400
    canon = _canonical_points(n)
    # a peak straddling the psi / delta seams and the sky_u seam
    canon[:, PRIME.index("psi_prime")] = torch.as_tensor(
        np.mod(rng.normal(0, 0.05, n) + 1, 1.0) - 1
    )
    canon[:, PRIME.index("delta_phase")] = torch.as_tensor(
        np.mod(rng.normal(0, 0.05, n) + 1, 1.0) - 1
    )
    canon[:, PRIME.index("sky_u")] = torch.as_tensor(
        np.mod(rng.normal(0, 0.02, n), 0.5)
    )
    dom = AdaptiveNetworkDomain(action, PRIME, circular_psi_phase=True)
    dom.update(canon)
    cu, cpsi, cdel, _ = dom._state()[0]
    assert cpsi == 0.0 and cdel == 0.0 and cu != 0.0
    dom = AdaptiveNetworkDomain(
        action, PRIME, circular_psi_phase=True, circular_sky_u=True
    )
    dom.update(canon)
    assert dom._identity()
    with pytest.raises(ValueError, match="allow_phase_mode"):
        AdaptiveFundamentalDomain(
            action, None, allow_phase_mode=True, circular_psi_phase=True
        )


# ---------------------------------------------------------------------------
# ET triangle, end to end
# ---------------------------------------------------------------------------
BNS_PARAMETERS = [
    "chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance",
    "theta_jn", "psi", "phase", "ra", "dec", "geocent_time",
]
BOUNDS = {
    "chirp_mass": [1.19, 1.21], "mass_ratio": [0.5, 1.0],
    "chi_1": [-0.05, 0.05], "chi_2": [-0.05, 0.05],
    "luminosity_distance": [25.0, 4000.0], "theta_jn": [0.0, np.pi],
    "psi": [0.0, np.pi], "phase": [0.0, 2 * np.pi],
    "ra": [0.0, 2 * np.pi], "dec": [-np.pi / 2, np.pi / 2],
    "geocent_time": [REFERENCE_TIME - 0.1, REFERENCE_TIME + 0.1],
}


def _stub_model():
    from nessai.model import Model

    class _StubModel(Model):
        def __init__(self):
            self.names = list(BNS_PARAMETERS)
            self.bounds = {k: np.asarray(v) for k, v in BOUNDS.items()}

        def log_prior(self, x):
            return np.log(self.in_bounds(x), dtype=float)

        def log_likelihood(self, x):
            return np.zeros(len(np.atleast_1d(x)))

    model = _StubModel()
    model.set_rng(np.random.default_rng(0))
    return model


def _prior_live_points(model, n, seed=7):
    from nessai.livepoint import numpy_array_to_live_points

    rng = np.random.default_rng(seed)
    theta = np.column_stack([rng.uniform(*BOUNDS[p], n) for p in BNS_PARAMETERS])
    theta[:, BNS_PARAMETERS.index("dec")] = np.arcsin(rng.uniform(-1, 1, n))
    theta[:, BNS_PARAMETERS.index("theta_jn")] = np.arccos(
        rng.uniform(-1, 1, n)
    )
    live = numpy_array_to_live_points(theta, BNS_PARAMETERS)
    live["logL"] = 0.0
    live["logP"] = model.log_prior(live)
    return live


def _check_trained_proposal(proposal, model, live):
    from nessai.flows.circular import CircularNeuralSplineFlow

    flow = proposal.flow.model
    assert isinstance(flow.base_flow, CircularNeuralSplineFlow)
    prime = list(proposal.prime_parameters)
    circ = [prime[i] for i in flow.circular_features]
    assert set(circ) == set(flow.circular_parameters)
    # the base flow was built in nessai's prime order, not the predicted one
    assert flow.param_names == prime
    assert sorted(flow.base_flow.circular_features) == sorted(
        prime.index(n) for n in flow.circular_parameters
    )
    mask = proposal.latent_real_mask
    assert not mask[flow.circular_features].any()
    proposal.train(live)
    with torch.no_grad():
        x, log_q = flow.sample_and_log_prob(500)
        np.testing.assert_allclose(
            log_q, flow.log_prob(x), rtol=1e-5, atol=1e-5
        )
    assert float(flow._log_domain_mass) == pytest.approx(0.0, abs=1e-12)
    worst = live[:1].copy()
    worst["logL"] = -np.inf
    proposal.populate(worst, n_samples=200)
    assert len(proposal.samples) > 0
    assert np.all(np.isfinite(model.log_prior(proposal.samples)))

    # resume from a checkpoint: the flow is rebuilt from its configuration
    # and the saved weights loaded into it
    import pickle

    weights_file = proposal.flow.weights_file
    assert weights_file is not None
    resumed = pickle.loads(pickle.dumps(proposal))
    resumed.resume(model, dict(proposal.flow_config), weights_file)
    rflow = resumed.flow.model
    rflow.eval()
    with torch.no_grad():
        x, _ = flow.sample_and_log_prob(300)
        np.testing.assert_allclose(
            rflow.log_prob(x), flow.log_prob(x), rtol=1e-5, atol=1e-5
        )
    return circ


@requires_circular
@pytest.mark.parametrize("adaptive_domain", [False, True])
def test_et_proposal_circular_end_to_end(tmp_path, adaptive_domain):
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, phase_reflection=True,
        polarisation_quarter=True, psi_single=True, sky_2d="auto",
        gaussianise_sky=True, adaptive_domain=adaptive_domain,
        circular_psi_phase=True, log_mass_ratio=False,
    )
    fm = cls._FlowModelClass
    assert fm.circular_parameters == ["psi_prime", "delta_phase"]
    assert isinstance(fm.canonical_transform, CircularCanonicalTransform)
    model = _stub_model()
    proposal = cls(
        model,
        output=str(tmp_path),
        poolsize=200,
        reparameterisations=triangular_group_reparameterisations(
            BNS_PARAMETERS, REFERENCE_TIME, log_mass_ratio=False
        ),
        flow_config={
            "ftype": "circular", "n_blocks": 2, "n_neurons": 8,
            "n_layers": 1, "real_transform": "affine",
        },
        training_config={"max_epochs": 3, "patience": 3},
        latent_temperature=1.4,
    )
    proposal.initialise()
    circ = _check_trained_proposal(
        proposal, model, _prior_live_points(model, 1000)
    )
    assert set(circ) == {"psi_prime", "delta_phase"}


@requires_circular
def test_et_proposal_circular_incompatibilities():
    with pytest.raises(RuntimeError, match="phase_recanon"):
        make_et_group_flow_proposal(
            BNS_PARAMETERS, REFERENCE_TIME, phase_reflection=True,
            polarisation_quarter=True, phase_recanon=True,
            circular_psi_phase=True,
        )
    with pytest.raises(RuntimeError, match="phase_reflection"):
        make_et_group_flow_proposal(
            BNS_PARAMETERS, REFERENCE_TIME, phase_reflection=False,
            polarisation_quarter=True, circular_psi_phase=True,
        )
