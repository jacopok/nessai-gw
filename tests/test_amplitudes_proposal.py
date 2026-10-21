"""The amplitude proposals: the triangle's A/E channels (eight sky elements)
and the network's two sites (no group), and their standardisation."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nessai_gw.group_mixture import (
    ET_EMR_DETECTOR_TENSORS,
    ET_EMR_PLANE_NORMAL,
    ET_EMR_VERTEX,
    PrimeSpaceSkyGroupAction,
    TriangularDetectorGroupAction,
    make_et_group_flow_proposal,
    triangular_group_reparameterisations,
)
from nessai_gw.network_group import (
    DetectorNetworkGeometry,
    make_network_group_flow_proposal,
    network_group_reparameterisations,
)

REFERENCE_TIME = 1187008882.4
PARAMETERS = [
    "chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance",
    "theta_jn", "psi", "phase", "ra", "dec", "geocent_time",
]
AMPLITUDES = ["ln_amp_1", "arg_amp_1", "ln_amp_2", "arg_amp_2"]
BOUNDS = {
    "chirp_mass": [1.19, 1.21], "mass_ratio": [0.5, 1.0],
    "chi_1": [-0.05, 0.05], "chi_2": [-0.05, 0.05],
    "luminosity_distance": [25.0, 4000.0], "theta_jn": [0.0, np.pi],
    "psi": [0.0, np.pi], "phase": [0.0, 2 * np.pi],
    "ra": [0.0, 2 * np.pi], "dec": [-np.pi / 2, np.pi / 2],
    "geocent_time": [REFERENCE_TIME - 0.1, REFERENCE_TIME + 0.1],
}
FLOW = {"n_blocks": 2, "n_neurons": 8, "n_layers": 1, "ftype": "circular"}


def _has_group_mixture():
    try:
        import nessai.flowmodel.group_mixture  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_group_mixture(),
    reason="nessai.flowmodel.group_mixture not available",
)


def _stub_model():
    from nessai.model import Model

    class _StubModel(Model):
        def __init__(self):
            self.names = list(PARAMETERS)
            self.bounds = {k: np.asarray(v) for k, v in BOUNDS.items()}

        def log_prior(self, x):
            return np.log(self.in_bounds(x), dtype=float)

        def log_likelihood(self, x):
            return np.zeros(len(np.atleast_1d(x)))

    model = _StubModel()
    model.set_rng(np.random.default_rng(0))
    return model


def _ideal_triangle():
    """Three 60-degree interferometers in the ET-EMR plane, 120 degrees apart:
    the geometry the triangle's group is exact for."""
    action = TriangularDetectorGroupAction(
        REFERENCE_TIME, ET_EMR_PLANE_NORMAL, ET_EMR_VERTEX
    )
    e1, e2, _ = action._basis_np

    def arm(a):
        return np.cos(a) * e1 + np.sin(a) * e2

    def l_tensor(a):
        u, v = arm(a), arm(a + np.pi / 3)
        return 0.5 * (np.outer(u, u) - np.outer(v, v))

    return np.stack([l_tensor(0.3 + 2 * np.pi * k / 3) for k in range(3)])


def _prior_draws(model, n, seed=7, theta_jn=None, centre=None, width=None):
    from nessai.livepoint import numpy_array_to_live_points

    rng = np.random.default_rng(seed)
    if centre is None:
        theta = np.column_stack([rng.uniform(*BOUNDS[k], n) for k in PARAMETERS])
        theta[:, PARAMETERS.index("dec")] = np.arcsin(rng.uniform(-1, 1, n))
    else:
        theta = np.column_stack([
            centre[k] + width[k] * rng.standard_normal(n) for k in PARAMETERS
        ])
    live = numpy_array_to_live_points(theta, PARAMETERS)
    live["logL"] = 0.0
    live["logP"] = model.log_prior(live)
    return live[np.isfinite(live["logP"])]


def _triangle_proposal(tmp_path, tensors, **kwargs):
    cls = make_et_group_flow_proposal(
        PARAMETERS, REFERENCE_TIME, effective_spin=True, amplitudes=True,
        **kwargs,
    )
    proposal = cls(
        _stub_model(),
        output=str(tmp_path),
        poolsize=200,
        reparameterisations=triangular_group_reparameterisations(
            PARAMETERS, REFERENCE_TIME, effective_spin=True, amplitudes=True,
            amplitude_tensors=tensors,
        ),
        flow_config=FLOW,
        training_config={"max_epochs": 3, "patience": 3},
    )
    proposal.initialise()
    return proposal


def _network_proposal(tmp_path, **kwargs):
    from test_network_group import DE, IT, TENSORS

    geometry = DetectorNetworkGeometry(
        [IT, DE], [30.0, 40.0], REFERENCE_TIME, names=["IT", "DE"],
        tensors=TENSORS,
    )
    cls = make_network_group_flow_proposal(
        PARAMETERS, geometry, network_amplitudes=True, **kwargs
    )
    proposal = cls(
        _stub_model(),
        output=str(tmp_path),
        poolsize=200,
        reparameterisations=network_group_reparameterisations(
            PARAMETERS, geometry, network_amplitudes=True
        ),
        flow_config=FLOW,
        training_config={"max_epochs": 2, "patience": 2},
    )
    proposal.initialise()
    return proposal


def test_triangle_reparameterisations():
    reps = triangular_group_reparameterisations(
        PARAMETERS, REFERENCE_TIME, amplitudes=True
    )
    first, *_ = reps
    assert reps[first]["reparameterisation"] == "network-amplitudes"
    assert reps[first]["parameters"] == [
        "luminosity_distance", "theta_jn", "psi", "phase"
    ]
    np.testing.assert_array_equal(
        reps[first]["tensors"], ET_EMR_DETECTOR_TENSORS
    )
    assert not {"theta_jn", "phase", "psi"} & set(reps)
    with pytest.raises(ValueError, match="excludes"):
        triangular_group_reparameterisations(
            PARAMETERS, REFERENCE_TIME, amplitudes=True, chirp_distance=True,
            chirp_distance_k0=0,
        )


def test_triangle_excludes():
    for kwargs in ({"adaptive_domain": True}, {"circular_psi_phase": True},
                   {"phase_recanon": True}):
        with pytest.raises(ValueError, match="amplitudes excludes"):
            make_et_group_flow_proposal(
                PARAMETERS, REFERENCE_TIME, amplitudes=True, **kwargs
            )


def test_sky_action_needs_eight_elements():
    action = TriangularDetectorGroupAction(
        REFERENCE_TIME, ET_EMR_PLANE_NORMAL, ET_EMR_VERTEX,
        polarisation_quarter=True,
    )
    with pytest.raises(ValueError, match="eight-element"):
        PrimeSpaceSkyGroupAction(action, ["sky_u", "sky_v"])


def test_triangle_prime_action_matches_physical(tmp_path):
    """Mapping a point's physical images through the reparameterisations
    gives the prime-space images: the amplitudes and t_det do not move, the
    sky moves by the eight elements.  Exact for the ideal triangle."""
    proposal = _triangle_proposal(tmp_path, _ideal_triangle())
    prime = list(proposal.prime_parameters)
    assert set(AMPLITUDES) | {"sky_u", "sky_v", "t_det"} <= set(prime)
    assert not {"psi_prime", "delta_phase", "theta_jn_prime"} & set(prime)
    flow = proposal.flow.model
    assert flow.group_size == 8
    assert sorted(prime[i] for i in flow.circular_features) == sorted(
        ["arg_amp_1", "arg_amp_2"]
    )

    model = _stub_model()
    live = _prior_draws(model, 300)
    # away from the detector plane, where the channels' responses are
    # parallel, and from face-on / face-off
    keep = (np.abs(np.sin(live["theta_jn"])) > 0.05)
    live = live[keep]
    n = len(live)
    physical = TriangularDetectorGroupAction(
        REFERENCE_TIME, ET_EMR_PLANE_NORMAL, ET_EMR_VERTEX
    )
    action = PrimeSpaceSkyGroupAction(physical, prime)
    x_prime, _ = proposal.rescale(live)
    x_prime = x_prime[:n]
    base = {k: torch.as_tensor(x_prime[k].copy()) for k in prime}
    as_t = lambda a: torch.as_tensor(np.array(a, dtype=float))  # noqa: E731
    for m in range(8):
        modes = torch.full((n,), m)
        out = physical({
            "ra": as_t(live["ra"]), "sin_dec": as_t(np.sin(live["dec"])),
            "psi": as_t(live["psi"]),
            "cos_theta_jn": as_t(np.cos(live["theta_jn"])),
            "phase": as_t(live["phase"]),
            "geocent_time": as_t(live["geocent_time"]),
        }, modes)
        image = live.copy()
        image["ra"] = out["ra"].numpy()
        image["dec"] = np.arcsin(out["sin_dec"].numpy())
        image["psi"] = out["psi"].numpy()
        image["theta_jn"] = np.arccos(out["cos_theta_jn"].numpy())
        image["phase"] = out["phase"].numpy()
        image["geocent_time"] = out["geocent_time"].numpy()
        y_prime, _ = proposal.rescale(image)
        acted = action(base, modes)
        for k in prime:
            diff = acted[k].numpy() - y_prime[k][:n]
            if k in ("sky_u", "arg_amp_1", "arg_amp_2"):
                period = 1.0 if k == "sky_u" else 2 * np.pi
                diff = np.mod(diff + period / 2, period) - period / 2
            # t_det: float64 GPS seconds round to 2.4e-7 s, here / 5e-3 s
            atol = 2e-4 if k == "t_det" else 1e-7
            np.testing.assert_allclose(diff, 0.0, atol=atol, err_msg=f"{k}, element {m}")
        back = action(acted, modes, inverse=True)
        for k in prime:
            np.testing.assert_allclose(back[k].numpy(), x_prime[k], atol=1e-12)
    # exactly one image of each point is canonical
    in_domain = torch.stack([
        action.in_fundamental_domain(action(base, torch.full((n,), m)))
        for m in range(8)
    ])
    assert torch.all(in_domain.sum(0) == 1)


def test_triangle_initialise_train_and_draw(tmp_path):
    proposal = _triangle_proposal(tmp_path, ET_EMR_DETECTOR_TENSORS)
    model = _stub_model()
    live = _prior_draws(model, 1000)
    x_prime, log_j = proposal.rescale(live)
    back, log_j_inv = proposal.inverse_rescale(x_prime)
    n = len(live)
    # the inverse loses digits near face-on / off and near the detector
    # plane, where the two channels' responses become parallel
    action = TriangularDetectorGroupAction(
        REFERENCE_TIME, ET_EMR_PLANE_NORMAL, ET_EMR_VERTEX
    )
    _, beta, _ = action._to_frame(
        *(torch.as_tensor(np.array(live[k])) for k in ("ra", "dec", "psi"))
    )
    ok = (np.abs(np.sin(live["theta_jn"])) > 0.05) & (
        np.abs(beta.numpy()) > 0.05
    )
    for name in ("luminosity_distance", "theta_jn", "ra", "dec"):
        np.testing.assert_allclose(
            back[name][:n][ok], live[name][ok], rtol=1e-6
        )
    np.testing.assert_allclose(log_j[:n][ok], -log_j_inv[:n][ok], atol=1e-6)
    proposal.train(live)
    worst = live[:1].copy()
    worst["logL"] = -np.inf
    proposal.populate(worst, n_samples=200)
    assert len(proposal.samples) > 0
    assert np.all(np.isfinite(model.log_prior(proposal.samples)))


#: a narrow posterior with its phase on the +-pi seam of arg Z
_CENTRE = dict(chirp_mass=1.2, mass_ratio=0.8, chi_1=0.0, chi_2=0.0,
               luminosity_distance=400.0, theta_jn=0.6, psi=1.0, phase=0.0,
               ra=1.0, dec=0.3, geocent_time=REFERENCE_TIME)
_WIDTH = dict(chirp_mass=1e-3, mass_ratio=0.05, chi_1=0.01, chi_2=0.01,
              luminosity_distance=10.0, theta_jn=0.03, psi=0.02, phase=0.02,
              ra=0.01, dec=0.01, geocent_time=1e-4)


@pytest.mark.parametrize("kind", ["triangle", "network"])
def test_standardisation(tmp_path, kind):
    """Every coordinate the base flow sees is standardised: the real ones to
    zero mean and unit variance, the circular ones (the amplitudes'
    arguments) by the Möbius map, which spreads a narrow angle round the
    circle."""
    if kind == "triangle":
        proposal = _triangle_proposal(
            tmp_path, ET_EMR_DETECTOR_TENSORS, circular_standardisation=True
        )
    else:
        proposal = _network_proposal(
            tmp_path, circular_standardisation=True
        )
    model = _stub_model()
    live = _prior_draws(model, 2000, centre=_CENTRE, width=_WIDTH)
    proposal.train(live)
    prime = list(proposal.prime_parameters)
    flow = proposal.flow.model
    assert flow.circular_standardisation
    x_prime, _ = proposal.rescale(live)
    x_prime = x_prime[:len(live)]
    x = torch.as_tensor(np.column_stack([x_prime[k] for k in prime]))
    with torch.no_grad():
        # each point's canonical image, as the standardisation is fitted on
        assigned, pre, _ = flow._assign_branch(x.to(flow._canon_mean.dtype))
        canon = pre[assigned, torch.arange(len(x))]
        u = flow.canonical_to_standardised(canon).numpy()
    circular = set(flow.circular_features)
    assert {prime.index(k) for k in ("arg_amp_1", "arg_amp_2")} <= circular
    for i, k in enumerate(prime):
        if i in circular:
            assert abs(np.exp(1j * u[:, i]).mean()) < 0.2, k
        else:
            assert abs(u[:, i].mean()) < 0.05, k
            assert u[:, i].std() == pytest.approx(1.0, abs=0.05), k
