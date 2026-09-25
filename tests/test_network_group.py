"""Tests for the detector-network group proposal in :mod:`nessai_gw.network_group`."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nessai_gw._geometry import geocenter_time_delay
from nessai_gw.group_mixture import (
    SkyOctantProbit,
    detector_vertex_from_geodetic,
)
from nessai_gw.network_group import (
    AdaptiveNetworkDomain,
    DetectorNetworkGeometry,
    PolarisationPhaseGroupAction,
    PrimeSpacePolarisationPhaseAction,
    make_network_group_flow_proposal,
    network_group_reparameterisations,
)
from nessai_gw.reparameterisations.sky import EqualAreaSky

REFERENCE_TIME = 1187008882.4


def _l_tensor(lat, lon, xarm_azimuth):
    """Tensor of a 90-degree L on the local horizontal plane (azimuth from
    North towards East), as bilby builds it from its geodetic parameters."""
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
    )
    x = np.cos(xarm_azimuth) * north + np.sin(xarm_azimuth) * east
    y = -np.sin(xarm_azimuth) * north + np.cos(xarm_azimuth) * east
    return 0.5 * (np.outer(x, x) - np.outer(y, y))


# ET 2L: Sardinia and Lusatia (ET_1L_IT / ET_1L_DE)
_IT_LATLON = (np.radians(40 + 31 / 60), np.radians(9 + 25 / 60))
_DE_LATLON = (np.radians(51.275), np.radians(14.100))
IT = detector_vertex_from_geodetic(*_IT_LATLON)
DE = detector_vertex_from_geodetic(*_DE_LATLON)
TENSORS = np.array([
    _l_tensor(*_IT_LATLON, np.radians(70.0)),
    _l_tensor(*_DE_LATLON, np.radians(115.0)),
])

PARAMETERS = [
    "chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance",
    "theta_jn", "psi", "phase", "ra", "dec", "geocent_time",
]
PRIME = ["theta_jn_prime", "psi_prime", "delta_phase", "sky_u", "sky_v", "t_det"]


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


@pytest.fixture(scope="module")
def geometry():
    return DetectorNetworkGeometry([IT, DE], [30.0, 40.0], REFERENCE_TIME,
                                   names=["IT", "DE"], tensors=TENSORS)


def _sky(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n))


def _earth_fixed(ra, dec, gmst):
    phi = ra - gmst
    return np.column_stack(
        [np.cos(dec) * np.cos(phi), np.cos(dec) * np.sin(phi), np.sin(dec)]
    )


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_timing_vertex_is_snr_squared_barycentre(geometry):
    w = np.array([30.0, 40.0]) ** 2
    np.testing.assert_allclose(
        geometry.timing_vertex, (w[0] * IT + w[1] * DE) / w.sum()
    )


def test_barycentre_time_is_weighted_mean_of_arrival_times(geometry):
    ra, dec = _sky(500)
    g = geometry.gmst
    bar = geocenter_time_delay(geometry.timing_vertex, g, ra, dec)
    per = [geocenter_time_delay(v, g, ra, dec) for v in (IT, DE)]
    np.testing.assert_allclose(
        bar, geometry.timing_weights @ np.stack(per), atol=1e-15
    )


def test_baseline_runs_from_second_to_loudest_site(geometry):
    b = (DE - IT) / np.linalg.norm(DE - IT)  # DE is louder
    np.testing.assert_allclose(geometry.baseline, b)
    assert [s["members"] for s in geometry.sites] == [["DE"], ["IT"]]


def test_sky_frame_is_baseline_aligned(geometry):
    """EqualAreaSky with the network rotation gives sky_v = (1 - cos a) / 2,
    ``a`` the angle to the baseline, i.e. sky_v is an affine function of the
    arrival-time difference between the two sites."""
    R = geometry.sky_frame_rotation
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    ra, dec = _sky(500, seed=1)
    rep = EqualAreaSky(
        parameters=["ra", "dec"],
        prior_bounds={"ra": [0, 2 * np.pi], "dec": [-np.pi / 2, np.pi / 2]},
        rotation=R,
    )
    from nessai.livepoint import numpy_array_to_live_points

    x = numpy_array_to_live_points(np.column_stack([ra, dec]), ["ra", "dec"])
    xp = numpy_array_to_live_points(np.zeros((500, 2)), ["sky_u", "sky_v"])
    _, xp, _ = rep.reparameterise(x, xp, np.zeros(500))
    cos_a = _earth_fixed(ra, dec, geometry.gmst) @ geometry.baseline
    np.testing.assert_allclose(xp["sky_v"], 0.5 * (1 - cos_a), atol=1e-12)
    dt = geocenter_time_delay(IT, geometry.gmst, ra, dec) - geocenter_time_delay(
        DE, geometry.gmst, ra, dec
    )
    slope = np.polyfit(dt, xp["sky_v"], 1)
    np.testing.assert_allclose(np.polyval(slope, dt), xp["sky_v"], atol=1e-9)


def test_sky_seam_is_horizontal(geometry):
    """beta = 0 (the sky_u seam) is perpendicular to both the baseline and the
    mid-baseline vertical; beta = pi/2 is the vertical."""
    R_ef = geometry.sky_frame_rotation @ np.linalg.inv(
        np.array([[np.cos(geometry.gmst), np.sin(geometry.gmst), 0],
                  [-np.sin(geometry.gmst), np.cos(geometry.gmst), 0],
                  [0, 0, 1]])
    )
    e1, e2, _ = R_ef
    up = (IT + DE) / np.linalg.norm(IT + DE)
    assert abs(e1 @ up) < 1e-12
    assert abs(e1 @ geometry.baseline) < 1e-12
    assert e2 @ up > 0.99


def test_colocated_detectors_merge_into_one_site():
    et = [IT + np.array([dx, dy, 0.0]) for dx, dy in ((0, 0), (1e4, 0), (0, 1e4))]
    geo = DetectorNetworkGeometry(
        et + [DE], [10.0, 20.0, 20.0, 25.0], REFERENCE_TIME,
        names=["E1", "E2", "E3", "DE"],
    )
    assert len(geo.sites) == 2
    assert geo.sites[0]["members"] == ["E2", "E3", "E1"]
    assert geo.sites[0]["snr"] == pytest.approx(30.0)
    # baseline joins the (loudest) ET site to DE, not two ET arms
    assert np.linalg.norm(geo.sites[0]["vertex"] - IT) < 2e4
    b = geo.sites[0]["vertex"] - DE
    np.testing.assert_allclose(geo.baseline, b / np.linalg.norm(b))


def test_single_site_falls_back_to_equatorial_sky():
    geo = DetectorNetworkGeometry([IT, IT + 1e4], [10.0, 10.0], REFERENCE_TIME)
    assert geo.baseline is None
    np.testing.assert_array_equal(geo.sky_frame_rotation, np.eye(3))


def test_from_interferometers():
    class _Geom:
        def __init__(self, v):
            self.vertex = v

    class _Ifo:
        def __init__(self, name, v, snr):
            self.name = name
            self.geometry = _Geom(v)
            self.meta_data = {"optimal_SNR": snr}

    ifos = [_Ifo("IT", IT, 30.0), _Ifo("DE", DE, 40.0)]
    a = DetectorNetworkGeometry.from_interferometers(ifos, REFERENCE_TIME)
    b = DetectorNetworkGeometry.from_interferometers(
        ifos, REFERENCE_TIME, snrs={"DE": 40.0, "IT": 30.0}
    )
    for geo in (a, b):
        np.testing.assert_allclose(
            geo.sky_frame_rotation,
            DetectorNetworkGeometry(
                [IT, DE], [30.0, 40.0], REFERENCE_TIME
            ).sky_frame_rotation,
        )
    # the stubs carry no detector tensors
    assert a.tensors is None
    for ifo, d in zip(ifos, TENSORS):
        ifo.detector_tensor = d
    c = DetectorNetworkGeometry.from_interferometers(ifos, REFERENCE_TIME)
    np.testing.assert_array_equal(c.tensors, TENSORS)
    assert c.reference_detector == 1  # DE is louder
    del ifos[0].meta_data["optimal_SNR"]
    with pytest.raises(ValueError, match="optimal_SNR"):
        DetectorNetworkGeometry.from_interferometers(ifos, REFERENCE_TIME)


def test_tensor_shape_is_checked():
    with pytest.raises(ValueError, match=r"\(2, 3, 3\) detector tensors"):
        DetectorNetworkGeometry(
            [IT, DE], [30.0, 40.0], REFERENCE_TIME, tensors=TENSORS[:1]
        )


# ---------------------------------------------------------------------------
# group action
# ---------------------------------------------------------------------------
def _response_22(psi, phase, a, b, cplus, ccross):
    """(2, 2)-mode strain of a detector with pattern amplitudes (a, b)."""
    fp = a * np.cos(2 * psi) + b * np.sin(2 * psi)
    fc = -a * np.sin(2 * psi) + b * np.cos(2 * psi)
    e = np.exp(-2j * phase)
    return fp * cplus * e + fc * (-1j) * ccross * e


def _physical_points(n, seed=2):
    rng = np.random.default_rng(seed)
    return {
        "cos_theta_jn": torch.as_tensor(rng.uniform(-1, 1, n)),
        "psi": torch.as_tensor(rng.uniform(0, np.pi, n)),
        "phase": torch.as_tensor(rng.uniform(0, 2 * np.pi, n)),
    }


@pytest.mark.parametrize("k", range(4))
def test_physical_action_leaves_every_22_response_unchanged(k):
    pts = _physical_points(200)
    out = PolarisationPhaseGroupAction()(pts, torch.full((200,), k))
    rng = np.random.default_rng(3)
    for _ in range(5):  # arbitrary detectors / sources
        a, b, cp, cc = rng.normal(size=4)
        h0 = _response_22(pts["psi"].numpy(), pts["phase"].numpy(), a, b, cp, cc)
        h1 = _response_22(out["psi"].numpy(), out["phase"].numpy(), a, b, cp, cc)
        np.testing.assert_allclose(h1, h0, atol=1e-12)


def _prime_points(n, seed=4):
    rng = np.random.default_rng(seed)
    return {
        "theta_jn_prime": torch.as_tensor(rng.uniform(-1, 1, n)),
        "psi_prime": torch.as_tensor(rng.uniform(-1, 1, n)),
        "delta_phase": torch.as_tensor(rng.uniform(-1, 1, n)),
        "sky_u": torch.as_tensor(rng.uniform(0, 1, n)),
        "sky_v": torch.as_tensor(rng.uniform(0, 1, n)),
        "t_det": torch.as_tensor(rng.normal(size=n)),
    }


def _prime_to_physical(p):
    psi = np.mod((p["psi_prime"].numpy() + 1) * np.pi / 2, np.pi)
    sign = np.where(p["theta_jn_prime"].numpy() > 0, -1.0, 1.0)
    delta = np.mod((p["delta_phase"].numpy() + 1) * np.pi, 2 * np.pi)
    return psi, np.mod(delta - sign * psi, 2 * np.pi), sign


@pytest.fixture(scope="module")
def prime_action():
    return PrimeSpacePolarisationPhaseAction(PRIME)


@pytest.mark.parametrize("k", range(4))
def test_prime_action_matches_physical(prime_action, k):
    p = _prime_points(300)
    psi, phase, sign = _prime_to_physical(p)
    out = prime_action(p, torch.full((300,), k))
    psi_o, phase_o, sign_o = _prime_to_physical(out)
    np.testing.assert_array_equal(sign_o, sign)
    np.testing.assert_allclose(psi_o, np.mod(psi + k * np.pi / 2, np.pi), atol=1e-12)
    d = np.angle(np.exp(1j * (phase_o - (phase - k * np.pi / 2))))
    np.testing.assert_allclose(d, 0.0, atol=1e-12)
    for n in ("theta_jn_prime", "sky_u", "sky_v", "t_det"):
        assert torch.equal(out[n], p[n])


@pytest.mark.parametrize("k", range(4))
def test_prime_action_round_trip(prime_action, k):
    p = _prime_points(300)
    modes = torch.full((300,), k)
    back = prime_action(prime_action(p, modes), modes, inverse=True)
    for n in PRIME:
        d = back[n] - p[n]
        if n in ("psi_prime", "delta_phase"):
            d = torch.remainder(d + 1, 2.0) - 1
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-12), n


@pytest.mark.parametrize("offsets", [(0.0, 0.0), (0.7, 2.1)])
def test_fundamental_domain_partitions_orbits(offsets):
    act = PrimeSpacePolarisationPhaseAction(PRIME, *offsets)
    p = _prime_points(400)
    hits = sum(
        act.in_fundamental_domain(act(p, torch.full((400,), k))).long()
        for k in range(4)
    )
    assert torch.all(hits == 1)


def test_prime_action_missing_coordinate():
    with pytest.raises(RuntimeError, match="delta_phase"):
        PrimeSpacePolarisationPhaseAction(["psi_prime", "theta_jn_prime"])


# ---------------------------------------------------------------------------
# adaptive seams + sky probit
# ---------------------------------------------------------------------------
def _canonical(act, n, psi0, delta0, u0, seed=5):
    """Points peaked on the (psi, delta_phase, sky_u) seams, folded to D0."""
    rng = np.random.default_rng(seed)
    p = _prime_points(n, seed)
    p["psi_prime"] = torch.as_tensor(
        np.mod(2 * (psi0 + 0.05 * rng.normal(size=n)), 2 * np.pi) / np.pi - 1
    )
    p["delta_phase"] = torch.as_tensor(
        np.mod(delta0 + 0.05 * rng.normal(size=n), 2 * np.pi) / np.pi - 1
    )
    p["sky_u"] = torch.as_tensor(np.mod(u0 + 0.02 * rng.normal(size=n), 1.0))
    z = torch.stack([p[k] for k in PRIME], dim=1)
    for k in range(4):
        d = act({n_: z[:, i] for i, n_ in enumerate(PRIME)}, torch.full((n,), k))
        m = act.in_fundamental_domain(d)
        if k == 0:
            out = torch.stack([d[n_] for n_ in PRIME], 1)
        out[m] = torch.stack([d[n_] for n_ in PRIME], 1)[m]
    return out


def test_adaptive_domain_moves_seams_off_the_peak(prime_action):
    # peaks right on psi = 0, delta_phase = pi and sky_u = 0 -- all seams
    canon = _canonical(prime_action, 2000, 0.0, np.pi, 0.0)
    dom = AdaptiveNetworkDomain(prime_action, PRIME)
    assert dom.update(canon)
    y = dom.forward(canon)
    assert bool(dom._box_valid(y, False).all())
    i = {n: PRIME.index(n) for n in PRIME}
    for n, lo, hi in (("psi_prime", -1, 0), ("delta_phase", -1, 0), ("sky_u", 0, 1)):
        v = y[:, i[n]]
        # the peak now sits well inside the box, not on its walls
        span = hi - lo
        assert float(((v - lo) < 0.05 * span).float().mean()) < 0.01, n
        assert float(((hi - v) < 0.05 * span).float().mean()) < 0.01, n
    back = dom.inverse(y)
    assert torch.allclose(back, canon, atol=1e-10)


def test_sky_probit_unit_scales_round_trip_and_jacobian():
    names = ["sky_u", "sky_v", "t_det"]
    tr = SkyOctantProbit(names, u_scale=1.0, v_scale=1.0)
    rng = np.random.default_rng(6)
    canon = torch.as_tensor(
        np.column_stack([rng.uniform(0, 1, 50), rng.uniform(0, 1, 50),
                         rng.normal(size=50)])
    )
    t, lj = tr.forward(canon)
    back, lj_inv = tr.inverse(t)
    assert torch.allclose(back, canon, atol=1e-10)
    assert torch.allclose(lj, -lj_inv)
    jac = torch.autograd.functional.jacobian(
        lambda c: tr.forward(c[None])[0][0], canon[0]
    )
    assert float(torch.logdet(jac)) == pytest.approx(float(lj[0]), abs=1e-8)


# ---------------------------------------------------------------------------
# reparameterisations + proposal
# ---------------------------------------------------------------------------
def test_reparameterisations_use_the_barycentre(geometry):
    reps = network_group_reparameterisations(PARAMETERS, geometry)
    t = reps["geocent_time"]
    assert t["reparameterisation"] == "detector-center-time"
    np.testing.assert_allclose(t["vertex"], geometry.timing_vertex)
    assert reps["phase"] == {"reparameterisation": "polarisation-phase"}
    assert reps["theta_jn"]["update_bounds"] is False


def test_reparameterisations_default_to_chirp_distance(geometry):
    """The chirp distance is the default, at the loudest detector."""
    spec = network_group_reparameterisations(PARAMETERS, geometry)[
        "luminosity_distance"
    ]
    assert spec["reparameterisation"] == "chirp-distance"
    np.testing.assert_array_equal(spec["tensors"], TENSORS)
    assert spec["k0"] == geometry.reference_detector == 1
    assert "luminosity_distance" not in network_group_reparameterisations(
        PARAMETERS, geometry, chirp_distance=False
    )


def test_reparameterisations_default_to_effective_spin(geometry):
    reps = network_group_reparameterisations(PARAMETERS, geometry)
    assert reps["chi_1"] == {
        "reparameterisation": "effective-spin",
        "parameters": ["chi_1", "chi_2"],
    }
    assert "chi_2" not in reps
    off = network_group_reparameterisations(
        PARAMETERS, geometry, effective_spin=False
    )
    assert off["chi_1"] == off["chi_2"] == {"reparameterisation": "aligned-spin"}


def test_chirp_distance_needs_tensors():
    geo = DetectorNetworkGeometry([IT, DE], [30.0, 40.0], REFERENCE_TIME)
    with pytest.raises(ValueError, match="detector tensors"):
        network_group_reparameterisations(PARAMETERS, geo)
    # fine without it, and without a distance to reparameterise
    network_group_reparameterisations(PARAMETERS, geo, chirp_distance=False)
    network_group_reparameterisations(
        [p for p in PARAMETERS if p != "luminosity_distance"], geo
    )


@pytest.mark.parametrize("k", range(4))
def test_chirp_distance_is_invariant_under_the_action(geometry, k):
    """``|R_k0|`` is unchanged by ``g`` whatever the detector geometry, so the
    prime-space action may pass ``chirp_distance`` through."""
    from nessai_gw._effective_distance import response_R

    pts = _physical_points(300)
    out = PolarisationPhaseGroupAction()(pts, torch.full((300,), k))
    ra, dec = _sky(300, seed=4)
    theta_jn = np.arccos(pts["cos_theta_jn"].numpy())

    def abs_r(psi):
        return np.abs(response_R(
            geometry.tensors, ra, dec, psi, theta_jn, geometry.gmst
        ))

    np.testing.assert_allclose(
        abs_r(out["psi"].numpy()), abs_r(pts["psi"].numpy()), rtol=1e-12
    )


@requires_group_mixture
def test_make_network_group_flow_proposal_class(geometry):
    cls = make_network_group_flow_proposal(PARAMETERS, geometry)
    import nessai_gw.group_mixture as gm

    assert cls.__qualname__ == "NetworkGroupFlowProposal"
    assert gm.NetworkGroupFlowProposal is cls
    fm = cls._FlowModelClass
    assert fm.group_size == 4
    assert fm.mode_factor_sizes == [4]
    with pytest.raises(RuntimeError, match="psi"):
        make_network_group_flow_proposal(
            [p for p in PARAMETERS if p != "psi"], geometry
        )


def _stub_model(names, bounds):
    from nessai.model import Model

    class _StubModel(Model):
        def __init__(self):
            self.names = list(names)
            self.bounds = bounds

        def log_prior(self, x):
            return np.log(self.in_bounds(x), dtype=float)

        def log_likelihood(self, x):
            return np.zeros(len(np.atleast_1d(x)))

    return _StubModel()


@requires_group_mixture
def test_network_proposal_initialise_train_and_draw(geometry, tmp_path):
    """End to end: the proposal binds to nessai's real prime order, trains a
    (tiny) group-mixture flow on prior draws and proposes in-bounds points."""
    bounds = {
        "chirp_mass": [1.19, 1.21], "mass_ratio": [0.5, 1.0],
        "chi_1": [-0.05, 0.05], "chi_2": [-0.05, 0.05],
        "luminosity_distance": [25.0, 4000.0], "theta_jn": [0.0, np.pi],
        "psi": [0.0, np.pi], "phase": [0.0, 2 * np.pi],
        "ra": [0.0, 2 * np.pi], "dec": [-np.pi / 2, np.pi / 2],
        "geocent_time": [REFERENCE_TIME - 0.1, REFERENCE_TIME + 0.1],
    }
    model = _stub_model(PARAMETERS, {k: np.asarray(v) for k, v in bounds.items()})
    model.set_rng(np.random.default_rng(0))
    cls = make_network_group_flow_proposal(PARAMETERS, geometry)
    proposal = cls(
        model,
        output=str(tmp_path),
        poolsize=200,
        reparameterisations=network_group_reparameterisations(
            PARAMETERS, geometry
        ),
        flow_config={"n_blocks": 1, "n_neurons": 8, "n_layers": 1},
        training_config={"max_epochs": 3, "patience": 3},
    )
    proposal.initialise()
    prime = list(proposal.prime_parameters)
    assert {
        "sky_u", "sky_v", "psi_prime", "delta_phase", "t_det",
        "chirp_distance", "chi_eff_prime", "chi_diff_prime",
    } <= set(prime)
    assert not {
        "luminosity_distance_prime", "chi_1_prime", "chi_2_prime"
    } & set(prime)
    assert proposal.flow.model.param_names == prime
    (q_reparam,) = [
        r for r in proposal._reparameterisation.values()
        if "mass_ratio" in r.parameters and len(r.parameters) == 1
    ]
    assert q_reparam.has_pre_rescaling  # sampled as ln q

    rng = np.random.default_rng(7)
    from nessai.livepoint import numpy_array_to_live_points

    theta = np.column_stack([
        rng.uniform(*bounds[n], 1000) for n in PARAMETERS
    ])
    theta[:, PARAMETERS.index("dec")] = np.arcsin(rng.uniform(-1, 1, 1000))
    live = numpy_array_to_live_points(theta, PARAMETERS)
    live["logL"] = 0.0
    live["logP"] = model.log_prior(live)
    # chirp_distance round-trips (rescale may append boundary-inverted
    # duplicates after the originals)
    x_prime, log_j = proposal.rescale(live)
    back, log_j_inv = proposal.inverse_rescale(x_prime)
    n = len(live)
    for name in ("luminosity_distance", "chi_1", "chi_2"):
        np.testing.assert_allclose(
            back[name][:n], live[name], rtol=1e-9, atol=1e-12
        )
    np.testing.assert_allclose(log_j[:n], -log_j_inv[:n], atol=1e-8)
    proposal.train(live)
    worst = live[:1].copy()
    worst["logL"] = -np.inf
    proposal.populate(worst, n_samples=200)
    x = proposal.samples
    assert len(x) > 0
    assert np.all(np.isfinite(model.log_prior(x)))
