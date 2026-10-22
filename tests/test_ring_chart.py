"""Tests for the ring chart of two separated interferometers
(nessai_gw.sheet_chart.RingChart)."""

import pickle

import numpy as np
import pytest
from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model

from nessai_gw._ellipse import _complex_response
from nessai_gw._geometry import geocenter_time_delay, greenwich_mean_sidereal_time
from nessai_gw.sheet_chart import (
    LATTICE,
    RingChart,
    RingChartFlowProposal,
    _wrap,
    ring_chart_proposal_class,
)

T0 = 1187008882.4
#: ET 2L (MisA): ET_1L_IT (Sardinia) and ET_1L_DE (Lusatia), arms 45 deg apart
TENSORS = np.array([
    [[-0.19200120924503938, -0.11476687904642796, 0.24361711846573672],
     [-0.11476687904642796, 0.4809661654487176, 0.04040336345524519],
     [0.24361711846573672, 0.04040336345524519, -0.28896495620367824]],
    [[0.18433201627456766, -0.3437777575365753, -0.07620072004017435],
     [-0.3437777575365753, -0.18433201627456766, 0.3033677825093413],
     [-0.07620072004017435, 0.3033677825093413, 0.0]],
])
VERTICES = np.array([
    [4790201.377886417, 794444.3662803672, 4121767.028097608],
    [3877750.579027443, 974023.6218030028, 4952740.915258101],
])
NAMES = [
    "chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance",
    "theta_jn", "psi", "phase", "ra", "dec", "geocent_time",
]
BOUNDS = {
    "chirp_mass": (1.1, 1.3), "mass_ratio": (0.5, 1.0), "chi_1": (-0.05, 0.05),
    "chi_2": (-0.05, 0.05), "luminosity_distance": (1e-3, 1e5), "theta_jn": (0.0, np.pi),
    "psi": (0.0, np.pi), "phase": (0.0, 2 * np.pi), "ra": (0.0, 2 * np.pi),
    "dec": (-np.pi / 2, np.pi / 2), "geocent_time": (T0 - 0.1, T0 + 0.1),
}
SHARES = np.array([0.1, 0.7, 0.1, 0.1])


def make_chart(sign=None):
    return RingChart(TENSORS, VERTICES, T0, NAMES, phase_sign=sign)


def ring_points(n=800, sign=-1, noise=0.01, seed=0, spread=0.3):
    """Points near the ring degeneracy of a fiducial signal, spread along
    the azimuth about the baseline, with known branches."""
    rng = np.random.default_rng(seed)
    chart = make_chart()
    gmst = greenwich_mean_sidereal_time(T0)
    ra_f, dec_f, theta_f, psi_f, phase_f = 3.4462, -0.4081, 1.0, 1.57, 1.57
    z_f = _complex_response(TENSORS, np.array([ra_f]), np.array([dec_f]), gmst)
    a_f, b_f = chart._alpha_beta(theta_f, 1.0, psi_f, phase_f, sign)
    amps_f = a_f * z_f[0] + b_f * np.conj(z_f[0])
    az_f, cos_f = chart._ring(np.array([ra_f]), np.array([dec_f]))
    az = az_f + spread * rng.normal(size=n)
    cos_alpha = cos_f + 1e-3 * rng.normal(size=n)
    ra, dec = chart._sky(az, cos_alpha)
    z = _complex_response(TENSORS, ra, dec, gmst)
    eps = noise * rng.normal(size=(4, n))
    amps = amps_f[None, :] * np.exp(eps[:2].T + 1j * eps[2:].T)
    det = z[:, 0] * np.conj(z[:, 1]) - np.conj(z[:, 0]) * z[:, 1]
    a = (amps[:, 0] * np.conj(z[:, 1]) - np.conj(z[:, 0]) * amps[:, 1]) / det
    b = (z[:, 0] * amps[:, 1] - z[:, 1] * amps[:, 0]) / det
    sa, sb = np.sqrt(np.abs(a)), np.sqrt(np.abs(b))
    A, B = np.angle(a), np.angle(b)
    branch = rng.choice(4, size=n, p=SHARES)
    x = numpy_array_to_live_points(np.zeros((n, len(NAMES))), NAMES)
    x["chirp_mass"] = 1.2 + 1e-5 * rng.normal(size=n)
    x["mass_ratio"] = rng.uniform(0.7, 1.0, n)
    x["chi_1"] = rng.uniform(-0.04, 0.04, n)
    x["chi_2"] = rng.uniform(-0.04, 0.04, n)
    x["luminosity_distance"] = (sa + sb) ** -2
    x["theta_jn"] = np.arccos((sa - sb) / (sa + sb))
    x["psi"] = np.remainder((B - A) / 4 + LATTICE[branch, 0], np.pi)
    x["phase"] = np.remainder(sign * (A + B) / 4 + LATTICE[branch, 1], 2 * np.pi)
    x["ra"], x["dec"] = ra, dec
    x["geocent_time"] = T0 + 1e-4 * rng.normal(size=n) - geocenter_time_delay(VERTICES[0], gmst, ra, dec)
    x["logL"] = -np.sum(eps**2, axis=0) / noise**2
    return x, branch


def test_round_trip_and_branches():
    x, branch = ring_points()
    chart = make_chart().fit(x)
    y, k, log_j = chart.forward(x)
    out, valid, log_j_inv = chart.inverse(y, k)
    assert valid.all()
    np.testing.assert_allclose(log_j, log_j_inv, atol=1e-8)
    for n in NAMES:
        period = {"ra": 2 * np.pi, "psi": np.pi, "phase": 2 * np.pi}.get(n)
        d = _wrap(x[n] - out[n], period) if period else x[n] - out[n]
        assert np.max(np.abs(d)) < 1e-7 * max(1.0, np.max(np.abs(x[n]))), n
    table = np.zeros((4, 4), dtype=int)
    np.add.at(table, (branch, k), 1)
    assert table.max(axis=1).sum() >= 0.99 * len(x)
    np.testing.assert_allclose(np.sort(chart.p_k), np.sort(SHARES), atol=0.05)


def test_residuals_narrow_and_whitened():
    x, _ = ring_points(noise=0.01)
    chart = make_chart().fit(x)
    c, _, _ = chart.columns_of(x)
    sd = c.std(axis=0)
    for n in ("ring_ln_a1", "ring_ln_a2", "ring_arg_a1", "ring_arg_a2"):
        assert sd[chart.columns.index(n)] < 0.02, n
    assert sd[chart.columns.index("ring_cos_alpha")] < 2e-3
    # psi, cos theta_jn and the distance move along the ring
    assert np.std(np.cos(x["theta_jn"])) > 0.05
    y, _, _ = chart.forward(x)
    np.testing.assert_allclose(np.cov(y, rowvar=False), np.eye(chart.dims), atol=1e-6)


@pytest.mark.parametrize("sign", [-1, 1])
def test_phase_sign_detected(sign):
    x, _ = ring_points(sign=sign)
    assert make_chart().fit(x).phase_sign == sign


def test_jacobian_matches_finite_differences():
    x, _ = ring_points()
    chart = make_chart().fit(x)
    for i in (0, 10, 100):
        p0 = x[i : i + 1]
        c0, _, lj = chart.columns_of(p0)
        jac = np.zeros((len(NAMES), len(NAMES)))
        for j, n in enumerate(NAMES):
            p = p0.copy()
            h = 1e-3 if n == "geocent_time" else 1e-7 * max(1.0, abs(float(p0[n][0])))
            p[n] = p0[n] + h
            jac[:, j] = (chart.columns_of(p)[0][0] - c0[0]) / h
        assert abs(np.linalg.slogdet(jac)[1] - lj[0]) < 1e-3


def test_inverse_outside_domain_and_far_tails(recwarn):
    x, _ = ring_points()
    chart = make_chart().fit(x)
    y, k, _ = chart.forward(x[:5])
    c = y @ chart.chol.T + chart.mean
    c[0, chart.columns.index("ring_cos_alpha")] = 1.2
    c[1, chart.columns.index("ring_arg_a2")] = 4.0
    c[2, chart.columns.index("ring_azimuth")] = -3.5
    y = np.linalg.solve(chart.chol, (c - chart.mean).T).T
    y[3, :] = 1e37
    out, valid, log_j = chart.inverse(y, k)
    assert valid.tolist() == [False, False, False, False, True]
    assert np.isfinite(log_j).all()
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_geometry_checked():
    with pytest.raises(ValueError, match="two interferometers"):
        RingChart(np.zeros((3, 3, 3)), np.zeros((3, 3)), T0, NAMES)
    with pytest.raises(ValueError, match="baseline"):
        RingChart(TENSORS, np.stack([VERTICES[0], VERTICES[0]]), T0, NAMES)
    with pytest.raises(RuntimeError, match="fitted"):
        make_chart().forward(ring_points(n=50)[0])


class RingModel(Model):
    def __init__(self):
        self.names = list(NAMES)
        self.bounds = {n: np.array(b) for n, b in BOUNDS.items()}

    def log_prior(self, x):
        return np.where(self.in_bounds(x), 0.0, -np.inf)

    def log_likelihood(self, x):
        return np.zeros(x.size)


@pytest.fixture()
def proposal(tmp_path):
    model = RingModel()
    prop = RingChartFlowProposal(
        model,
        TENSORS,
        VERTICES,
        T0,
        flow_config={"n_blocks": 2, "n_neurons": 16, "n_layers": 1},
        training_config={"max_epochs": 5, "patience": 5, "batch_size": 256},
        output=str(tmp_path),
        poolsize=100,
        plot=False,
    )
    prop.initialise()
    return prop, model


def test_proposal_density_consistent(proposal):
    prop, model = proposal
    x, _ = ring_points()
    prop.train(x, plot=False)
    z = prop.sample_latent_distribution(500)
    xs, log_q = prop.backward_pass(z)
    assert len(xs) > 0
    _, log_q_fwd = prop.forward_pass(xs)
    # float32 flow; geocent_time ~ 1e9 s resolved to ~2e-7 s against an
    # arrival-time spread of 1e-4 s
    np.testing.assert_allclose(log_q, log_q_fwd, atol=0.05)
    _, k, _ = prop.chart.forward(xs)
    shares = np.bincount(k, minlength=4) / len(k)
    np.testing.assert_allclose(shares, prop.chart.p_k, atol=0.08)


def test_proposal_pickle_resume(proposal, tmp_path):
    prop, model = proposal
    x, _ = ring_points()
    prop.train(x, plot=False)
    _, log_q = prop.forward_pass(x[:50])
    weights = str(tmp_path / "w.pt")
    prop.flow.save_weights(weights)
    copy = pickle.loads(pickle.dumps(prop))
    copy.resume(model, {"n_blocks": 7, "ftype": "spline"}, weights)
    _, log_q_copy = copy.forward_pass(x[:50])
    np.testing.assert_allclose(log_q, log_q_copy, atol=1e-6)


def test_default_flow_config():
    prop = RingChartFlowProposal(RingModel(), TENSORS, VERTICES, T0, poolsize=10, plot=False)
    assert prop.sheet_flow_config["n_blocks"] == 8


def test_bound_proposal_class(tmp_path):
    """The geometry bound into the class: nessai passes only the keyword
    arguments it knows, and checkpoints unpickle through the module-scope
    name."""
    from nessai.proposal.utils import check_proposal_kwargs

    cls = ring_chart_proposal_class(TENSORS, VERTICES, T0, phase_sign=-1)
    kwargs = check_proposal_kwargs(
        cls, {"poolsize": 100, "plot": False, "output": str(tmp_path), "flow_config": {"n_blocks": 2}}
    )
    prop = cls(RingModel(), **kwargs)
    assert prop.chart.fixed_phase_sign == -1
    np.testing.assert_allclose(prop.chart.tensors, TENSORS)
    assert prop.sheet_flow_config == {"n_blocks": 2}
    copy = pickle.loads(pickle.dumps(prop))
    assert type(copy).__name__ == "BoundRingChartFlowProposal"
    np.testing.assert_allclose(copy.chart.vertices, VERTICES)


def test_branches_are_the_lattice():
    """The four (psi, phase) shifts of the lattice leave the chart's
    coordinates as they are, get four distinct labels and invert to
    themselves."""
    x, _ = ring_points()
    chart = make_chart().fit(x)
    y0, _, _ = chart.forward(x)
    labels = []
    for dpsi, dphase in LATTICE:
        xg = x.copy()
        xg["psi"] = np.remainder(x["psi"] + dpsi, np.pi)
        xg["phase"] = np.remainder(x["phase"] + dphase, 2 * np.pi)
        y, k, _ = chart.forward(xg)
        np.testing.assert_allclose(y, y0, atol=1e-8)
        out, valid, _ = chart.inverse(y, k)
        assert valid.all()
        assert np.max(np.abs(_wrap(out["psi"] - xg["psi"], np.pi))) < 1e-8
        assert np.max(np.abs(_wrap(out["phase"] - xg["phase"]))) < 1e-8
        labels.append(k)
    assert np.all(np.sort(np.stack(labels), axis=0) == np.arange(4)[:, None])
