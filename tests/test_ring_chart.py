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


def pole_points(n=40, seed=1):
    """Points within 0.2 rad of face-on and of face-off, anywhere on the sky."""
    rng = np.random.default_rng(seed)
    x, _ = ring_points(n=n, seed=seed)
    theta = rng.uniform(1e-3, 0.2, n)
    x["theta_jn"] = np.where(np.arange(n) % 2, theta, np.pi - theta)
    x["ra"] = rng.uniform(0, 2 * np.pi, n)
    x["dec"] = np.arcsin(rng.uniform(-0.9, 0.9, n))
    x["psi"] = rng.uniform(0, np.pi, n)
    x["phase"] = rng.uniform(0, 2 * np.pi, n)
    return x


def test_pole_regularisation_round_trip_and_jacobian():
    x, _ = ring_points()
    chart = RingChart(TENSORS, VERTICES, T0, NAMES, phase_sign=-1, pole_regularisation=0.1).fit(x)
    for pts in (x, pole_points()):
        y, k, log_j = chart.forward(pts)
        out, valid, log_j_inv = chart.inverse(y, k)
        assert valid.all()
        np.testing.assert_allclose(log_j, log_j_inv, atol=1e-8)
        for n in NAMES:
            period = {"ra": 2 * np.pi, "psi": np.pi, "phase": 2 * np.pi}.get(n)
            d = _wrap(pts[n] - out[n], period) if period else pts[n] - out[n]
            assert np.max(np.abs(d)) < 1e-7 * max(1.0, np.max(np.abs(pts[n]))), n
    pts = pole_points()
    for i in (0, 1, 7):
        p0 = pts[i : i + 1]
        c0, _, lj = chart.columns_of(p0)
        jac = np.zeros((len(NAMES), len(NAMES)))
        for j, n in enumerate(NAMES):
            p = p0.copy()
            h = 1e-3 if n == "geocent_time" else 1e-7 * max(1.0, abs(float(p0[n][0])))
            p[n] = p0[n] + h
            jac[:, j] = (chart.columns_of(p)[0][0] - c0[0]) / h
        assert abs(np.linalg.slogdet(jac)[1] - lj[0]) < 1e-3


def test_pole_regularisation_smooths_the_prior():
    """The isotropic prior's density in the chart, sin(theta_jn) / |dc/dx|,
    is ~ theta_jn^-6 towards face-on in the plain chart and flat with the
    regularisation."""
    x, _ = ring_points()
    p = np.repeat(x[:1], 3)
    p["theta_jn"] = [1e-2, 1e-3, np.pi - 1e-3]
    for t0, step in ((0.0, 6 * np.log(10)), (0.1, 0.0)):
        chart = RingChart(TENSORS, VERTICES, T0, NAMES, phase_sign=-1, pole_regularisation=t0).fit(x)
        _, _, log_j = chart.columns_of(p)
        log_density = np.log(np.sin(p["theta_jn"])) - log_j
        assert abs(log_density[1] - log_density[0] - step) < 0.05
        if t0:
            assert abs(log_density[2] - log_density[1]) < 5


def test_pole_regularisation_off_is_the_plain_chart():
    x, _ = ring_points()
    plain = make_chart(-1).fit(x)
    zero = RingChart(TENSORS, VERTICES, T0, NAMES, phase_sign=-1, pole_regularisation=0.0).fit(x)
    for a, b in zip(plain.forward(x), zero.forward(x)):
        np.testing.assert_array_equal(a, b)
    # a chart pickled before the option has no instance attribute
    state = pickle.loads(pickle.dumps(plain))
    del state.__dict__["pole_regularisation"]
    for a, b in zip(plain.forward(x), state.forward(x)):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="pole_regularisation"):
        RingChart(TENSORS, VERTICES, T0, NAMES, pole_regularisation=1.5)


def test_bound_proposal_class_pole_regularisation(tmp_path):
    cls = ring_chart_proposal_class(TENSORS, VERTICES, T0, phase_sign=-1, pole_regularisation=0.1,
                                    class_name="BoundRingChartFlowProposalPole")
    prop = cls(RingModel(), poolsize=100, plot=False, output=str(tmp_path))
    assert prop.chart.pole_regularisation == 0.1
    copy = pickle.loads(pickle.dumps(prop))
    assert copy.chart.pole_regularisation == 0.1


def test_arcs_round_trip_and_labels():
    """With ``arcs`` the ring is cut at the two crossings of det M = 0; the
    chart stays exact and the arc is a label next to the branch."""
    x, _ = ring_points(n=1500, spread=1.2)
    chart = RingChart(TENSORS, VERTICES, T0, NAMES, phase_sign=-1, pole_regularisation=0.1,
                      arcs=True).fit(x)
    assert len(chart.arc_centres) == 2
    assert chart.n_labels == 8 and len(chart.p_k) == 8
    np.testing.assert_allclose(chart.arc_half_widths.sum(), np.pi)
    y, k, log_j = chart.forward(x)
    out, valid, log_j_inv = chart.inverse(y, k)
    assert valid.all()
    np.testing.assert_allclose(log_j, log_j_inv, atol=1e-8)
    for n in NAMES:
        period = {"ra": 2 * np.pi, "psi": np.pi, "phase": 2 * np.pi}.get(n)
        d = _wrap(x[n] - out[n], period) if period else x[n] - out[n]
        assert np.max(np.abs(d)) < 1e-7 * max(1.0, np.max(np.abs(x[n]))), n
    # each point's arc is the one its azimuth is on
    arc, _ = chart._arc_of(chart._ring(x["ra"], x["dec"])[0])
    np.testing.assert_array_equal(k // len(LATTICE), arc)
    # the azimuth column is relative to the arc's centre: beyond its half
    # width a draw is outside the chart
    c = chart.columns_of(x[:1])[0].copy()
    i = chart.columns.index("ring_azimuth")
    c[0, i] = chart.arc_half_widths[0] + 0.01
    assert not chart._inverse(c, np.array([0]))[1][0]
    # without arcs: one arc, the plain chart's labels
    plain = RingChart(TENSORS, VERTICES, T0, NAMES, phase_sign=-1, pole_regularisation=0.1).fit(x)
    assert plain.arc_centres is None and plain.n_labels == len(LATTICE)


def test_population_to_cap():
    """Rejection weights min(w, c) / c with population_to_cap, the batch
    maximum otherwise."""
    from types import SimpleNamespace
    from nessai_gw.sheet_chart import RingChartFlowProposal
    prop = object.__new__(RingChartFlowProposal)
    prop._truncation_scheme = SimpleNamespace(log_weight_cap=1.0)
    prop.population_weight_cap_mass = None
    prop.clip_population_weights = False
    log_w = np.array([-3.0, 0.0, 1.0])
    np.testing.assert_allclose(prop._get_population_log_weights(log_w), [-4.0, -1.0, 0.0])
    prop.population_to_cap = True
    np.testing.assert_allclose(prop._get_population_log_weights(log_w - 2.0), [-6.0, -3.0, -2.0])
    np.testing.assert_allclose(prop._get_population_log_weights(log_w + 1.0), [-3.0, 0.0, 0.0])
