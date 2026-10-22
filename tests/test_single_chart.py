"""Tests for the chart of a single interferometer
(nessai_gw.sheet_chart.SingleChart)."""

import pickle

import numpy as np
import pytest
from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model

from nessai_gw._ellipse import _complex_response
from nessai_gw._geometry import geocenter_time_delay, greenwich_mean_sidereal_time
from nessai_gw.sheet_chart import (
    LATTICE,
    SingleChart,
    SingleChartFlowProposal,
    _wrap,
    single_chart_proposal_class,
)

T0 = 1187008882.4
#: the 40 km CE (bilby_xG's CE)
TENSOR = np.array([
    [-0.11982305924327216, 0.37690377769476086, 0.2320083726716809],
    [0.37690377769476086, -0.1072073484615781, 0.12417231500238687],
    [0.2320083726716809, 0.12417231500238687, 0.2270304077048501],
])
VERTEX = np.array([-2545772.6288342914, -3635740.1054883287, 4565350.08533747])
NAMES = [
    "chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance",
    "theta_jn", "psi", "phase", "ra", "dec", "geocent_time",
]
BOUNDS = {
    "chirp_mass": (20, 40), "mass_ratio": (0.125, 1.0), "chi_1": (-0.99, 0.99),
    "chi_2": (-0.99, 0.99), "luminosity_distance": (1e-6, 1e6), "theta_jn": (0.0, np.pi),
    "psi": (0.0, np.pi), "phase": (0.0, 2 * np.pi), "ra": (0.0, 2 * np.pi),
    "dec": (-np.pi / 2, np.pi / 2), "geocent_time": (T0 - 0.1, T0 + 0.1),
}
SHARES = np.array([0.1, 0.7, 0.1, 0.1])


def make_chart(sign=None):
    return SingleChart(TENSOR, VERTEX, T0, NAMES, phase_sign=sign)


def single_points(n=800, sign=-1, noise=0.01, seed=0):
    """Points on the degeneracy of a fiducial amplitude: broad sky,
    inclination and polarisation, the amplitude and the detector time
    narrow, with known branches."""
    rng = np.random.default_rng(seed)
    gmst = greenwich_mean_sidereal_time(T0)
    amp_f = 3e-2 * np.exp(0.4j)
    ra = rng.uniform(0, 2 * np.pi, n)
    dec = np.arcsin(rng.uniform(-0.95, 0.95, n))
    cos_theta = rng.uniform(-0.9, 0.9, n)
    psi0 = rng.uniform(0, np.pi / 2, n)
    eps = noise * rng.normal(size=(2, n))
    amp = amp_f * np.exp(eps[0] + 1j * eps[1])
    z = _complex_response(TENSOR[None], ra, dec, gmst)[:, 0]
    rho = ((1 - cos_theta) / (1 + cos_theta)) ** 2 * np.exp(4j * psi0)
    alpha = amp / (z + rho * np.conj(z))
    phase0 = np.remainder((np.angle(alpha) + 2 * psi0) / (2 * sign), np.pi)
    branch = rng.choice(4, size=n, p=SHARES)
    x = numpy_array_to_live_points(np.zeros((n, len(NAMES))), NAMES)
    x["chirp_mass"] = 30 + 0.1 * rng.normal(size=n)
    x["mass_ratio"] = rng.uniform(0.5, 1.0, n)
    x["chi_1"] = rng.uniform(-0.5, 0.5, n)
    x["chi_2"] = rng.uniform(-0.5, 0.5, n)
    x["luminosity_distance"] = (1 + cos_theta) ** 2 / 4 / np.abs(alpha)
    x["theta_jn"] = np.arccos(cos_theta)
    x["psi"] = np.remainder(psi0 + LATTICE[branch, 0], np.pi)
    x["phase"] = np.remainder(phase0 + LATTICE[branch, 1], 2 * np.pi)
    x["ra"], x["dec"] = ra, dec
    x["geocent_time"] = T0 + 1e-4 * rng.normal(size=n) - geocenter_time_delay(VERTEX, gmst, ra, dec)
    x["logL"] = -np.sum(eps**2, axis=0) / noise**2
    return x, branch


def test_round_trip_and_branches():
    x, branch = single_points()
    chart = make_chart().fit(x)
    y, k, log_j = chart.forward(x)
    out, valid, log_j_inv = chart.inverse(y, k)
    assert valid.all()
    np.testing.assert_allclose(log_j, log_j_inv, atol=1e-8)
    for n in NAMES:
        period = {"ra": 2 * np.pi, "psi": np.pi, "phase": 2 * np.pi}.get(n)
        d = _wrap(x[n] - out[n], period) if period else x[n] - out[n]
        assert np.max(np.abs(d)) < 1e-7 * max(1.0, np.max(np.abs(x[n]))), n
    np.testing.assert_allclose(np.sort(chart.p_k), np.sort(SHARES), atol=0.05)


def test_branches_are_the_lattice():
    """The four (psi, phase) shifts of the lattice leave the chart's
    coordinates as they are, get four distinct labels and invert to
    themselves. (The labels need not match the generator's branches: the
    chart takes psi modulo pi / 2 about its own centre.)"""
    x, _ = single_points()
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


def test_residuals_narrow_and_whitened():
    x, _ = single_points(noise=0.01)
    chart = make_chart().fit(x)
    c, _, _ = chart.columns_of(x)
    sd = c.std(axis=0)
    for n in ("single_ln_amplitude", "single_arg_amplitude"):
        assert sd[chart.columns.index(n)] < 0.02, n
    # the phase the group proposal uses stays broad on a broad sky
    delta_phase = x["phase"] + np.sign(np.cos(x["theta_jn"])) * x["psi"]
    assert abs(np.mean(np.exp(2j * delta_phase))) < 0.5
    y, _, _ = chart.forward(x)
    np.testing.assert_allclose(np.cov(y, rowvar=False), np.eye(chart.dims), atol=1e-6)


@pytest.mark.parametrize("sign", [-1, 1])
def test_phase_sign_detected(sign):
    x, _ = single_points(sign=sign)
    assert make_chart().fit(x).phase_sign == sign


def test_jacobian_matches_finite_differences():
    x, _ = single_points()
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
    x, _ = single_points()
    chart = make_chart().fit(x)
    y, k, _ = chart.forward(x[:5])
    c = y @ chart.chol.T + chart.mean
    c[0, chart.columns.index("single_cos_theta_jn")] = 1.1
    c[1, chart.columns.index("single_psi4")] = 3.5
    c[2, chart.columns.index("single_sin_dec")] = -1.2
    y = np.linalg.solve(chart.chol, (c - chart.mean).T).T
    y[3, :] = 1e37
    out, valid, log_j = chart.inverse(y, k)
    assert valid.tolist() == [False, False, False, False, True]
    assert np.isfinite(log_j).all()
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_geometry_checked():
    with pytest.raises(ValueError, match="one interferometer"):
        SingleChart(np.zeros((2, 3, 3)), VERTEX, T0, NAMES)
    with pytest.raises(RuntimeError, match="fitted"):
        make_chart().forward(single_points(n=50)[0])


class SingleModel(Model):
    def __init__(self):
        self.names = list(NAMES)
        self.bounds = {n: np.array(b) for n, b in BOUNDS.items()}

    def log_prior(self, x):
        return np.where(self.in_bounds(x), 0.0, -np.inf)

    def log_likelihood(self, x):
        return np.zeros(x.size)


@pytest.fixture()
def proposal(tmp_path):
    model = SingleModel()
    prop = SingleChartFlowProposal(
        model,
        TENSOR,
        VERTEX,
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
    x, _ = single_points()
    prop.train(x, plot=False)
    z = prop.sample_latent_distribution(500)
    xs, log_q = prop.backward_pass(z)
    assert len(xs) > 0
    _, log_q_fwd = prop.forward_pass(xs)
    np.testing.assert_allclose(log_q, log_q_fwd, atol=0.02)
    _, k, _ = prop.chart.forward(xs)
    shares = np.bincount(k, minlength=4) / len(k)
    np.testing.assert_allclose(shares, prop.chart.p_k, atol=0.08)


def test_proposal_pickle_resume(proposal, tmp_path):
    prop, model = proposal
    x, _ = single_points()
    prop.train(x, plot=False)
    _, log_q = prop.forward_pass(x[:50])
    weights = str(tmp_path / "w.pt")
    prop.flow.save_weights(weights)
    copy = pickle.loads(pickle.dumps(prop))
    copy.resume(model, {"n_blocks": 7, "ftype": "spline"}, weights)
    _, log_q_copy = copy.forward_pass(x[:50])
    np.testing.assert_allclose(log_q, log_q_copy, atol=1e-6)


def test_bound_proposal_class(tmp_path):
    from nessai.proposal.utils import check_proposal_kwargs

    cls = single_chart_proposal_class(TENSOR, VERTEX, T0, phase_sign=-1)
    kwargs = check_proposal_kwargs(cls, {"poolsize": 100, "plot": False, "output": str(tmp_path)})
    prop = cls(SingleModel(), **kwargs)
    assert prop.chart.fixed_phase_sign == -1
    assert prop.sheet_flow_config["n_blocks"] == 8
    copy = pickle.loads(pickle.dumps(prop))
    assert type(copy).__name__ == "BoundSingleChartFlowProposal"
    np.testing.assert_allclose(copy.chart.tensor, TENSOR)
