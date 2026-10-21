"""Tests for the sheet chart (nessai_gw.sheet_chart)."""

import pickle

import numpy as np
import pytest
from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model

from nessai_gw._ellipse import PolarisationEllipse
from nessai_gw._geometry import geocenter_time_delay, greenwich_mean_sidereal_time
from nessai_gw.group_mixture import ET_EMR_PLANE_NORMAL, ET_EMR_VERTEX
from nessai_gw.sheet_chart import (
    LATTICE,
    SheetChart,
    SheetChartFlowProposal,
    SheetChartRegions,
    _wrap,
)

T0 = 1187008882.4
NAMES = [
    "chirp_mass",
    "mass_ratio",
    "chi_1",
    "chi_2",
    "luminosity_distance",
    "theta_jn",
    "psi",
    "phase",
    "ra",
    "dec",
    "geocent_time",
]
BOUNDS = {
    "chirp_mass": (1.1, 1.3),
    "mass_ratio": (0.5, 1.0),
    "chi_1": (-0.05, 0.05),
    "chi_2": (-0.05, 0.05),
    "luminosity_distance": (1.0, 1e5),
    "theta_jn": (0.0, np.pi),
    "psi": (0.0, np.pi),
    "phase": (0.0, 2 * np.pi),
    "ra": (0.0, 2 * np.pi),
    "dec": (-np.pi / 2, np.pi / 2),
    "geocent_time": (T0 - 0.1, T0 + 0.1),
}
SHARES = np.array([0.1, 0.7, 0.1, 0.1])


def sheet_points(n=800, sign=-1, noise=0.01, seed=0):
    """Points near the sheet of a face-on fiducial, in an edge-on patch of
    the sky, with known branches."""
    rng = np.random.default_rng(seed)
    fid = dict(ra=3.4462, dec=-0.4081, psi=1.57, theta_jn=0.3491)
    ellipse = PolarisationEllipse(ET_EMR_PLANE_NORMAL, T0, fid)
    # an edge-on patch: centre where |cos iota*| is small
    ra_g = rng.uniform(0, 2 * np.pi, 4000)
    dec_g = np.arcsin(rng.uniform(-1, 1, 4000))
    i0 = np.argmin(np.abs(ellipse.cos_iota(ra_g, dec_g)))
    ra = np.remainder(ra_g[i0] + 0.15 * rng.normal(size=n), 2 * np.pi)
    dec = np.clip(dec_g[i0] + 0.1 * rng.normal(size=n), -1.5, 1.5)
    a0, b0 = ellipse.coefficients(ra, dec)
    scale = 2e-3 * np.exp(0.7j)
    eps = noise * (rng.normal(size=(4, n)))
    alpha = scale * a0 * np.exp(eps[0] + 1j * eps[1])
    beta = scale * b0 * np.exp(eps[2] + 1j * eps[3])
    sa, sb = np.sqrt(np.abs(alpha)), np.sqrt(np.abs(beta))
    theta = np.arccos((sa - sb) / (sa + sb))
    dist = (sa + sb) ** -2
    A, B = np.angle(alpha), np.angle(beta)
    branch = rng.choice(4, size=n, p=SHARES)
    psi = np.remainder((B - A) / 4 + LATTICE[branch, 0], np.pi)
    phase = np.remainder(sign * (A + B) / 4 + LATTICE[branch, 1], 2 * np.pi)
    gmst = greenwich_mean_sidereal_time(T0)
    t = T0 + 1e-4 * rng.normal(size=n) - geocenter_time_delay(ET_EMR_VERTEX, gmst, ra, dec)
    x = numpy_array_to_live_points(np.zeros((n, len(NAMES))), NAMES)
    x["chirp_mass"] = 1.2 + 1e-5 * rng.normal(size=n)
    x["mass_ratio"] = rng.uniform(0.7, 1.0, n)
    x["chi_1"] = rng.uniform(-0.04, 0.04, n)
    x["chi_2"] = rng.uniform(-0.04, 0.04, n)
    x["luminosity_distance"] = dist
    x["theta_jn"] = theta
    x["psi"] = psi
    x["phase"] = phase
    x["ra"] = ra
    x["dec"] = dec
    x["geocent_time"] = t
    x["logL"] = -np.sum(eps**2, axis=0) / noise**2
    return x, branch


def make_chart(sign=None):
    return SheetChart(ET_EMR_PLANE_NORMAL, ET_EMR_VERTEX, T0, NAMES, phase_sign=sign)


def test_round_trip_and_branches():
    x, branch = sheet_points()
    chart = make_chart().fit(x)
    y, k, log_j = chart.forward(x)
    out, valid, log_j_inv = chart.inverse(y, k)
    assert valid.all()
    np.testing.assert_allclose(log_j, log_j_inv, atol=1e-9)
    for n in NAMES:
        period = {"ra": 2 * np.pi, "psi": np.pi, "phase": 2 * np.pi}.get(n)
        d = _wrap(x[n] - out[n], period) if period else x[n] - out[n]
        assert np.max(np.abs(d)) < 1e-8 * max(1.0, np.max(np.abs(x[n]))), n
    # the label is a relabelling of the branch the points were made on
    # (but for the few the generator's own principal-value cut relabels)
    table = np.zeros((4, 4), dtype=int)
    np.add.at(table, (branch, k), 1)
    assert table.max(axis=1).sum() >= 0.99 * len(x)
    np.testing.assert_allclose(np.sort(chart.p_k), np.sort(SHARES), atol=0.05)


def test_residuals_narrow_and_whitened():
    x, _ = sheet_points(noise=0.01)
    chart = make_chart().fit(x)
    c, _, _ = chart.columns_of(x)
    sd = c.std(axis=0)
    names = chart.columns
    for n in ("sheet_ln_alpha", "sheet_ln_beta", "sheet_arg_alpha", "sheet_arg_beta"):
        assert sd[names.index(n)] < 0.03, n
    assert np.std(x["psi"]) > 0.1
    y, _, _ = chart.forward(x)
    np.testing.assert_allclose(np.cov(y, rowvar=False), np.eye(chart.dims), atol=1e-6)


@pytest.mark.parametrize("sign", [-1, 1])
def test_phase_sign_detected(sign):
    x, _ = sheet_points(sign=sign)
    assert make_chart().fit(x).phase_sign == sign


def test_jacobian_matches_finite_differences():
    x, _ = sheet_points()
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
        # geocent_time ~ 1e9: its finite difference is good to ~1e-6
        assert abs(np.linalg.slogdet(jac)[1] - lj[0]) < 1e-3


def test_inverse_outside_domain():
    x, _ = sheet_points()
    chart = make_chart().fit(x)
    y, k, _ = chart.forward(x[:3])
    c = y @ chart.chol.T + chart.mean
    j = chart.columns.index("sheet_sin_dec")
    c[0, j] = 1.5
    j = chart.columns.index("sheet_arg_alpha")
    c[1, j] = 4.0
    out, valid, _ = chart.inverse(np.linalg.solve(chart.chol, (c - chart.mean).T).T, k)
    assert valid.tolist() == [False, False, True]


class SheetModel(Model):
    def __init__(self):
        self.names = list(NAMES)
        self.bounds = {n: np.array(b) for n, b in BOUNDS.items()}

    def log_prior(self, x):
        return np.where(self.in_bounds(x), 0.0, -np.inf)

    def log_likelihood(self, x):
        return np.zeros(x.size)


@pytest.fixture()
def proposal(tmp_path):
    model = SheetModel()
    prop = SheetChartFlowProposal(
        model,
        ET_EMR_PLANE_NORMAL,
        ET_EMR_VERTEX,
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
    x, _ = sheet_points()
    prop.train(x, plot=False)
    z = prop.sample_latent_distribution(500)
    xs, log_q = prop.backward_pass(z)
    assert len(xs) > 0
    _, log_q_fwd = prop.forward_pass(xs)
    # float32 flow, and geocent_time ~ 1e9 s resolved to ~2e-7 s against a
    # vertex-time spread of 1e-4 s; a wrong branch share or Jacobian would
    # be off by O(1)
    np.testing.assert_allclose(log_q, log_q_fwd, atol=0.02)
    # the drawn branches follow the fitted shares
    _, k, _ = prop.chart.forward(xs)
    shares = np.bincount(k, minlength=4) / len(k)
    np.testing.assert_allclose(shares, prop.chart.p_k, atol=0.08)


def test_proposal_pickle_resume(proposal, tmp_path):
    prop, model = proposal
    x, _ = sheet_points()
    prop.train(x, plot=False)
    _, log_q = prop.forward_pass(x[:50])
    weights = str(tmp_path / "w.pt")
    prop.flow.save_weights(weights)
    copy = pickle.loads(pickle.dumps(prop))
    # the sampler passes its main proposal's flow configuration: ignored
    copy.resume(model, {"n_blocks": 7, "ftype": "spline"}, weights)
    _, log_q_copy = copy.forward_pass(x[:50])
    np.testing.assert_allclose(log_q, log_q_copy, atol=1e-6)


def test_regions_factory_criterion():
    x, _ = sheet_points()
    factory = SheetChartRegions(ET_EMR_PLANE_NORMAL, ET_EMR_VERTEX, T0)
    assert factory.wants(x)
    face_on = x.copy()
    face_on["theta_jn"] = 0.3
    assert not factory.wants(face_on)
    pickle.loads(pickle.dumps(factory))
