"""Tests for the charts' intrinsic coordinates
(nessai_gw.sheet_chart._Chart.set_coordinates): the group proposals'
effective spins and tides, Doppler chirp mass and the time and phase at
reference frequencies, in the sheet, ring and single charts."""

import pickle

import numpy as np
import pytest
from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model
from test_ring_chart import TENSORS, VERTICES, ring_points
from test_sheet_chart import sheet_points
from test_single_chart import TENSOR, VERTEX, single_points

from nessai_gw.group_mixture import ET_EMR_PLANE_NORMAL, ET_EMR_VERTEX
from nessai_gw.reparameterisations.mass import doppler_factor
from nessai_gw.reparameterisations.time import pn_time_to_merger
from nessai_gw.sheet_chart import (
    RingChart,
    RingChartFlowProposal,
    SheetChart,
    SheetChartRegions,
    SingleChart,
    _wrap,
    ring_chart_proposal_class,
)

T0 = 1187008882.4
BASE = [
    "chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance",
    "theta_jn", "psi", "phase", "ra", "dec", "geocent_time",
]
NAMES = BASE[:4] + ["lambda_1", "lambda_2"] + BASE[4:]
BOUNDS = {
    "chirp_mass": (1.1, 1.3), "mass_ratio": (0.5, 1.0), "chi_1": (-0.05, 0.05),
    "chi_2": (-0.05, 0.05), "lambda_1": (5.0, 2000.0), "lambda_2": (5.0, 2000.0),
    "luminosity_distance": (1e-3, 1e5), "theta_jn": (0.0, np.pi),
    "psi": (0.0, np.pi), "phase": (0.0, 2 * np.pi), "ra": (0.0, 2 * np.pi),
    "dec": (-np.pi / 2, np.pi / 2), "geocent_time": (T0 - 0.1, T0 + 0.1),
}
DOPPLER = [-0.5e-4, 0.8e-4, 0.3e-4]


def waveform_phase(params, frequencies):
    """A smooth stand-in for a waveform's 22-mode phase."""
    f = np.asarray(frequencies, dtype=float)[None, :]
    p = {k: np.asarray(v, dtype=float)[:, None] for k, v in params.items()}
    return (
        30.0 * (p["chirp_mass"] / 1.2) ** (-5 / 3) * (f / 50.0) ** (-5 / 3)
        + 4.0 * p["mass_ratio"] * np.log(f)
        + 20.0 * p["chi_1"] * f / 50.0
        + 2e-3 * (p["lambda_1"] + p["lambda_2"]) * (f / 100.0) ** (5 / 3)
    )


def time_to_merger(params, frequencies):
    """2PN plus a tidal stand-in."""
    f = np.asarray(frequencies, dtype=float)
    tau = np.stack([
        pn_time_to_merger(params["chirp_mass"], params["mass_ratio"], params["chi_1"],
                          params["chi_2"], fi)
        for fi in f
    ], axis=1)
    return tau - 1e-6 * (params["lambda_1"] + params["lambda_2"])[:, None]


COORDINATES = dict(
    effective_spin=True,
    effective_tidal_deformability=True,
    doppler_vector=DOPPLER,
    time_reference_frequency="adaptive",
    time_to_merger=time_to_merger,
    waveform_phase=waveform_phase,
)


def with_tides(x, seed=0):
    """``x`` with the tides, and the phase wound by the waveform's phase
    at 40 Hz so that a reference frequency concentrates it."""
    rng = np.random.default_rng(seed)
    out = numpy_array_to_live_points(np.zeros((len(x), len(NAMES))), NAMES)
    for n in BASE + ["logL"]:
        out[n] = x[n]
    out["lambda_1"] = rng.uniform(100, 900, len(x))
    out["lambda_2"] = rng.uniform(300, 1500, len(x))
    params = {n: out[n] for n in ("chirp_mass", "mass_ratio", "chi_1", "chi_2",
                                  "lambda_1", "lambda_2")}
    params["chirp_mass"] = params["chirp_mass"] * doppler_factor(out["ra"], out["dec"], DOPPLER)
    f, step = 40.0, 1e-3
    phi = waveform_phase(params, [f * (1 - step), f, f * (1 + step)])
    intercept = phi[:, 1] - (phi[:, 2] - phi[:, 0]) / (2 * step)
    out["phase"] = np.remainder(out["phase"] - 0.5 * intercept, 2 * np.pi)
    return out


def ring():
    return RingChart(TENSORS, VERTICES, T0, NAMES, phase_sign=-1, pole_regularisation=0.1)


def sheet():
    return SheetChart(ET_EMR_PLANE_NORMAL, ET_EMR_VERTEX, T0, NAMES, phase_sign=-1)


def single():
    return SingleChart(TENSOR, VERTEX, T0, NAMES, phase_sign=-1)


#: (the single chart's points are a BBH's)
BBH_BOUNDS = dict(BOUNDS, chirp_mass=(20.0, 40.0), chi_1=(-0.99, 0.99), chi_2=(-0.99, 0.99))
CASES = {
    "ring": (ring, lambda: ring_points()[0], BOUNDS),
    "sheet": (sheet, lambda: sheet_points()[0], BOUNDS),
    "single": (single, lambda: single_points()[0], BBH_BOUNDS),
}


def fitted(kind, **coordinates):
    make, points, bounds = CASES[kind]
    x = with_tides(points())
    chart = make().set_coordinates(bounds, **(coordinates or COORDINATES))
    return chart.fit(x), x


@pytest.mark.parametrize("kind", CASES)
def test_round_trip(kind):
    chart, x = fitted(kind)
    y, k, log_j = chart.forward(x)
    out, valid, log_j_inv = chart.inverse(y, k)
    assert valid.all()
    np.testing.assert_allclose(log_j, log_j_inv, atol=1e-7)
    for n in NAMES:
        period = {"ra": 2 * np.pi, "psi": np.pi, "phase": 2 * np.pi}.get(n)
        d = _wrap(x[n] - out[n], period) if period else x[n] - out[n]
        assert np.max(np.abs(d)) < 1e-7 * max(1.0, np.max(np.abs(x[n]))), n


@pytest.mark.parametrize("kind", CASES)
def test_jacobian_matches_finite_differences(kind):
    """The time and phase shifts depend on the intrinsic parameters and the
    Doppler chirp mass on the sky: the Jacobian stays triangular in the
    order (sky, intrinsic, time, amplitudes)."""
    chart, x = fitted(kind)
    for i in (0, 10, 100):
        p0 = x[i : i + 1]
        c0, _, lj = chart.columns_of(p0)
        jac = np.zeros((len(NAMES), len(NAMES)))
        for j, n in enumerate(NAMES):
            p = p0.copy()
            # (a GPS time resolves ~2.4e-7 s; the time column is linear in it)
            h = 1e-2 if n == "geocent_time" else 1e-7 * max(1.0, abs(float(p0[n][0])))
            p[n] = p0[n] + h
            jac[:, j] = (chart.columns_of(p)[0][0] - c0[0]) / h
        assert abs(np.linalg.slogdet(jac)[1] - lj[0]) < 1e-3


@pytest.mark.parametrize("kind", CASES)
def test_columns_and_reference_frequencies(kind):
    chart, x = fitted(kind)
    prefix = chart.prefix
    for n in ("doppler_chirp_mass", "chi_eff_prime", "chi_diff_prime",
              "lambda_tilde_prime", "lambda_diff_prime", "ln_mass_ratio"):
        assert prefix + n in chart.columns, n
    assert chart._phase.reference_frequency is not None
    assert chart._time.reference_frequency is not None
    assert "phase at" in chart.describe() and "time at" in chart.describe()
    # the waveform's phase is taken out: the phase columns narrow
    plain, _ = fitted(kind, effective_spin=True)
    c, _, _ = chart.columns_of(x)
    c0, _, _ = plain.columns_of(x)
    args = [n for n in chart.columns if "arg" in n]
    narrower = [np.std(c[:, chart.columns.index(n)]) < 0.5 * np.std(c0[:, plain.columns.index(n)])
                for n in args]
    assert any(narrower)


def test_doppler_chirp_mass_column():
    chart, x = fitted("ring", doppler_vector=DOPPLER)
    c, _, _ = chart.columns_of(x)
    np.testing.assert_allclose(
        c[:, chart.columns.index("ring_doppler_chirp_mass")],
        x["chirp_mass"] * doppler_factor(x["ra"], x["dec"], DOPPLER),
    )
    assert chart._time is None and chart._phase is None and chart._spin is None


def test_no_coordinates_is_the_plain_chart():
    """``set_coordinates`` with nothing on, and a chart pickled before the
    option, are the plain chart."""
    x = with_tides(ring_points()[0])
    plain = ring().fit(x)
    none = ring().set_coordinates(BOUNDS).fit(x)
    assert plain.columns == none.columns
    for a, b in zip(plain.forward(x), none.forward(x)):
        np.testing.assert_array_equal(a, b)
    old = pickle.loads(pickle.dumps(plain))
    for key in ("coordinates", "doppler_vector", "_spin", "_tides", "_time", "_phase"):
        old.__dict__.pop(key, None)
    for a, b in zip(plain.forward(x), old.forward(x)):
        np.testing.assert_array_equal(a, b)


def test_coordinates_checked():
    with pytest.raises(ValueError, match="time_to_merger"):
        ring().set_coordinates(BOUNDS, time_to_merger=time_to_merger)
    names = [n for n in NAMES if n != "chirp_mass"]
    chart = RingChart(TENSORS, VERTICES, T0, names, phase_sign=-1)
    with pytest.raises(ValueError, match="chirp_mass"):
        chart.set_coordinates(BOUNDS, doppler_vector=DOPPLER)
    with pytest.raises(ValueError, match="unknown chart coordinates"):
        ring_chart_proposal_class(TENSORS, VERTICES, T0, coordinates={"chirp_distance": True},
                                  class_name="BoundRingChartFlowProposalBad")


class TidalModel(Model):
    def __init__(self):
        self.names = list(NAMES)
        self.bounds = {n: np.array(b) for n, b in BOUNDS.items()}

    def log_prior(self, x):
        return np.where(self.in_bounds(x), 0.0, -np.inf)

    def log_likelihood(self, x):
        return np.zeros(x.size)


def test_proposal_with_coordinates(tmp_path):
    """Through the bound proposal class (string waveform functions, as
    xg_pe passes them): trains, draws, scores consistently and pickles."""
    coordinates = dict(COORDINATES, time_to_merger="test_chart_coordinates:time_to_merger",
                       waveform_phase="test_chart_coordinates:waveform_phase")
    cls = ring_chart_proposal_class(TENSORS, VERTICES, T0, phase_sign=-1, pole_regularisation=0.1,
                                    coordinates=coordinates,
                                    class_name="BoundRingChartFlowProposalCoordinates")
    model = TidalModel()
    prop = cls(model, poolsize=100, plot=False, output=str(tmp_path),
               flow_config={"n_blocks": 2, "n_neurons": 16, "n_layers": 1},
               training_config={"max_epochs": 5, "patience": 5, "batch_size": 256})
    assert isinstance(prop, RingChartFlowProposal)
    assert "ring_lambda_tilde_prime" in prop.chart.columns
    prop.initialise()
    x = with_tides(ring_points()[0])
    prop.train(x, plot=False)
    assert prop.chart._phase.reference_frequency is not None
    xs, log_q = prop.backward_pass(prop.sample_latent_distribution(500))
    assert len(xs) > 0
    _, log_q_fwd = prop.forward_pass(xs)
    np.testing.assert_allclose(log_q, log_q_fwd, atol=0.05)
    copy = pickle.loads(pickle.dumps(prop))
    y0 = prop.chart.forward(x[:20])[0]
    np.testing.assert_allclose(copy.chart.forward(x[:20])[0], y0)


def test_regions_pass_coordinates():
    regions = SheetChartRegions(ET_EMR_PLANE_NORMAL, ET_EMR_VERTEX, T0, phase_sign=-1,
                                coordinates={"doppler_vector": np.array(DOPPLER)})
    assert regions.coordinates == {"doppler_vector": DOPPLER}
    # (regions pickled before the option)
    del regions.__dict__["coordinates"]
    assert getattr(regions, "coordinates", None) is None
