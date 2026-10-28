"""Tests for the ``sheet-residual`` polarisation/phase coordinates:
:class:`nessai_gw._ellipse.SheetPhase`,
:class:`nessai_gw.reparameterisations.phase.SheetResidualReparameterisation`
and their wiring into the triangle's 32-element group proposal."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nessai_gw._ellipse import PolarisationEllipse, SheetPhase  # noqa: E402
from nessai_gw.group_mixture import (  # noqa: E402
    ET_EMR_PLANE_NORMAL,
    CircularCanonicalTransform,
    ETTriangleGroupAction,
    PrimeSpaceTriangularGroupAction,
    _prime_parameter_names,
    make_et_group_flow_proposal,
    triangular_group_reparameterisations,
)
from nessai_gw.reparameterisations.phase import (  # noqa: E402
    SheetResidualReparameterisation,
)


def _load_circular_tests():
    # --import-mode=importlib: a sibling test module is loaded by path
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).with_name("test_circular.py")
    spec = importlib.util.spec_from_file_location("_test_circular", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_tc = _load_circular_tests()
BNS_PARAMETERS = _tc.BNS_PARAMETERS
_check_trained_proposal = _tc._check_trained_proposal
_prior_live_points = _tc._prior_live_points
_stub_model = _tc._stub_model
requires_circular = _tc.requires_circular


@pytest.fixture(autouse=True)
def float64():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)

REFERENCE_TIME = 1187008882.4
#: an edge-on fiducial, so the sheet's vortices are generic
FIDUCIAL = {"ra": 3.4462, "dec": -0.4081, "psi": 1.57, "theta_jn": 1.3}
PRIME_NAMES = ["sky_u", "sky_v", "psi_res", "delta_phase_res", "theta_jn_prime",
               "t_det"]


def _sheet(offsets=(0.3, -1.1)):
    sheet = SheetPhase(PolarisationEllipse(
        ET_EMR_PLANE_NORMAL, REFERENCE_TIME, FIDUCIAL))
    sheet.offset_psi, sheet.offset_sum = offsets
    return sheet


def _physical(n, seed=3):
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=[(k, "f8") for k in
                           ("ra", "dec", "psi", "phase", "theta_jn", "logL")])
    x["ra"] = rng.uniform(0, 2 * np.pi, n)
    x["dec"] = np.arcsin(rng.uniform(-1, 1, n))
    x["psi"] = rng.uniform(0, np.pi, n)
    x["phase"] = rng.uniform(0, 2 * np.pi, n)
    x["theta_jn"] = np.arccos(rng.uniform(-1, 1, n))
    x["logL"] = rng.normal(size=n)
    return x


def _circ(a, period):
    d = np.mod(a, period)
    return np.minimum(d, period - d)


# -- SheetPhase ---------------------------------------------------------------

def test_sheet_phase_torch_matches_numpy():
    sheet = _sheet()
    x = _physical(500)
    psi, phase = sheet.predict(x["ra"], x["dec"])
    tpsi, tphase = sheet.predict_torch(
        torch.as_tensor(x["ra"]), torch.as_tensor(np.sin(x["dec"])))
    # principal values may sit on opposite sides of a branch line only at
    # round-off; compare modulo the quarter-turn lattice
    assert _circ(4 * (tpsi.numpy() - psi), 2 * np.pi).max() < 1e-8
    assert _circ(2 * (tpsi.numpy() + tphase.numpy() - psi - phase),
                 2 * np.pi).max() < 1e-8


def test_sheet_phase_reproduces_fiducial():
    """On the sheet the fiducial predicts its own polarisation."""
    sheet = _sheet(offsets=(0.0, 0.0))
    psi, _ = sheet.predict(np.array([FIDUCIAL["ra"]]),
                           np.array([FIDUCIAL["dec"]]))
    assert _circ(psi[0] - FIDUCIAL["psi"], np.pi / 2) < 1e-8


def test_sheet_phase_refit_centres_and_moves_fiducial():
    sheet = _sheet(offsets=(0.0, 0.0))
    x = _physical(2000)
    # live points on the sheet of a different fiducial, with small scatter
    other = SheetPhase(PolarisationEllipse(
        ET_EMR_PLANE_NORMAL, REFERENCE_TIME,
        {"ra": 1.0, "dec": 0.4, "psi": 0.2, "theta_jn": 1.7}))
    psi_s, phase_s = other.predict(x["ra"], x["dec"])
    rng = np.random.default_rng(0)
    x["psi"] = np.mod(psi_s + 0.03 * rng.normal(size=len(x)), np.pi)
    x["phase"] = np.mod(phase_s + 0.7 + 0.03 * rng.normal(size=len(x)),
                        2 * np.pi)
    best = int(np.argmax(x["logL"]))
    sheet.refit(x)
    assert sheet.ellipse.fiducial["ra"] == pytest.approx(x["ra"][best])
    psi, phase = sheet.predict(x["ra"], x["dec"])
    r_psi = np.mod(x["psi"] - psi, np.pi / 2)
    r_sum = np.mod(x["psi"] + x["phase"] - psi - phase, np.pi)
    assert np.median(_circ(r_psi - np.pi / 4, np.pi / 2)) < 0.1
    assert np.median(_circ(r_sum - np.pi / 2, np.pi)) < 0.1


# -- the reparameterisation ---------------------------------------------------

def _reparam(sheet=None):
    return SheetResidualReparameterisation(
        parameters=["psi", "phase"],
        prior_bounds={"psi": [0, np.pi], "phase": [0, 2 * np.pi]},
        sheet=_sheet() if sheet is None else sheet)


def _apply(rep, x):
    xp = np.zeros(len(x), dtype=[(n, "f8") for n in rep.prime_parameters])
    _, xp, lj = rep.reparameterise(x.copy(), xp, np.zeros(len(x)))
    return xp, lj


def test_reparam_round_trip_and_range():
    rep = _reparam()
    x = _physical(3000)
    xp, lj = _apply(rep, x)
    for n in rep.prime_parameters:
        assert xp[n].min() >= -1 and xp[n].max() < 1
    y = x.copy()
    y["psi"] = y["phase"] = np.nan
    y, _, lj_inv = rep.inverse_reparameterise(y, xp, np.zeros(len(x)))
    assert _circ(y["psi"] - x["psi"], np.pi).max() < 1e-9
    assert _circ(y["phase"] - x["phase"], 2 * np.pi).max() < 1e-9
    np.testing.assert_allclose(lj, -lj_inv)


def test_reparam_log_jacobian_numerical():
    rep = _reparam()
    x = _physical(200)
    xp, lj = _apply(rep, x)
    eps = 1e-6
    cols = []
    for k in ("psi", "phase"):
        y = x.copy()
        y[k] = y[k] + eps
        yp, _ = _apply(rep, y)
        cols.append(np.stack([
            (yp[n] - xp[n] + 1) % 2 - 1 for n in rep.prime_parameters
        ], -1) / eps)
    jac = np.stack(cols, -1)
    np.testing.assert_allclose(
        np.log(np.abs(np.linalg.det(jac))), lj, atol=1e-5)


def test_reparam_quarter_turn_glues_straight():
    """The polarisation/phase quarter turn moves psi_res by one (half its
    range) and leaves delta_phase_res, so the folded cell is a product torus."""
    rep = _reparam()
    x = _physical(3000)
    xp, _ = _apply(rep, x)
    y = x.copy()
    y["psi"] = np.mod(y["psi"] + np.pi / 2, np.pi)
    y["phase"] = np.mod(y["phase"] - np.pi / 2, 2 * np.pi)
    yp, _ = _apply(rep, y)
    assert _circ(yp["psi_res"] - xp["psi_res"] - 1.0, 2.0).max() < 1e-9
    # delta_phase_res is unchanged, or shifted by the half turn (one) when psi_res
    # wraps
    d = _circ(yp["delta_phase_res"] - xp["delta_phase_res"], 1.0)
    assert d.max() < 1e-9


def test_reparam_update_refits_sheet():
    rep = _reparam()
    x = _physical(500)
    before = dict(rep.sheet.ellipse.fiducial)
    rep.update(x)
    assert rep.sheet.ellipse.fiducial != before


# -- the prime-space group action ---------------------------------------------

@pytest.fixture(scope="module")
def action():
    base = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, polarisation_quarter=True,
        phase_reflection=True)
    return PrimeSpaceTriangularGroupAction(base, PRIME_NAMES, sheet=_sheet())


@pytest.fixture(scope="module")
def prime_points():
    rng = np.random.default_rng(11)
    n = 4000
    return {
        "sky_u": torch.as_tensor(rng.uniform(0, 1, n)),
        "sky_v": torch.as_tensor(rng.uniform(0, 1, n)),
        "psi_res": torch.as_tensor(rng.uniform(-1, 1, n)),
        "delta_phase_res": torch.as_tensor(rng.uniform(-1, 1, n)),
        "theta_jn_prime": torch.as_tensor(rng.uniform(-1, 1, n)),
        "t_det": torch.as_tensor(rng.uniform(-20, 20, n)),
    }


def test_action_round_trip(action, prime_points):
    n = len(prime_points["t_det"])
    for g in range(action._action.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        back = action(action(prime_points, modes), modes, inverse=True)
        for name in ("psi_res", "delta_phase_res"):
            d = torch.remainder(back[name] - prime_points[name], 2.0)
            d = torch.minimum(d, 2.0 - d)
            assert d.max() < 1e-8, (g, name)


def test_action_matches_physical(action, prime_points):
    """The prime action equals the physical action mapped through the
    reparameterisation."""
    rep = _reparam(sheet=action._sheet)
    n = len(prime_points["t_det"])
    acted, _ = action._decode(prime_points)
    phys0 = {"ra": acted["ra"], "sin_dec": acted["sin_dec"],
             "cos_theta_jn": torch.full_like(acted["ra"], 0.3),
             "psi": acted["psi"], "phase": acted["phase"]}
    base = action._action
    for g in range(base.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        phys = base(phys0, modes)
        x = np.zeros(n, dtype=[(k, "f8") for k in ("ra", "dec", "psi",
                                                    "phase")])
        x["ra"] = phys["ra"].numpy()
        x["dec"] = np.arcsin(phys["sin_dec"].numpy())
        x["psi"] = phys["psi"].numpy()
        x["phase"] = phys["phase"].numpy()
        want, _ = _apply(rep, x)
        fwd = action(prime_points, modes)
        for name in ("psi_res", "delta_phase_res"):
            d = torch.remainder(fwd[name] - torch.as_tensor(want[name]), 2.0)
            d = torch.minimum(d, 2.0 - d)
            assert d.max() < 1e-6, (g, name)


def test_fundamental_domain_partitions_orbit(action, prime_points):
    n = len(prime_points["t_det"])
    count = torch.zeros(n)
    for g in range(action._action.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        count += action.in_fundamental_domain(
            action(prime_points, modes)).float()
    assert torch.all(count == 1), count.unique(return_counts=True)


def test_fundamental_domain_is_the_residual_cell(action, prime_points):
    inside = action.in_fundamental_domain(prime_points)
    pts = {k: v[inside] for k, v in prime_points.items()}
    assert (pts["psi_res"] < 0).all() and (pts["delta_phase_res"] < 0).all()


# -- factory wiring -----------------------------------------------------------

def test_reparameterisations_and_prime_names():
    sheet = _sheet()
    reps = triangular_group_reparameterisations(
        BNS_PARAMETERS, REFERENCE_TIME, phase_coordinates="sheet-residual",
        sheet_phase=sheet)
    assert reps["sheet-residual"]["sheet"] is sheet
    assert "phase" not in reps and "psi" not in reps
    prime = _prime_parameter_names(
        BNS_PARAMETERS, REFERENCE_TIME, psi_single=True,
        phase_coordinates="sheet-residual", sheet_phase=sheet)
    assert "psi_res" in prime and "delta_phase_res" in prime
    assert not {"psi_prime", "delta_phase", "psi_x"} & set(prime)


def test_sheet_phase_required_both_ways():
    with pytest.raises(ValueError, match="sheet"):
        triangular_group_reparameterisations(
            BNS_PARAMETERS, REFERENCE_TIME, phase_coordinates="sheet-residual")
    with pytest.raises(ValueError, match="sheet"):
        triangular_group_reparameterisations(
            BNS_PARAMETERS, REFERENCE_TIME, sheet_phase=_sheet())


def test_needs_polarisation_quarter():
    with pytest.raises(ValueError, match="polarisation_quarter"):
        make_et_group_flow_proposal(
            BNS_PARAMETERS, REFERENCE_TIME, phase_reflection=True,
            polarisation_quarter=False, phase_coordinates="sheet-residual",
            sheet_phase=_sheet())


@requires_circular
@pytest.mark.parametrize("standardise", [False, True])
def test_et_proposal_circular_end_to_end(tmp_path, standardise):
    """sheet-residual on the circular torus: psi_res and delta_phase_res are the
    circles, and a proposal trains and samples."""
    sheet = _sheet()
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, phase_reflection=True,
        polarisation_quarter=True, psi_single=True,
        phase_coordinates="sheet-residual", sheet_phase=sheet,
        circular_psi_phase=True, log_mass_ratio=False,
        circular_standardisation=standardise,
    )
    fm = cls._FlowModelClass
    assert fm.circular_parameters == ["psi_res", "delta_phase_res"]
    assert isinstance(fm.canonical_transform, CircularCanonicalTransform)
    model = _stub_model()
    proposal = cls(
        model, output=str(tmp_path), poolsize=200,
        reparameterisations=triangular_group_reparameterisations(
            BNS_PARAMETERS, REFERENCE_TIME, log_mass_ratio=False,
            phase_coordinates="sheet-residual", sheet_phase=sheet),
        flow_config={"ftype": "circular", "n_blocks": 2, "n_neurons": 8,
                     "n_layers": 1, "real_transform": "affine"},
        training_config={"max_epochs": 3, "patience": 3},
        latent_temperature=1.4,
    )
    proposal.initialise()
    circ = _check_trained_proposal(
        proposal, model, _prior_live_points(model, 1000))
    assert set(circ) == {"psi_res", "delta_phase_res"}


def test_et_proposal_plain_end_to_end(tmp_path):
    import pickle

    sheet = _sheet()
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, phase_reflection=True,
        polarisation_quarter=True, psi_single=True,
        phase_coordinates="sheet-residual", sheet_phase=sheet,
        log_mass_ratio=False,
    )
    model = _stub_model()
    proposal = cls(
        model, output=str(tmp_path), poolsize=200,
        reparameterisations=triangular_group_reparameterisations(
            BNS_PARAMETERS, REFERENCE_TIME, log_mass_ratio=False,
            phase_coordinates="sheet-residual", sheet_phase=sheet),
        flow_config={"n_blocks": 2, "n_neurons": 8, "n_layers": 1},
        training_config={"max_epochs": 3, "patience": 3},
    )
    proposal.initialise()
    assert {"psi_res", "delta_phase_res"} <= set(proposal.prime_parameters)
    live = _prior_live_points(model, 1000)
    proposal.train(live)
    flow = proposal.flow.model
    with torch.no_grad():
        x, log_q = flow.sample_and_log_prob(500)
        np.testing.assert_allclose(
            log_q, flow.log_prob(x), rtol=1e-5, atol=1e-5)
    worst = live[:1].copy()
    worst["logL"] = -np.inf
    proposal.populate(worst, n_samples=200)
    assert len(proposal.samples) > 0
    assert np.all(np.isfinite(model.log_prior(proposal.samples)))
    # one SheetPhase, shared by the reparameterisation and the group action,
    # also after a checkpoint
    weights_file = proposal.flow.weights_file
    resumed = pickle.loads(pickle.dumps(proposal))
    resumed.resume(model, dict(proposal.flow_config), weights_file)
    reps = [r for r in resumed._reparameterisation.values()
            if isinstance(r, SheetResidualReparameterisation)]
    assert len(reps) == 1
    assert reps[0].sheet is not sheet  # unpickled
    assert reps[0].sheet is resumed.flow.model.prime_space_action._sheet
    assert reps[0].sheet.ellipse.fiducial == sheet.ellipse.fiducial


# -- the phase at a reference frequency ---------------------------------------

def _wf_phase(params, frequencies):
    """Stand-in 22-mode phase (as in test_phase): a time term the tangent
    intercept removes, a chirp term, a constant."""
    f = np.atleast_1d(frequencies)[None, :]
    chirp = (30.0 * (params["chirp_mass"] - 1.2) / 0.01)[:, None]
    return (-2 * np.pi * f * 0.02 * params["chi_1"][:, None]
            + chirp * (f / 30.0) ** (-5.0 / 3.0)
            + 5.0 * params["chi_2"][:, None])


_F_TRUE = float(np.geomspace(8.0, 300.0, 33)[19])


def _intercept(params, f):
    chirp = 30.0 * (params["chirp_mass"] - 1.2) / 0.01
    return 5.0 * params["chi_2"] + chirp * (8.0 / 3.0) * (f / 30.0) ** (-5.0 / 3.0)


def _helix_on_sheet(sheet, n, seed):
    """Points on the sheet whose phase at the merger winds with the chirp
    mass: phase = phase*(n) - a(f_true) / 2 + noise."""
    rng = np.random.default_rng(seed)
    names = ("ra", "dec", "psi", "phase", "theta_jn", "chirp_mass",
             "mass_ratio", "chi_1", "chi_2", "logL")
    x = np.zeros(n, dtype=[(k, "f8") for k in names])
    x["ra"] = rng.uniform(0, 2 * np.pi, n)
    x["dec"] = np.arcsin(rng.uniform(-1, 1, n))
    x["theta_jn"] = np.arccos(rng.uniform(-1, 1, n))
    x["chirp_mass"] = rng.uniform(1.19, 1.21, n)
    x["mass_ratio"] = rng.uniform(0.7, 1.0, n)
    x["chi_1"] = rng.uniform(-0.05, 0.05, n)
    x["chi_2"] = rng.uniform(-0.05, 0.05, n)
    x["logL"] = rng.normal(size=n)
    params = {k: x[k] for k in ("chirp_mass", "mass_ratio", "chi_1", "chi_2")}
    params |= {"lambda_1": 0 * x["ra"], "lambda_2": 0 * x["ra"]}
    # on the sheet: the inclination too, so that any point is a fiducial
    # with the same response pattern
    x["theta_jn"] = np.arccos(sheet.ellipse.cos_iota(x["ra"], x["dec"]))
    psi_s, phase_s = sheet.predict(x["ra"], x["dec"])
    x["psi"] = np.mod(psi_s + rng.normal(0, 0.03, n), np.pi)
    x["phase"] = np.mod(phase_s - _intercept(params, _F_TRUE) / 2
                        + rng.normal(0, 0.03, n), 2 * np.pi)
    return x


def test_waveform_phase_on_the_sheet():
    sheet = _sheet(offsets=(0.0, 0.0))
    rep = SheetResidualReparameterisation(
        parameters=["psi", "phase"],
        prior_bounds={"psi": [0, np.pi], "phase": [0, 2 * np.pi]},
        sheet=sheet, waveform_phase=_wf_phase, tides=None)
    assert {"ra", "dec", "chirp_mass", "mass_ratio", "chi_1",
            "chi_2"} <= set(rep.requires)
    x = _helix_on_sheet(sheet, 3000, 5)
    plain = _reparam(sheet=_sheet(offsets=(0.0, 0.0)))
    xp0, lj0 = _apply(plain, x)
    conc = lambda v: abs(np.mean(np.exp(2j * np.pi * (v + 1))))  # noqa: E731
    assert conc(xp0["delta_phase_res"]) < 0.3        # winds with the chirp mass
    rep.update(x)
    assert rep._phase_ref.reference_frequency == pytest.approx(_F_TRUE)
    xp, lj = _apply(rep, x)
    assert conc(xp["delta_phase_res"]) > 0.9
    np.testing.assert_allclose(lj, lj0)
    y = x.copy()
    y["psi"] = y["phase"] = np.nan
    y, _, _ = rep.inverse_reparameterise(y, xp, np.zeros(len(x)))
    assert _circ(y["phase"] - x["phase"], 2 * np.pi).max() < 1e-9
    assert _circ(y["psi"] - x["psi"], np.pi).max() < 1e-9


@pytest.mark.parametrize("chirp_distance", [False, True])
def test_waveform_phase_entry_order(chirp_distance):
    """Ahead of the intrinsic parameters it reads, behind the chirp
    distance, which reads psi."""
    reps = triangular_group_reparameterisations(
        BNS_PARAMETERS, REFERENCE_TIME, phase_coordinates="sheet-residual",
        sheet_phase=_sheet(), waveform_phase="x:y", effective_spin=True,
        chirp_distance=chirp_distance, chirp_distance_k0=0)
    keys = list(reps)
    assert keys[int(chirp_distance)] == "sheet-residual"
    if chirp_distance:
        assert reps[keys[0]]["reparameterisation"] == "chirp-distance"
    assert reps["sheet-residual"]["waveform_phase"] == "x:y"


@pytest.mark.parametrize("chirp_distance", [False, True])
def test_et_proposal_waveform_phase_end_to_end(tmp_path, chirp_distance):
    """The inverse reads the intrinsic parameters, and the chirp distance's
    reads psi: nessai's ordering must give each its inputs (a round trip
    through the whole chain is the check)."""
    sheet = _sheet()
    wf = "test_chart_coordinates:waveform_phase"
    kw = dict(log_mass_ratio=True, effective_spin=True,
              phase_coordinates="sheet-residual", sheet_phase=sheet,
              waveform_phase=wf, chirp_distance=chirp_distance,
              chirp_distance_k0=0)
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, phase_reflection=True,
        polarisation_quarter=True, psi_single=True, **kw)
    model = _stub_model()
    proposal = cls(
        model, output=str(tmp_path), poolsize=200,
        reparameterisations=triangular_group_reparameterisations(
            BNS_PARAMETERS, REFERENCE_TIME, **kw),
        flow_config={"n_blocks": 2, "n_neurons": 8, "n_layers": 1},
        training_config={"max_epochs": 3, "patience": 3},
    )
    proposal.initialise()
    live = _prior_live_points(model, 1000)
    proposal.train(live)
    worst = live[:1].copy()
    worst["logL"] = -np.inf
    proposal.populate(worst, n_samples=200)
    assert len(proposal.samples) > 0
    assert np.all(np.isfinite(model.log_prior(proposal.samples)))
    # forward then inverse through nessai's full reparameterisation chain
    x = live[:300].copy()
    x_prime, lj = proposal.rescale(x)
    back, lj_back = proposal.inverse_rescale(x_prime)
    back = back[:len(x)]  # the leading rows; any augmented copies follow
    for k in ("psi", "phase", "chirp_mass", "mass_ratio", "chi_1",
              "luminosity_distance"):
        period = {"psi": np.pi, "phase": 2 * np.pi}.get(k)
        d = back[k] - x[k]
        if period:
            d = _circ(d, period)
        assert np.abs(d).max() < 1e-7, k


# -- ObservedPhase: the phase the detectors observe ---------------------------

def _observed(offsets=(0.4, -0.7), sign=-1.0, branch=0.9, hemisphere=False):
    from nessai_gw._ellipse import ObservedPhase
    from nessai_gw._geometry import greenwich_mean_sidereal_time
    from nessai_gw.group_mixture import ET_EMR_DETECTOR_TENSORS

    obs = ObservedPhase(ET_EMR_DETECTOR_TENSORS,
                        greenwich_mean_sidereal_time(REFERENCE_TIME))
    obs.offset_psi, obs.offset_sum = offsets
    obs.sign, obs.branch, obs.hemisphere = sign, branch, hemisphere
    return obs


@pytest.mark.parametrize("hemisphere", [False, True])
def test_observed_phase_torch_matches_numpy(hemisphere):
    obs = _observed(hemisphere=hemisphere)
    x = _physical(2000)
    c = np.cos(x["theta_jn"])
    want = obs.phase(x["ra"], x["dec"], c, x["psi"])
    got = obs.phase_torch(torch.as_tensor(x["ra"]), torch.as_tensor(np.sin(x["dec"])),
                          torch.as_tensor(c), torch.as_tensor(x["psi"])).numpy()
    assert _circ(got - want, 2 * np.pi).max() < 1e-9


def test_observed_phase_quarter_turn_invariant():
    """h is unchanged by the polarisation/phase quarter turn (mod pi)."""
    obs = _observed()
    x = _physical(2000)
    c = np.cos(x["theta_jn"])
    h0 = obs.observed(x["ra"], x["dec"], c, x["psi"])
    h1 = obs.observed(x["ra"], x["dec"], c, x["psi"] + np.pi / 2)
    assert _circ(h1 - h0, np.pi).max() < 1e-9


def test_observed_reparam_round_trip_and_quarter_turn():
    rep = _reparam(sheet=_observed())
    assert "theta_jn" in rep.requires
    x = _physical(3000)
    xp, lj = _apply(rep, x)
    y = x.copy()
    y["psi"] = y["phase"] = np.nan
    y, _, lj_inv = rep.inverse_reparameterise(y, xp, np.zeros(len(x)))
    assert _circ(y["psi"] - x["psi"], np.pi).max() < 1e-9
    assert _circ(y["phase"] - x["phase"], 2 * np.pi).max() < 1e-9
    np.testing.assert_allclose(lj, -lj_inv)
    y = x.copy()
    y["psi"] = np.mod(y["psi"] + np.pi / 2, np.pi)
    y["phase"] = np.mod(y["phase"] - np.pi / 2, 2 * np.pi)
    yp, _ = _apply(rep, y)
    assert _circ(yp["psi_res"] - xp["psi_res"] - 1.0, 2.0).max() < 1e-9
    assert _circ(yp["delta_phase_res"] - xp["delta_phase_res"], 1.0).max() < 1e-9


def test_observed_reparam_log_jacobian_numerical():
    rep = _reparam(sheet=_observed())
    x = _physical(200)
    xp, lj = _apply(rep, x)
    eps = 1e-6
    cols = []
    for k in ("psi", "phase"):
        y = x.copy()
        y[k] = y[k] + eps
        yp, _ = _apply(rep, y)
        cols.append(np.stack([(yp[n] - xp[n] + 1) % 2 - 1
                              for n in rep.prime_parameters], -1) / eps)
    np.testing.assert_allclose(
        np.log(np.abs(np.linalg.det(np.stack(cols, -1)))), lj, atol=1e-5)


def test_observed_refit_concentrates_a_wound_phase():
    """Points whose phase follows the observed phase of their sky and
    orientation: delta_phase winds, the residual is pinned."""
    truth = _observed(offsets=(0.0, 0.0), sign=1.0, branch=0.0)
    x = _physical(4000, seed=5)
    c = np.cos(x["theta_jn"])
    s = np.where(c >= 0, 1.0, -1.0)
    rng = np.random.default_rng(1)
    target = 1.0 + 0.05 * rng.normal(size=len(x))
    x["phase"] = np.mod(target - s * x["psi"]
                        - truth.observed(x["ra"], x["dec"], c, x["psi"]), 2 * np.pi)
    conc = lambda a: np.abs(np.mean(np.exp(2j * a)))
    assert conc(x["phase"] + s * x["psi"]) < 0.3
    rep = _reparam(sheet=_observed())
    rep.update(x)
    assert rep.sheet.sign == 1.0
    xp, _ = _apply(rep, x)
    assert conc(xp["delta_phase_res"] * np.pi) > 0.95


@pytest.fixture(scope="module", params=[False, True], ids=["fixed", "hemisphere"])
def observed_action(request):
    base = ETTriangleGroupAction(
        reference_time=REFERENCE_TIME, polarisation_quarter=True,
        phase_reflection=True)
    return PrimeSpaceTriangularGroupAction(
        base, PRIME_NAMES, sheet=_observed(hemisphere=request.param))


def test_observed_action_round_trip(observed_action, prime_points):
    n = len(prime_points["t_det"])
    for g in range(observed_action._action.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        back = observed_action(observed_action(prime_points, modes), modes, inverse=True)
        for name in ("psi_res", "delta_phase_res"):
            d = torch.remainder(back[name] - prime_points[name], 2.0)
            assert torch.minimum(d, 2.0 - d).max() < 1e-8, (g, name)


def test_observed_action_matches_physical(observed_action, prime_points):
    """With theta_jn's angle-sine coordinate u = 2 theta_jn / pi - 1: the
    prime action equals the physical action through the reparameterisation."""
    action = observed_action
    rep = _reparam(sheet=action._sheet)
    n = len(prime_points["t_det"])
    acted, _ = action._decode(prime_points)
    cos0 = -torch.sin(prime_points["theta_jn_prime"] * (0.5 * np.pi))
    phys0 = {"ra": acted["ra"], "sin_dec": acted["sin_dec"], "cos_theta_jn": cos0,
             "psi": acted["psi"], "phase": acted["phase"]}
    base = action._action
    for g in range(base.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        phys = base(phys0, modes)
        x = np.zeros(n, dtype=[(k, "f8") for k in ("ra", "dec", "psi", "phase",
                                                    "theta_jn")])
        x["ra"] = phys["ra"].numpy()
        x["dec"] = np.arcsin(phys["sin_dec"].numpy())
        x["psi"] = phys["psi"].numpy()
        x["phase"] = phys["phase"].numpy()
        x["theta_jn"] = np.arccos(np.clip(phys["cos_theta_jn"].numpy(), -1, 1))
        want, _ = _apply(rep, x)
        fwd = action(prime_points, modes)
        for name in ("psi_res", "delta_phase_res"):
            d = torch.remainder(fwd[name] - torch.as_tensor(want[name]), 2.0)
            assert torch.minimum(d, 2.0 - d).max() < 1e-6, (g, name)


def test_observed_fundamental_domain_partitions_orbit(observed_action, prime_points):
    n = len(prime_points["t_det"])
    count = torch.zeros(n)
    for g in range(observed_action._action.group_size):
        modes = torch.full((n,), g, dtype=torch.long)
        count += observed_action.in_fundamental_domain(
            observed_action(prime_points, modes)).float()
    assert torch.all(count == 1), count.unique(return_counts=True)


@requires_circular
def test_et_proposal_observed_phase_end_to_end(tmp_path):
    """An ObservedPhase in place of the sheet: a proposal trains and samples,
    and its training refits the prediction."""
    obs = _observed(offsets=(0.0, 0.0), sign=1.0, branch=0.0)
    cls = make_et_group_flow_proposal(
        BNS_PARAMETERS, REFERENCE_TIME, phase_reflection=True,
        polarisation_quarter=True, psi_single=True,
        phase_coordinates="sheet-residual", sheet_phase=obs,
        circular_psi_phase=True, log_mass_ratio=False,
        circular_standardisation=True,
    )
    model = _stub_model()
    proposal = cls(
        model, output=str(tmp_path), poolsize=200,
        reparameterisations=triangular_group_reparameterisations(
            BNS_PARAMETERS, REFERENCE_TIME, log_mass_ratio=False,
            phase_coordinates="sheet-residual", sheet_phase=obs),
        flow_config={"ftype": "circular", "n_blocks": 2, "n_neurons": 8,
                     "n_layers": 1, "real_transform": "affine"},
        training_config={"max_epochs": 3, "patience": 3},
        latent_temperature=1.4,
    )
    proposal.initialise()
    circ = _check_trained_proposal(
        proposal, model, _prior_live_points(model, 1000))
    assert set(circ) == {"psi_res", "delta_phase_res"}
