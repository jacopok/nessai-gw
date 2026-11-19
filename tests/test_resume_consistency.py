"""End to end: a run with the polarisation ellipse, checkpointed and resumed,
scores its samples exactly as the uninterrupted run did.

The ellipse's fitted lock slope and width are held by the theta_jn
reparameterisation and, separately, by the group action in the flow model.
A resume once rebuilt the flow model with the configuration's unfitted
ellipse, so the resumed run's density was not the one its samples were drawn
from (plain nessai: the action's ellipse was not the reparameterisation's;
i-nessai: every level used the class's one action, and the group mixture's
domain mass drifted with each draw)."""

import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nessai.livepoint import live_points_to_array  # noqa: E402
from nessai.model import Model  # noqa: E402

from nessai_gw._ellipse import PolarisationEllipse  # noqa: E402
from nessai_gw.group_mixture import (  # noqa: E402
    ET_EMR_PLANE_NORMAL,
    make_et_group_flow_proposal,
    triangular_group_reparameterisations,
)
from nessai_gw.reparameterisations import (  # noqa: E402
    PolarisationEllipseReparameterisation,
)

REFERENCE_TIME = 1187008882.4
FIDUCIAL = dict(ra=3.4462, dec=-0.4081, psi=1.57, theta_jn=0.3491)
PARAMETERS = [
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
#: cos theta_jn follows the sheet's inclination at this slope, so the fitted
#: lock slope (and the adaptive width) differ from the configuration's
LOCK = 0.6


@pytest.fixture(autouse=True)
def float64():
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)


def _ellipse():
    ellipse = PolarisationEllipse(ET_EMR_PLANE_NORMAL, REFERENCE_TIME, FIDUCIAL)
    ellipse.fit_lock = True
    return ellipse


class SheetModel(Model):
    """Uniform prior; the likelihood puts cos theta_jn at LOCK times the
    inclination the sky forces on the sheet, sharper with ``sigma``."""

    def __init__(self, ellipse, sigma=0.1):
        self.names = list(PARAMETERS)
        self.bounds = {k: np.asarray(BOUNDS[k]) for k in PARAMETERS}
        self._ellipse = ellipse
        self.sigma = sigma

    def log_prior(self, x):
        with np.errstate(divide="ignore"):
            return np.log(self.in_bounds(x), dtype=float)

    def to_unit_hypercube(self, x):
        u = x.copy()
        for n in self.names:
            lo, hi = self.bounds[n]
            u[n] = (x[n] - lo) / (hi - lo)
        return u

    def from_unit_hypercube(self, u):
        x = u.copy()
        for n in self.names:
            lo, hi = self.bounds[n]
            x[n] = lo + (hi - lo) * u[n]
        return x

    def log_unit_hypercube_jacobian(self, x):
        log_volume = sum(np.log(hi - lo) for lo, hi in self.bounds.values())
        return np.full(len(x), -log_volume)

    def log_likelihood(self, x):
        centre = self._ellipse.cos_iota(x["ra"], x["dec"])
        r = (np.cos(x["theta_jn"]) - LOCK * centre) / self.sigma
        return -0.5 * r**2


def _proposal_class(ellipse):
    cls = make_et_group_flow_proposal(
        PARAMETERS, REFERENCE_TIME, phase_reflection=True,
        polarisation_quarter=True, psi_single=True, log_mass_ratio=False,
        polarisation_ellipse=ellipse, polarisation_ellipse_adaptive_width=True,
    )
    reps = triangular_group_reparameterisations(
        PARAMETERS, REFERENCE_TIME, log_mass_ratio=False,
        polarisation_ellipse=ellipse, polarisation_ellipse_adaptive_width=True,
    )
    return cls, reps


def _new_process(ellipse, configured):
    """The configuration's ellipse as a new process builds it: the group
    proposal class (and its action) hold this object, which in this process
    the first proposal's fits have moved."""
    ellipse.__dict__.clear()  # the fits' attributes shadow class defaults
    ellipse.__dict__.update(pickle.loads(configured).__dict__)


def _ellipse_rep(proposal):
    reps = [r for r in proposal._reparameterisation.values()
            if isinstance(r, PolarisationEllipseReparameterisation)]
    assert len(reps) == 1
    return reps[0]


def _log_q(proposal, x, net_of_domain_mass=False):
    """ln q of physical points: the flow at their prime coordinates plus the
    reparameterisations' Jacobian. ``net_of_domain_mass``: without the group
    mixture's domain normalisation, a running estimate (EMA over the draws)
    that plain nessai re-measures on its first draw after a resume; a
    constant in ln q, it cancels in nessai's rejection sampling."""
    x_prime, log_j = proposal.rescale(x.copy())
    a = live_points_to_array(x_prime, names=proposal.prime_parameters, copy=True)
    log_q = np.asarray(proposal.flow.log_prob(a)) + log_j
    if net_of_domain_mass:
        log_q = log_q + float(proposal.flow.model._log_domain_mass)
    return log_q


def _sheet_points(model, n, seed):
    from nessai.livepoint import numpy_array_to_live_points

    rng = np.random.default_rng(seed)
    theta = np.column_stack([rng.uniform(*BOUNDS[p], n) for p in PARAMETERS])
    live = numpy_array_to_live_points(theta, PARAMETERS)
    live["dec"] = np.arcsin(rng.uniform(-1, 1, n))
    centre = model._ellipse.cos_iota(live["ra"], live["dec"])
    cos_t = np.clip(LOCK * centre + rng.normal(0, 0.1, n), -0.999, 0.999)
    live["theta_jn"] = np.arccos(cos_t)
    live["logL"] = model.log_likelihood(live)
    live["logP"] = model.log_prior(live)
    return live


def test_flowproposal_resume_scores_identically(tmp_path):
    """nessai: after a checkpoint the group action reads the resumed
    reparameterisation's (fitted) ellipse, so the density is unchanged, and
    a refit after the resume still moves both together."""
    ellipse = _ellipse()
    configured = pickle.dumps(ellipse)
    cls, reps = _proposal_class(ellipse)
    model = SheetModel(ellipse)
    model.set_rng(np.random.default_rng(0))
    proposal = cls(
        model, output=str(tmp_path), poolsize=200, reparameterisations=reps,
        flow_config={"n_blocks": 2, "n_neurons": 8, "n_layers": 1},
        training_config={"max_epochs": 3, "patience": 3},
    )
    proposal.initialise()
    live = _sheet_points(model, 2000, seed=1)
    proposal.train(live)
    rep = _ellipse_rep(proposal)
    action = proposal.flow.model.prime_space_action
    assert action._ellipse is rep.ellipse
    # the fit moved away from the configuration's ellipse
    fitted = rep.ellipse.lock_slope
    assert abs(fitted - 1.0) > 0.1

    x = _sheet_points(model, 1000, seed=2)
    before = _log_q(proposal, x, net_of_domain_mass=True)
    assert np.isfinite(before).mean() > 0.9

    state = pickle.dumps(proposal)
    _new_process(ellipse, configured)
    resumed = pickle.loads(state)
    resumed.resume(model, dict(proposal.flow_config), proposal.flow.weights_file)
    np.testing.assert_allclose(
        _log_q(resumed, x, net_of_domain_mass=True), before, rtol=0, atol=1e-12)
    rep_r = _ellipse_rep(resumed)
    action_r = resumed.flow.model.prime_space_action
    assert rep_r.ellipse is not ellipse  # unpickled
    assert action_r._ellipse is rep_r.ellipse
    assert rep_r.ellipse.lock_slope == fitted

    # a refit after the resume: the action follows the reparameterisation
    resumed.train(_sheet_points(model, 2000, seed=3))
    assert resumed.flow.model.prime_space_action._ellipse is _ellipse_rep(
        resumed).ellipse


class _Interrupt(Exception):
    pass


@pytest.mark.slow_integration_test
def test_inessai_levels_resume_scores_identically(tmp_path, monkeypatch):
    """i-nessai with the group proposal as its levels, each level with its own
    ellipse fit: the run's ln q of every sample, and the levels' densities at
    any point, are the same after a checkpoint and resume; and the
    uninterrupted run scores its own samples as it stored them."""
    from nessai.flowsampler import FlowSampler
    from nessai.samplers.importancesampler import ImportanceNestedSampler

    ellipse = _ellipse()
    configured = pickle.dumps(ellipse)
    cls, reps = _proposal_class(ellipse)
    model = SheetModel(ellipse, sigma=0.05)
    kwargs = dict(
        output=str(tmp_path),
        importance_nested_sampler=True,
        nlive=1000,
        min_samples=200,
        seed=1234,
        plot=False,
        checkpointing=True,
        checkpoint_on_iteration=True,
        checkpoint_interval=1,
        proposal_class="levels",
        proposal_kwargs=dict(
            flow_proposal_class=cls,
            flow_config={"n_blocks": 2, "n_neurons": 8, "n_layers": 1},
            training_config={"max_epochs": 5, "patience": 5},
            flow_proposal_kwargs=dict(reparameterisations=reps, poolsize=1000),
            n_pilot=2000,
        ),
    )
    n_levels = 3
    add = ImportanceNestedSampler.add_new_proposal
    live = {}

    def interrupted(self):
        if self.iteration >= n_levels:
            live["log_q"] = self.training_samples.log_q.copy()
            live["samples"] = self.training_samples.samples.copy()
            live["rescored"] = self.proposal._log_q_all(live["samples"])
            raise _Interrupt
        add(self)

    fs = FlowSampler(model, **kwargs)
    monkeypatch.setattr(ImportanceNestedSampler, "add_new_proposal", interrupted)
    with pytest.raises(_Interrupt):
        fs.run(plot=False, save=False)
    monkeypatch.undo()
    levels = fs.ns.proposal.levels
    assert len(levels) == n_levels
    slopes = [_ellipse_rep(p).ellipse.lock_slope for p in levels]
    # each level fitted its own ellipse, away from the configuration's
    assert len({id(_ellipse_rep(p).ellipse) for p in levels}) == n_levels
    assert all(abs(s - 1.0) > 0.1 for s in slopes), slopes

    # the uninterrupted run: its stored ln q is what its levels give now
    np.testing.assert_allclose(live["rescored"], live["log_q"], rtol=0, atol=1e-9)

    _new_process(ellipse, configured)
    resumed = FlowSampler(model, resume=True, **kwargs)
    ns = resumed.ns
    assert ns.iteration == n_levels
    assert len(ns.proposal.levels) == n_levels
    np.testing.assert_array_equal(ns.training_samples.samples, live["samples"])
    np.testing.assert_allclose(ns.training_samples.log_q, live["log_q"],
                               rtol=0, atol=1e-9)
    for p in ns.proposal.levels:
        assert p.flow.model.prime_space_action._ellipse is _ellipse_rep(p).ellipse
    assert [_ellipse_rep(p).ellipse.lock_slope for p in ns.proposal.levels] == slopes
    # and new draws: the resumed levels score them as the live ones do
    pts = ns.proposal.draw(500)[0]
    np.testing.assert_allclose(ns.proposal._log_q_all(pts),
                               fs.ns.proposal._log_q_all(pts), rtol=0, atol=1e-9)
