"""Tests for the mixture of two flow proposals
(nessai_gw.mixture_proposal), on a toy model with plain nessai proposals."""

import pickle

import numpy as np
import pytest
from nessai.livepoint import numpy_array_to_live_points
from nessai.model import Model
from nessai.proposal import FlowProposal
from scipy import stats

from nessai_gw.mixture_proposal import _component_log_q, mixture_proposal_class

FLOW = {"n_blocks": 2, "n_neurons": 16, "n_layers": 1}
TRAINING = {"max_epochs": 30, "patience": 30, "batch_size": 256}


class Box(Model):
    """Uniform prior on [-1, 1]^2, a flat likelihood."""

    def __init__(self):
        self.names = ["x", "y"]
        self.bounds = {"x": [-1.0, 1.0], "y": [-1.0, 1.0]}

    def log_prior(self, x):
        return np.log(self.in_bounds(x), dtype="float") - np.log(4.0)

    def log_likelihood(self, x):
        return np.zeros(x.size)


def live_points(model, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = numpy_array_to_live_points(rng.uniform(-1, 1, (n, 2)), model.names)
    x["logL"] = model.log_likelihood(x)
    x["logP"] = model.log_prior(x)
    return x


@pytest.fixture()
def mixture(tmp_path):
    model = Box()
    model.set_rng(np.random.default_rng(0))
    cls = mixture_proposal_class(
        FlowProposal, FlowProposal,
        dict(flow_config=dict(FLOW, n_blocks=3), training_config=TRAINING),
        mixture_weight=0.3, class_name="ToyMixtureFlowProposal",
    )
    prop = cls(model, flow_config=FLOW, training_config=TRAINING, output=str(tmp_path),
               poolsize=2000, plot=False)
    prop.rng = np.random.default_rng(1)
    prop.initialise()
    x = live_points(model)
    prop.current_live_points = x
    prop.train(x, plot=False)
    return prop, model, x


def test_component_densities_match_their_draws(mixture):
    prop, _, _ = mixture
    for k, comp in ((0, prop), (1, prop.secondary)):
        x, log_q = prop._draw_component(k, 500)
        assert len(x) > 100
        np.testing.assert_allclose(_component_log_q(comp, x), log_q, atol=1e-4)
    # one point (some reparameterisations' table lookups need two)
    x, log_q = prop._draw_component(0, 50)
    np.testing.assert_allclose(_component_log_q(prop, x[:1]), log_q[:1], atol=1e-4)


def test_mixture_log_q_and_pool_follow_the_prior(mixture):
    prop, model, x = mixture
    log_q, lq = prop.mixture_log_q(x[:200])
    w = 0.3
    np.testing.assert_allclose(
        log_q, np.log((1 - w) * np.exp(lq[0]) + w * np.exp(lq[1])), atol=1e-10)
    prop.populate(x[np.argmin(x["logL"])], n_samples=2000, plot=False)
    s = prop.samples
    assert s.size == 2000 and np.all(np.isfinite(s["logL"]))
    assert prop.populated and len(prop.indices) == 2000
    entry = prop.mixture_population_history[-1]
    assert sum(entry["drawn"]) > 0 and entry["drawn"][1] > 0
    # the pool follows min(prior, c q): with flows fitted to a uniform live
    # set, the prior on the box up to the clipped share
    assert prop.population_clipped_share < 0.05
    for n in model.names:
        assert stats.kstest((s[n] + 1) / 2, "uniform").pvalue > 1e-3, n
    new = prop.draw(None)
    assert new.dtype.names == s.dtype.names


def test_fitted_weight(tmp_path):
    model = Box()
    model.set_rng(np.random.default_rng(0))
    cls = mixture_proposal_class(FlowProposal, FlowProposal,
                                 dict(flow_config=FLOW, training_config=TRAINING),
                                 mixture_weight=None, min_weight=0.1,
                                 class_name="ToyFittedMixtureFlowProposal")
    prop = cls(model, flow_config=FLOW, training_config=TRAINING, output=str(tmp_path),
               poolsize=100, plot=False)
    prop.rng = np.random.default_rng(2)
    prop.initialise()
    x = live_points(model)
    prop.current_live_points = x
    prop.train(x, plot=False)
    w = prop.mixture_history[-1]["weight"]
    assert 0.1 <= w <= 0.9
    np.testing.assert_allclose(np.exp(prop.mixture_log_weights).sum(), 1.0)


def test_pickle_resume(mixture, tmp_path):
    prop, model, x = mixture
    log_q, _ = prop.mixture_log_q(x[:100])
    weights = str(tmp_path / "primary.pt")
    prop.flow.save_weights(weights)
    copy = pickle.loads(pickle.dumps(prop))
    assert type(copy).__name__ == "ToyMixtureFlowProposal"
    copy.resume(model, dict(FLOW), weights)
    log_q_copy, _ = copy.mixture_log_q(x[:100])
    np.testing.assert_allclose(log_q, log_q_copy, atol=1e-6)


def test_weight_checked_in_class():
    cls = mixture_proposal_class(FlowProposal, FlowProposal, mixture_weight=0.25,
                                 class_name="ToyCheckMixtureFlowProposal")
    assert cls.mixture_weight == 0.25 and cls.secondary_class is FlowProposal
    import nessai_gw.mixture_proposal as module
    assert getattr(module, "ToyCheckMixtureFlowProposal") is cls


def test_stage_switch(tmp_path):
    """With switch_lnx: the run's proposal's own pools until ln X reaches
    it, the secondary's after; after the switch only the secondary trains."""
    model = Box()
    model.set_rng(np.random.default_rng(0))
    cls = mixture_proposal_class(FlowProposal, FlowProposal,
                                 dict(flow_config=FLOW, training_config=TRAINING),
                                 switch_lnx=-2.0, switch_warmup=1.0,
                                 class_name="ToySwitchFlowProposal")
    prop = cls(model, flow_config=FLOW, training_config=TRAINING, output=str(tmp_path),
               poolsize=300, plot=False)
    prop.rng = np.random.default_rng(3)
    prop.initialise()
    x = live_points(model, n=1000)
    x["it"] = np.arange(1000)  # ln X ~ -1: before the switch, outside the warm-up
    prop.current_live_points = x
    prop.train(x, plot=False)
    assert prop.secondary.training_count == 0
    prop.populate(x[0], n_samples=300, plot=False)
    assert prop.switched_at is None and prop.samples.size == 300
    x["it"] = np.arange(1000) + 1600  # ln X -2.6: switched
    prop.current_live_points = x
    count = prop.training_count
    prop.populate(x[0], n_samples=300, plot=False)
    assert prop.switched_at == pytest.approx(-2.599)
    assert prop.secondary.training_count == 1  # trained at the switch
    assert prop.samples.size == 300 and len(prop.indices) == 300 and prop.populated
    flow_state = {k: v.clone() for k, v in prop.flow.model.state_dict().items()}
    prop.train(x, plot=False)
    assert prop.training_count == count + 1 and prop.secondary.training_count == 2
    for k, v in prop.flow.model.state_dict().items():
        assert (v == flow_state[k]).all()
    copy = pickle.loads(pickle.dumps(prop))
    assert copy.switched_at == prop.switched_at


def test_stage_switch_on_spread(tmp_path):
    model = Box()
    model.set_rng(np.random.default_rng(0))
    cls = mixture_proposal_class(FlowProposal, FlowProposal,
                                 dict(flow_config=FLOW, training_config=TRAINING),
                                 switch_spread=10.0, class_name="ToySpreadSwitchFlowProposal")
    prop = cls(model, flow_config=FLOW, training_config=TRAINING, output=str(tmp_path),
               poolsize=200, plot=False)
    prop.rng = np.random.default_rng(4)
    prop.initialise()
    x = live_points(model, n=1000)
    x["it"] = np.arange(1000)
    x["logL"] = np.linspace(-30, 0, 1000)  # spread 30: neither switch nor warm-up
    prop.current_live_points = x
    prop.train(x, plot=False)
    assert prop.secondary.training_count == 0
    prop.populate(x[0], n_samples=200, plot=False)
    assert prop.switched_at is None
    x["logL"] = np.linspace(-15, 0, 1000)  # spread 15: warm-up
    prop.train(x, plot=False)
    assert prop.secondary.training_count == 1 and prop.switched_at is None
    x["logL"] = np.linspace(-8, 0, 1000)  # spread 8: switched
    prop.populate(x[0], n_samples=200, plot=False)
    assert prop.switched_at is not None and prop.samples.size == 200


def test_stage_switch_race(tmp_path):
    """With switch_race: from race_start_lnx every race_every-th pool is the
    secondary's, each pool's cost per new point recorded; the switch follows
    the recorded costs."""
    model = Box()
    model.set_rng(np.random.default_rng(0))
    cls = mixture_proposal_class(FlowProposal, FlowProposal,
                                 dict(flow_config=FLOW, training_config=TRAINING),
                                 switch_race=True, race_start_lnx=-0.5, race_every=2,
                                 race_wins=2, class_name="ToyRaceFlowProposal")
    prop = cls(model, flow_config=FLOW, training_config=TRAINING, output=str(tmp_path),
               poolsize=200, plot=False)
    prop.rng = np.random.default_rng(5)
    prop.initialise()
    x = live_points(model, n=1000)
    x["it"] = np.arange(1000) - 800  # ln X -0.2: before the race
    x["logL"] = np.linspace(-1, 0, 1000)
    prop.current_live_points = x
    prop.train(x, plot=False)
    assert prop.secondary.training_count == 0
    prop.populate(x[0], n_samples=200, plot=False)
    assert len(prop.race_history) == 0 and prop.switched_at is None
    x["it"] = np.arange(1000)  # ln X -1: racing
    prop.train(x, plot=False)
    assert prop.secondary.training_count == 1
    for _ in range(8):
        prop.populate(x[0], n_samples=200, plot=False)
        assert prop.samples.size == 200 and prop.populated
        if prop.switched_at is not None:
            break
    h = prop.race_history
    # a secondary pool after each run's pool, unless one lost by over 4x
    for i, e in enumerate(h):
        if e["component"] == 1:
            assert i > 0 and h[i - 1]["component"] == 0
    wins, switched = 0, None
    for i, e in enumerate(h):
        if e["component"] == 1:
            wins = wins + 1 if e["cost"] < h[i - 1]["cost"] else 0
            if wins >= 2:
                switched = i
                break
    assert (switched is None) == (prop.switched_at is None)
    if switched is not None:
        assert switched == len(h) - 1
        n = len(h)
        prop.populate(x[0], n_samples=200, plot=False)
        assert len(prop.race_history) == n  # no more racing


def test_race_backoff():
    cls = mixture_proposal_class(FlowProposal, FlowProposal, switch_race=True, race_every=4,
                                 class_name="ToyBackoffFlowProposal")
    prop = object.__new__(cls)
    run = lambda c: dict(component=0, cost=c)
    sec = lambda c: dict(component=1, cost=c)
    assert prop._race_gap([]) == 3
    assert prop._race_gap([run(1.0)] * 3 + [sec(10.0)]) == 7
    assert prop._race_gap([run(1.0)] * 3 + [sec(10.0)] + [run(1.0)] * 7 + [sec(10.0)]) == 15
    assert prop._race_gap([run(1.0)] * 3 + [sec(10.0)] + [run(1.0)] * 7 + [sec(2.0)]) == 3
    many = []
    for _ in range(8):
        many += [run(1.0)] * 3 + [sec(100.0)]
    assert prop._race_gap(many) == 63
