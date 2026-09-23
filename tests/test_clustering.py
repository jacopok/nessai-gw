"""Tests for the live-fitted diagonal split in nessai_gw.clustering."""

import numpy as np
import pytest
import torch

from nessai_gw.clustering import (
    MIN_BRANCH_FRACTION,
    DiagonalSplitClusterWrapper,
    fit_live_diagonal_split,
    gaussian_split_gain,
    make_diagonal_split_cluster_flow,
    split_coordinates,
)


def _two_populations(n_plateau=3000, n_clump=3000, n_extra=6, seed=0):
    """Plateau: low sky_v, tight inclination; clump: sky_v ~ 0.4, broad
    inclination -- the heteroscedastic pair seen on ET-Delta.  ``t`` carries
    (c, v) plus ``n_extra`` coordinates whose scale also differs by branch."""
    rng = np.random.default_rng(seed)
    c = np.concatenate(
        [rng.normal(0.17, 0.05, n_plateau), rng.uniform(0.35, 1.0, n_clump)]
    )
    v = np.concatenate(
        [rng.uniform(0.0, 0.3, n_plateau), rng.normal(0.42, 0.02, n_clump)]
    )
    extra = np.concatenate(
        [
            rng.normal(0.0, 1.0, (n_plateau, n_extra)),
            rng.normal(0.0, 0.1, (n_clump, n_extra)),
        ]
    )
    truth = np.concatenate([np.zeros(n_plateau, int), np.ones(n_clump, int)])
    return c, v, np.column_stack([c, v, extra]), truth


def test_split_coordinates_theta_convention():
    theta = np.array([0.0, 0.3491, np.pi / 2, 2.5])
    canon = np.column_stack([2 * theta / np.pi - 1, [0.1, 0.2, 0.3, 0.4]])
    c, v = split_coordinates(canon, ["theta_jn_prime", "sky_v"])
    np.testing.assert_allclose(c, np.cos(theta), atol=1e-12)
    np.testing.assert_allclose(v, [0.1, 0.2, 0.3, 0.4])


def test_gain_of_meaningless_split_is_minus_label_entropy():
    rng = np.random.default_rng(1)
    t = rng.normal(size=(20000, 5))
    labels = rng.integers(0, 2, len(t))
    assert gaussian_split_gain(t, labels) == pytest.approx(
        -np.log(2), abs=0.01
    )


def test_gain_minus_inf_when_a_side_is_too_small():
    t = np.random.default_rng(2).normal(size=(100, 3))
    labels = np.zeros(100, int)
    labels[:2] = 1
    assert gaussian_split_gain(t, labels) == -np.inf


def test_fit_recovers_heteroscedastic_boundary():
    c, v, t, truth = _two_populations()
    fit = fit_live_diagonal_split(c, v, t)
    assert fit is not None
    lab = (c * fit["normal"][0] + v * fit["normal"][1] > fit["offset"]).astype(
        int
    )
    assert np.mean(lab == truth) > 0.97
    assert fit["fraction"] == pytest.approx(lab.mean())
    assert fit["gain"] > 1.0
    # the scan's cumulative-moment gain equals the direct evaluation
    assert fit["gain"] == pytest.approx(gaussian_split_gain(t, lab), abs=1e-6)


def test_fit_on_single_gaussian_gives_no_useful_gain():
    rng = np.random.default_rng(3)
    t = rng.normal(size=(6000, 6))
    fit = fit_live_diagonal_split(t[:, 0], t[:, 1], t)
    assert fit["gain"] < 0.05


def test_fit_respects_min_fraction():
    c, v, t, _ = _two_populations(n_plateau=5800, n_clump=200)
    fit = fit_live_diagonal_split(c, v, t, min_fraction=0.1)
    assert 0.1 - 1e-3 <= fit["fraction"] <= 0.9 + 1e-3
    assert fit_live_diagonal_split(c[:150], v[:150], t[:150]) is None


class _FakeWrapper(DiagonalSplitClusterWrapper):
    """Only the state ``_k_want`` reads; ``_split_coords`` reads (c, v) off
    the first two columns of ``t``."""

    def __init__(self, active=False):
        torch.nn.Module.__init__(self)
        self.n_experts = 2
        self._n_active = torch.tensor(2 if active else 1)
        self._clustering_seen = torch.tensor(True)
        self._split_normal = torch.zeros(2, dtype=torch.float64)
        self._split_offset = torch.zeros((), dtype=torch.float64)
        self._split_fitted = torch.zeros((), dtype=torch.bool)
        self._split_gain = torch.full((), -np.inf)

    def _split_coords(self, t):
        return t[:, 0], t[:, 1]


def test_k_want_activates_on_heteroscedastic_split():
    _, _, t, truth = _two_populations()
    w = _FakeWrapper()
    assert w._k_want(t, None, None, None) == 2
    assert bool(w._split_fitted)
    assert np.mean(w._split_labels(t[:, 0], t[:, 1]) == truth) > 0.97


def test_k_want_stays_one_on_single_gaussian():
    t = np.random.default_rng(4).normal(size=(6000, 6))
    assert _FakeWrapper()._k_want(t, None, None, None) == 1


def test_k_want_respects_activate_min_round():
    _, _, t, _ = _two_populations()
    w = _FakeWrapper()
    w.activate_min_round = 3
    assert [w._k_want(t, None, None, None) for _ in range(3)] == [1, 1, 2]


def test_active_split_collapses_when_gain_drops():
    _, _, t, _ = _two_populations()
    w = _FakeWrapper(active=True)
    assert w._k_want(t, None, None, None) == 2
    t_single = np.random.default_rng(5).normal(size=(6000, t.shape[1]))
    assert w._k_want(t_single, None, None, None) == 1


def test_active_line_kept_unless_new_fit_is_better_by_margin():
    _, _, t, _ = _two_populations()
    w = _FakeWrapper(active=True)
    w._k_want(t, None, None, None)
    stored = w._split_normal.clone(), float(w._split_offset)
    w.refit_margin = np.inf
    _, _, t2, _ = _two_populations(seed=7)
    w._k_want(t2, None, None, None)
    assert torch.equal(w._split_normal, stored[0])
    assert float(w._split_offset) == stored[1]


class _StubExpert(torch.nn.Module):
    group_size = 8

    def _carry_over_group_state(self, old):
        pass


def test_split_buffers_are_state_and_survive_reset_carry_over():
    cls = type("W", (DiagonalSplitClusterWrapper,), {})
    old = cls([_StubExpert(), _StubExpert()], 4)
    for key in (
        "_split_normal",
        "_split_offset",
        "_split_fitted",
        "_split_gain",
    ):
        assert key in old.state_dict()
    old._store_split(dict(normal=np.array([1.5, -2.0]), offset=0.25))
    old._split_gain.fill_(1.2)
    new = cls([_StubExpert(), _StubExpert()], 4)
    new._carry_over_group_state(old)
    assert torch.equal(new._split_normal, old._split_normal)
    assert float(new._split_offset) == 0.25
    assert bool(new._split_fitted)
    assert float(new._split_gain) == pytest.approx(1.2)


def test_make_diagonal_split_cluster_flow_sets_attributes():
    from nessai.flowmodel.group_mixture import ClusteredGroupMixtureFlowModel

    from nessai_gw.group_mixture import ETTriangleGroupAction

    action = ETTriangleGroupAction(
        reference_time=1187008882.4, polarisation_quarter=True
    )
    kw = dict(
        group_action_fn=action,
        group_size=action.group_size,
        param_names=[
            "sky_u",
            "sky_v",
            "psi_prime",
            "delta_phase",
            "theta_jn_prime",
            "t_det",
        ],
        in_fundamental_domain=action.in_fundamental_domain,
    )
    cls = make_diagonal_split_cluster_flow(**kw)
    assert issubclass(cls, ClusteredGroupMixtureFlowModel)
    assert cls.n_clusters_max == 2
    assert cls.wrapper_cls.gain_on == DiagonalSplitClusterWrapper.gain_on
    assert cls.wrapper_cls.activate_fraction == MIN_BRANCH_FRACTION

    cls2 = make_diagonal_split_cluster_flow(
        gain_on=0.7,
        gain_off=0.1,
        activate_fraction=0.08,
        activate_min_round=5,
        **kw,
    )
    w = cls2.wrapper_cls
    assert (
        w.gain_on,
        w.gain_off,
        w.activate_fraction,
        w.activate_min_round,
    ) == (0.7, 0.1, 0.08, 5)
