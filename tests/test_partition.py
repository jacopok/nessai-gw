"""Split coordinates and splitters for nessai's partitioned nested sampling."""

import pickle

import numpy as np
import pytest
from nessai.livepoint import empty_structured_array

from nessai_gw.partition import LineSplitter, SkyCoordinates


class _Sky:
    """A stand-in for EqualAreaSky: what SkyCoordinates reads."""

    prime_parameters = ["sky_u", "sky_v"]

    def __init__(self, R):
        self._R = R


class _Proposal:
    def __init__(self, R):
        self._reparameterisation = {"sky": _Sky(R)}


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


@pytest.fixture
def points():
    rng = np.random.default_rng(3)
    p = empty_structured_array(2000, names=["ra", "dec", "theta_jn"])
    p["ra"] = rng.uniform(0, 2 * np.pi, 2000)
    p["dec"] = np.arcsin(rng.uniform(-1, 1, 2000))
    p["theta_jn"] = np.arccos(rng.uniform(-1, 1, 2000))
    return p


def test_sky_coordinates_match_equal_area_sky(points):
    """sky_u, sky_v as EqualAreaSky computes them."""
    from nessai_gw.reparameterisations.sky import EqualAreaSky

    rng = np.random.default_rng(1)
    R = _rotation(rng)
    rep = EqualAreaSky(
        parameters=["ra", "dec"],
        prior_bounds={"ra": [0, 2 * np.pi], "dec": [-np.pi / 2, np.pi / 2]},
        rotation=R,
    )
    x_prime = empty_structured_array(len(points), names=["sky_u", "sky_v"])
    _, x_prime, _ = rep.reparameterise(points, x_prime, np.zeros(len(points)))
    coords = SkyCoordinates(["sky_u", "sky_v"]).freeze(_Proposal(R))
    c = coords(points)
    np.testing.assert_allclose(c[:, 0], x_prime["sky_u"], atol=1e-12)
    np.testing.assert_allclose(c[:, 1], x_prime["sky_v"], atol=1e-12)
    assert coords.period_list() == [1.0, None]


def test_sky_coordinates_plane_fold(points):
    R = np.eye(3)
    plain = SkyCoordinates(["cos_theta_jn", "sky_v"]).freeze(_Proposal(R))
    folded = SkyCoordinates(["cos_theta_jn", "sky_v"], fold="plane").freeze(
        _Proposal(R)
    )
    a, b = plain(points), folded(points)
    assert np.all(b[:, 1] <= 0.5)
    refl = a[:, 1] >= 0.5
    np.testing.assert_allclose(b[refl, 1], 1 - a[refl, 1])
    np.testing.assert_allclose(b[refl, 0], -a[refl, 0])
    np.testing.assert_allclose(b[~refl], a[~refl])


def test_sky_coordinates_frozen_frame(points):
    """A frozen copy keeps the rotation it was frozen with."""
    rng = np.random.default_rng(2)
    proposal = _Proposal(_rotation(rng))
    frozen = SkyCoordinates(["sky_u"]).freeze(proposal)
    before = frozen(points)
    proposal._reparameterisation["sky"]._R = _rotation(rng)
    np.testing.assert_array_equal(frozen(points), before)
    again = pickle.loads(pickle.dumps(frozen))
    np.testing.assert_array_equal(again(points), before)
    with pytest.raises(RuntimeError):
        SkyCoordinates(["sky_u"])(points)


def test_sky_coordinates_names():
    with pytest.raises(ValueError):
        SkyCoordinates(["sky_w"])
    with pytest.raises(ValueError):
        SkyCoordinates(["sky_u"], fold="other")


def test_line_splitter_heteroscedastic():
    """Two branches of very different shapes along a line: split; one
    Gaussian: no split."""
    rng = np.random.default_rng(0)
    n = 1200
    plateau = np.c_[rng.normal(0.2, 0.02, n), rng.uniform(0.0, 0.3, n)]
    clump = np.c_[rng.uniform(0.5, 1.0, n), rng.normal(0.4, 0.01, n)]
    coords = np.r_[plateau, clump]
    splitter = LineSplitter()
    cut = splitter.propose(coords, [None, None])
    assert cut is not None and cut.info["gain"] > splitter.gain_on
    labels = cut.classify(coords)
    assert labels[:n].mean() < 0.05 and labels[n:].mean() > 0.95
    blob = rng.normal(size=(2 * n, 2))
    assert splitter.propose(blob, [None, None]) is None
    assert splitter.propose(blob, [None, None], force=True) is not None


class _TriggerWrapper:
    """The state DiagonalSplitTrigger reads off a DiagonalSplitClusterWrapper;
    routes by its line on (x0, x1) unless ``routed`` is set."""

    def __init__(self, active, pending=False, routed=None):
        import torch

        self._clustering_seen = torch.tensor(True)
        self._n_active = torch.tensor(2 if active else 1)
        self._split_fitted = torch.tensor(True)
        self._pending_split_train = torch.tensor(pending)
        self._split_normal = torch.tensor([1.0, 2.0], dtype=torch.float64)
        self._split_offset = torch.tensor(0.5, dtype=torch.float64)
        self._split_gain = torch.tensor(1.3)
        self.routed = routed

    def _route(self, x):
        import torch

        if self.routed is not None:
            return torch.as_tensor(self.routed)
        return (x @ self._split_normal > self._split_offset).long()


class _TriggerProposal:
    def __init__(self, wrapper):
        self.flow = type("F", (), {"model": wrapper})()

    def _training_data_as_prime_tensor(self, points):
        import torch

        return torch.as_tensor(
            np.column_stack([points["x0"], points["x1"]]), dtype=torch.float64
        )


def _trigger(monkeypatch):
    import nessai_gw.clustering as clustering

    from nessai_gw.partition import DiagonalSplitTrigger

    # the stand-in passes for the wrapper class
    monkeypatch.setattr(
        clustering, "DiagonalSplitClusterWrapper", _TriggerWrapper
    )
    return DiagonalSplitTrigger()


def _plane_points(n=1000):
    rng = np.random.default_rng(5)
    p = empty_structured_array(n, names=["x0", "x1"])
    p["x0"] = rng.uniform(-1, 1, n)
    p["x1"] = rng.uniform(0, 0.5, n)
    return p, np.column_stack([p["x0"], p["x1"]])


def test_diagonal_split_trigger_takes_the_active_line(monkeypatch):
    trigger = _trigger(monkeypatch)
    points, coords = _plane_points()
    for state in (dict(active=False), dict(active=True, pending=True)):
        proposal = _TriggerProposal(_TriggerWrapper(**state))
        assert trigger.propose(coords, [None, None], points, proposal) is None
    proposal = _TriggerProposal(_TriggerWrapper(active=True))
    cut = trigger.propose(coords, [None, None], points, proposal)
    np.testing.assert_array_equal(cut.normal, [1.0, 2.0])
    assert cut.offset == 0.5
    assert cut.info["agreement"] == 1.0
    assert cut.info["gain"] == pytest.approx(1.3)
    np.testing.assert_array_equal(
        cut.classify(coords), coords @ [1.0, 2.0] > 0.5
    )
    assert (
        trigger.propose(coords, [None, None], points, proposal, max_labels=1)
        is None
    )


def test_diagonal_split_trigger_refuses_a_line_off_the_routing(monkeypatch):
    trigger = _trigger(monkeypatch)
    points, coords = _plane_points()
    routed = (coords[:, 0] > 0).astype(int)  # not the line
    proposal = _TriggerProposal(_TriggerWrapper(active=True, routed=routed))
    assert trigger.propose(coords, [None, None], points, proposal) is None


def test_diagonal_split_trigger_needs_the_diagonal_split(monkeypatch):
    trigger = _trigger(monkeypatch)
    points, coords = _plane_points()
    with pytest.raises(RuntimeError):
        trigger.propose(coords, [None, None], points, _TriggerProposal(None))
