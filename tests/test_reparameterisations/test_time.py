"""Tests for :class:`DetectorCenterTimeReparameterisation`."""

import numpy as np
import pytest
from nessai.livepoint import dict_to_live_points, empty_structured_array

from nessai_gw._geometry import geocenter_time_delay, greenwich_mean_sidereal_time
from nessai_gw.group_mixture import ET_EMR_VERTEX
from nessai_gw.reparameterisations import DetectorCenterTimeReparameterisation

REFERENCE_TIME = 1187008882.4
SCALE = 5e-3


class TestDetectorCenterTime:
    prior_bounds = {"geocent_time": [REFERENCE_TIME - 0.1, REFERENCE_TIME + 0.1]}

    def _reparam(self, scale=SCALE):
        return DetectorCenterTimeReparameterisation(
            parameters="geocent_time",
            prior_bounds=self.prior_bounds,
            vertex=ET_EMR_VERTEX,
            reference_time=REFERENCE_TIME,
            scale=scale,
        )

    def test_init(self):
        reparam = self._reparam()
        assert reparam.prime_parameters == ["t_det"]
        assert reparam.requires == ["ra", "dec"]
        assert reparam.scale == SCALE

    def test_requires_vertex_and_reference_time(self):
        with pytest.raises(ValueError, match="requires `vertex`"):
            DetectorCenterTimeReparameterisation(
                parameters="geocent_time", prior_bounds=self.prior_bounds
            )

    def test_must_act_on_geocent_time(self):
        with pytest.raises(RuntimeError, match="must act on"):
            DetectorCenterTimeReparameterisation(
                parameters="ra",
                prior_bounds={"ra": [0.0, 2 * np.pi]},
                vertex=ET_EMR_VERTEX,
                reference_time=REFERENCE_TIME,
            )

    def _points(self, n, rng):
        ra = rng.uniform(0, 2 * np.pi, n)
        dec = np.arcsin(rng.uniform(-1, 1, n))
        gt = REFERENCE_TIME + rng.uniform(-0.05, 0.05, n)
        return ra, dec, gt

    def test_reparameterise_values(self):
        reparam = self._reparam()
        rng = np.random.default_rng(1)
        n = 25
        ra, dec, gt = self._points(n, rng)
        x = dict_to_live_points(
            {"geocent_time": gt, "ra": ra, "dec": dec}
        )
        x_prime = empty_structured_array(n, names=["t_det", "ra", "dec"])
        x_prime["ra"] = ra
        x_prime["dec"] = dec
        log_j = np.zeros(n)
        _, x_prime, log_j = reparam.reparameterise(x, x_prime, log_j)

        gmst = greenwich_mean_sidereal_time(REFERENCE_TIME)
        delay = geocenter_time_delay(ET_EMR_VERTEX, gmst, ra, dec)
        # epoch subtracted first: gt + delay would round onto the ~0.24 us
        # float64 grid of GPS times
        want = ((gt - REFERENCE_TIME) + delay) / SCALE
        np.testing.assert_allclose(x_prime["t_det"], want, atol=1e-9)
        np.testing.assert_allclose(log_j, -np.log(SCALE), atol=1e-12)

    @pytest.mark.integration_test
    def test_invertible(self):
        reparam = self._reparam()
        rng = np.random.default_rng(2)
        n = 50
        ra, dec, gt = self._points(n, rng)
        x = dict_to_live_points({"geocent_time": gt, "ra": ra, "dec": dec})
        names = ["t_det", "ra", "dec"]
        x_prime = empty_structured_array(n, names=names)
        x_prime["ra"] = ra
        x_prime["dec"] = dec
        log_j = np.zeros(n)

        x_f, x_prime_f, log_j_f = reparam.reparameterise(
            x.copy(), x_prime.copy(), log_j.copy()
        )
        np.testing.assert_array_equal(x_f["geocent_time"], gt)

        x_in = x_f.copy()
        x_in["geocent_time"] = np.nan
        x_i, _, log_j_i = reparam.inverse_reparameterise(
            x_in, x_prime_f.copy(), log_j_f.copy()
        )
        np.testing.assert_allclose(x_i["geocent_time"], gt, rtol=1e-12)
        np.testing.assert_allclose(log_j_i, 0.0, atol=1e-10)

    def test_reflection_shifts_geocent_time_by_delay_change(self):
        """When a group reflection moves (ra, dec), holding t_det fixed
        reconstructs geocent_time shifted by delay(n) - delay(n')."""
        reparam = self._reparam()
        rng = np.random.default_rng(3)
        n = 30
        ra, dec, gt = self._points(n, rng)
        gmst = greenwich_mean_sidereal_time(REFERENCE_TIME)

        # forward to t_det
        x = dict_to_live_points({"geocent_time": gt, "ra": ra, "dec": dec})
        x_prime = empty_structured_array(n, names=["t_det", "ra", "dec"])
        x_prime["ra"] = ra
        x_prime["dec"] = dec
        _, x_prime, _ = reparam.reparameterise(x, x_prime, np.zeros(n))

        # a "reflection" of the sky direction (arbitrary here): dec -> -dec
        ra2, dec2 = ra, -dec
        x_in = empty_structured_array(
            n, names=["geocent_time", "ra", "dec"]
        )
        x_in["ra"] = ra2
        x_in["dec"] = dec2
        x_in["geocent_time"] = np.nan
        x_i, _, _ = reparam.inverse_reparameterise(
            x_in, x_prime.copy(), np.zeros(n)
        )

        d0 = geocenter_time_delay(ET_EMR_VERTEX, gmst, ra, dec)
        d1 = geocenter_time_delay(ET_EMR_VERTEX, gmst, ra2, dec2)
        np.testing.assert_allclose(
            x_i["geocent_time"], gt + d0 - d1, rtol=1e-12
        )


# --- time at a reference frequency -------------------------------------------

from nessai_gw.reparameterisations.mass import doppler_factor  # noqa: E402
from nessai_gw.reparameterisations.time import (  # noqa: E402
    _SOLAR_MASS_TIME,
    pn_time_to_merger,
)

#: roughly the Earth's orbital velocity over c
DOPPLER = np.array([5.6e-5, 7.4e-5, 3.2e-5])
INTRINSIC = ("chirp_mass", "mass_ratio", "chi_1", "chi_2")


def _full_points(n, seed, t_spread=0.02):
    rng = np.random.default_rng(seed)
    x = dict_to_live_points({
        "chirp_mass": rng.uniform(1.1835, 1.1840, n),
        "mass_ratio": rng.uniform(0.6, 1.0, n),
        "chi_1": rng.uniform(-0.05, 0.05, n),
        "chi_2": rng.uniform(-0.05, 0.05, n),
        "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": np.arcsin(rng.uniform(-1, 1, n)),
        "geocent_time": REFERENCE_TIME + rng.uniform(-t_spread, t_spread, n),
    })
    return x


def _ref_reparam(reference_frequency=20.0, doppler_vector=None, **kwargs):
    return DetectorCenterTimeReparameterisation(
        parameters="geocent_time",
        prior_bounds={"geocent_time": [REFERENCE_TIME - 0.1, REFERENCE_TIME + 0.1]},
        vertex=ET_EMR_VERTEX,
        reference_time=REFERENCE_TIME,
        reference_frequency=reference_frequency,
        doppler_vector=doppler_vector,
        **kwargs,
    )


def _forward(reparam, x):
    x_prime = empty_structured_array(len(x), names=["t_det"])
    return reparam.reparameterise(x.copy(), x_prime, np.zeros(len(x)))


def test_pn_time_to_merger_newtonian_limit():
    """At low frequency the 2PN time tends to the Newtonian chirp time."""
    mc, f = 1.1838, 4.0
    newtonian = 5 / 256 * (np.pi * f) ** (-8 / 3) * (mc * _SOLAR_MASS_TIME) ** (-5 / 3)
    tau = pn_time_to_merger(mc, 0.9, 0.0, 0.0, f)
    assert tau == pytest.approx(newtonian, rel=0.01)
    assert tau > newtonian  # the 1PN correction lengthens the inspiral
    # roughly 2.8 min from 20 Hz for a GW170817-like binary
    assert pn_time_to_merger(mc, 0.9, 0.0, 0.0, 20.0) == pytest.approx(169, rel=0.02)


def test_reference_frequency_init():
    reparam = _ref_reparam(doppler_vector=DOPPLER)
    assert reparam.prime_parameters == ["t_det"]
    assert reparam.requires == ["ra", "dec", *INTRINSIC]
    if hasattr(reparam, "inverse_input_parameters"):  # nessai >= 0.16
        assert set(INTRINSIC) <= set(reparam.inverse_input_parameters)
    assert reparam.reference_frequency == 20.0
    assert not reparam.adaptive
    no_spin = _ref_reparam(spins=None)
    assert no_spin.requires == ["ra", "dec", "chirp_mass", "mass_ratio"]
    adaptive = _ref_reparam("adaptive", frequency_range=(10.0, 40.0))
    assert adaptive.adaptive
    assert adaptive.reference_frequency == pytest.approx(20.0)


@pytest.mark.parametrize("kwargs", [
    dict(reference_frequency=-1.0),
    dict(frequency_range=(30.0, 10.0)),
    dict(doppler_vector=[1e-4, 0.0]),
    dict(spins=("chi_1",)),
])
def test_reference_frequency_init_errors(kwargs):
    kwargs = {"reference_frequency": 20.0, **kwargs}
    with pytest.raises(ValueError):
        _ref_reparam(**kwargs)


@pytest.mark.parametrize("doppler_vector", [None, DOPPLER])
def test_reference_frequency_round_trip(doppler_vector):
    reparam = _ref_reparam(doppler_vector=doppler_vector)
    x = _full_points(200, 4)
    _, x_prime, log_j = _forward(reparam, x)
    np.testing.assert_allclose(log_j, -np.log(SCALE))

    x_in = x.copy()
    x_in["geocent_time"] = np.nan
    x_out, _, log_j_out = reparam.inverse_reparameterise(
        x_in, x_prime.copy(), log_j.copy()
    )
    np.testing.assert_allclose(
        x_out["geocent_time"] - REFERENCE_TIME,
        x["geocent_time"] - REFERENCE_TIME, atol=1e-12,
    )
    np.testing.assert_allclose(log_j_out, 0.0, atol=1e-10)


def test_reference_frequency_is_merger_time_minus_tau():
    plain = _ref_reparam(reference_frequency=None)
    shifted = _ref_reparam(doppler_vector=DOPPLER)
    x = _full_points(50, 5)
    _, merger, _ = _forward(plain, x)
    _, at_f, _ = _forward(shifted, x)
    mc_earth = x["chirp_mass"] * doppler_factor(x["ra"], x["dec"], DOPPLER)
    tau = pn_time_to_merger(mc_earth, x["mass_ratio"], x["chi_1"], x["chi_2"], 20.0)
    np.testing.assert_allclose(
        (merger["t_det"] - at_f["t_det"]) * SCALE,
        tau - shifted._tau_ref, atol=1e-9,
    )
    # tau_ref is set once, from the first batch, and keeps the coordinate
    # near zero
    assert shifted._tau_ref == pytest.approx(np.median(tau))
    assert abs(np.median(at_f["t_det"])) < 20


def test_coordinate_depends_on_the_earth_frame_chirp_mass_only():
    """With the Doppler vector, two points with the same Earth-frame chirp
    mass (and t_det, q, spins) have the same coordinate whatever the sky:
    the coordinate stays invariant under the group, like t_det."""
    reparam = _ref_reparam(doppler_vector=DOPPLER)
    plain = _ref_reparam(reference_frequency=None)
    x = _full_points(40, 6)
    y = x.copy()
    y["ra"] = np.mod(x["ra"] + 2.0, 2 * np.pi)
    y["dec"] = -x["dec"]
    y["chirp_mass"] = x["chirp_mass"] * (
        doppler_factor(x["ra"], x["dec"], DOPPLER)
        / doppler_factor(y["ra"], y["dec"], DOPPLER)
    )
    # same t_det: move geocent_time by the change in delay
    _, tx, _ = _forward(plain, x)
    _, ty, _ = _forward(plain, y)
    y["geocent_time"] = y["geocent_time"] + (tx["t_det"] - ty["t_det"]) * SCALE
    _, fx, _ = _forward(reparam, x)
    _, fy, _ = _forward(reparam, y)
    # GPS times near 1e9 s resolve ~0.24 us = 5e-5 in prime units
    np.testing.assert_allclose(fx["t_det"], fy["t_det"], atol=1e-4)


def test_adaptive_update_finds_the_frequency_of_least_spread():
    """Points whose arrival time at 25 Hz is the same up to 0.1 ms: the
    adaptive update must land near 25 Hz and recentre tau_ref."""
    f_true = 25.0
    reparam = _ref_reparam("adaptive", doppler_vector=DOPPLER,
                           frequency_range=(8.0, 300.0))
    plain = _ref_reparam(reference_frequency=None)
    x = _full_points(2000, 7)
    mc_earth = x["chirp_mass"] * doppler_factor(x["ra"], x["dec"], DOPPLER)
    tau = pn_time_to_merger(mc_earth, x["mass_ratio"], x["chi_1"], x["chi_2"], f_true)
    rng = np.random.default_rng(8)
    target = tau - np.median(tau) + rng.normal(0, 1e-4, len(x))
    _, t_det, _ = _forward(plain, x)
    x["geocent_time"] = x["geocent_time"] + (target - t_det["t_det"] * SCALE)

    reparam.update(x)
    assert reparam.reference_frequency == pytest.approx(f_true, rel=0.05)
    _, at_f, _ = _forward(reparam, x)
    assert np.std(at_f["t_det"]) * SCALE < 2e-4
    assert np.std(t_det["t_det"]) * SCALE > 1e-3  # the merger time is spread


def test_old_pickles_load_as_merger_time():
    import pickle

    reparam = _reparam_for_pickle()
    state = reparam.__dict__.copy()
    for key in ("reference_frequency", "adaptive", "frequency_range",
                "doppler_vector", "_chirp_mass", "_mass_ratio", "_spins",
                "_tau_ref"):
        state.pop(key)
    old = DetectorCenterTimeReparameterisation.__new__(
        DetectorCenterTimeReparameterisation
    )
    old.__dict__.update(state)
    loaded = pickle.loads(pickle.dumps(old))
    assert loaded.reference_frequency is None
    x = _full_points(10, 9)
    _, a, _ = _forward(loaded, x)
    _, b, _ = _forward(reparam, x)
    np.testing.assert_array_equal(a["t_det"], b["t_det"])


def _reparam_for_pickle():
    return _ref_reparam(reference_frequency=None)


@pytest.mark.integration_test
def test_reference_frequency_through_gw_flow_proposal():
    """The real ``CombinedReparameterisation`` built from the triangular
    wiring accepts the order (its inverse reads the masses and spins) and
    round-trips."""
    from nessai.model import Model

    from nessai_gw.group_mixture import triangular_group_reparameterisations
    from nessai_gw.proposals import GWFlowProposal

    names = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "ra", "dec",
             "theta_jn", "psi", "phase", "geocent_time"]
    bounds = {
        "chirp_mass": (1.18, 1.19), "mass_ratio": (0.5, 1.0),
        "chi_1": (-0.05, 0.05), "chi_2": (-0.05, 0.05),
        "ra": (0.0, 2 * np.pi), "dec": (-np.pi / 2, np.pi / 2),
        "theta_jn": (0.0, np.pi), "psi": (0.0, np.pi), "phase": (0.0, 2 * np.pi),
        "geocent_time": (REFERENCE_TIME - 0.1, REFERENCE_TIME + 0.1),
    }

    class _Model(Model):
        def __init__(self):
            self.names = list(names)
            self.bounds = {k: np.asarray(v, dtype=float) for k, v in bounds.items()}

        def log_prior(self, x):
            return np.zeros(len(np.atleast_1d(x)))

        def log_likelihood(self, x):
            return np.zeros(len(np.atleast_1d(x)))

    reps = triangular_group_reparameterisations(
        names, REFERENCE_TIME, vertex=ET_EMR_VERTEX, effective_spin=True,
        doppler_vector=DOPPLER, time_reference_frequency="adaptive",
    )
    assert list(reps)[0] == "geocent_time"
    proposal = GWFlowProposal(_Model(), poolsize=100, reparameterisations=reps,
                              fallback_reparameterisation="zscore")
    proposal.set_rescaling()
    assert "t_det" in proposal.prime_parameters

    rng = np.random.default_rng(10)
    n = 300
    x = _full_points(n, 11)
    x_full = empty_structured_array(n, names=names)
    for name in names:
        x_full[name] = (x[name] if name in x.dtype.names
                        else rng.uniform(*bounds[name], n))
    proposal.check_state(x_full)        # the adaptive update, as before training
    (time_reparam,) = [
        r for r in proposal._reparameterisation.values()
        if isinstance(r, DetectorCenterTimeReparameterisation)
    ]
    assert time_reparam.reference_frequency != pytest.approx(np.sqrt(8 * 300))
    x_prime, log_j = proposal.rescale(x_full.copy())
    x_back, log_j_back = proposal.inverse_rescale(x_prime)
    # boundary inversion of the mass ratio may append mirror copies in
    # blocks of n rows, each of which maps back to the same point
    for k in range(len(x_prime) // n):
        rows = slice(k * n, (k + 1) * n)
        np.testing.assert_allclose(
            x_back["geocent_time"][rows] - REFERENCE_TIME,
            x_full["geocent_time"] - REFERENCE_TIME, atol=1e-10,
        )
    np.testing.assert_allclose(log_j + log_j_back, 0.0, atol=1e-8)


def test_triangular_wiring_time_reference_frequency():
    from nessai_gw.group_mixture import triangular_group_reparameterisations

    names = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "ra", "dec",
             "theta_jn", "psi", "phase", "geocent_time"]
    reps = triangular_group_reparameterisations(
        names, REFERENCE_TIME, effective_spin=True, doppler_vector=DOPPLER,
        time_reference_frequency=20,
    )
    entry = reps["geocent_time"]
    assert list(reps)[0] == "geocent_time"
    assert entry["reference_frequency"] == 20.0
    assert entry["spins"] == ["chi_1", "chi_2"]
    np.testing.assert_array_equal(entry["doppler_vector"], DOPPLER)
    default = triangular_group_reparameterisations(names, REFERENCE_TIME)
    assert "reference_frequency" not in default["geocent_time"]
    with pytest.raises(ValueError, match="time_reference_frequency needs"):
        triangular_group_reparameterisations(
            [n for n in names if n != "mass_ratio"], REFERENCE_TIME,
            time_reference_frequency="adaptive",
        )
