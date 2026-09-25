"""Tests for the ``log-mass-ratio`` (ln q) reparameterisation."""

import numpy as np
import pytest
from nessai.livepoint import numpy_array_to_live_points

from nessai_gw.group_mixture import triangular_group_reparameterisations
from nessai_gw.reparameterisations import known_reparameterisations

Q_MIN, Q_MAX = 0.125, 1.0


@pytest.fixture
def reparam():
    known = known_reparameterisations.get("log-mass-ratio")
    cls, config = known.class_fn, dict(known.keyword_arguments)
    config["update_bounds"] = False
    config["detect_edges"] = False
    config["boundary_inversion"] = False
    return cls(
        parameters="mass_ratio",
        prior_bounds={"mass_ratio": [Q_MIN, Q_MAX]},
        **config,
    )


def _live(q):
    return numpy_array_to_live_points(q[:, None], ["mass_ratio"])


def test_is_rescaled_log_q(reparam):
    q = np.random.default_rng(0).uniform(Q_MIN, Q_MAX, 500)
    x = _live(q)
    x_prime = numpy_array_to_live_points(np.zeros((500, 1)), ["mass_ratio_prime"])
    _, x_prime, log_j = reparam.reparameterise(x, x_prime, np.zeros(500))
    lo, hi = np.log(Q_MIN), np.log(Q_MAX)
    expected = 2 * (np.log(q) - lo) / (hi - lo) - 1
    np.testing.assert_allclose(x_prime["mass_ratio_prime"], expected)
    # d x' / d q = 2 / ((hi - lo) q)
    np.testing.assert_allclose(log_j, np.log(2 / (hi - lo)) - np.log(q))


def test_round_trip(reparam):
    q = np.random.default_rng(1).uniform(Q_MIN, Q_MAX, 500)
    x = _live(q)
    x_prime = numpy_array_to_live_points(np.zeros((500, 1)), ["mass_ratio_prime"])
    _, x_prime, log_j = reparam.reparameterise(x, x_prime, np.zeros(500))
    back = _live(np.zeros(500))
    back, _, log_j_inv = reparam.inverse_reparameterise(
        back, x_prime, np.zeros(500)
    )
    np.testing.assert_allclose(back["mass_ratio"], q, rtol=1e-12)
    np.testing.assert_allclose(log_j, -log_j_inv, atol=1e-12)


NAMES = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "theta_jn", "psi",
         "phase", "ra", "dec", "geocent_time"]


def test_group_wiring_defaults_to_log_q():
    reps = triangular_group_reparameterisations(NAMES, 1187008882.4)
    assert reps["mass_ratio"] == {"reparameterisation": "log-mass-ratio"}
    off = triangular_group_reparameterisations(
        NAMES, 1187008882.4, log_mass_ratio=False
    )
    assert "mass_ratio" not in off  # left to the default ``mass_ratio``


def test_log_q_sorts_after_effective_spin():
    """``effective-spin`` reads x-space mass_ratio on its inverse, so it must
    come first (the inverse pass walks the order in reverse)."""
    names = list(triangular_group_reparameterisations(
        NAMES, 1187008882.4, effective_spin=True
    ))
    assert names.index("chi_1") < names.index("mass_ratio")
