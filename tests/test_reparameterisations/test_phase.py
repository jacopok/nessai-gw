from unittest.mock import create_autospec, patch

import numpy as np
import pytest
from nessai.livepoint import (
    dict_to_live_points,
    empty_structured_array,
)
from nessai.utils.testing import assert_structured_arrays_equal

from nessai_gw.reparameterisations import (
    ArgAlphaBetaReparameterisation,
    DeltaPhaseReparameterisation,
    FittedPhaseRotation,
    PolarisationPhaseReparameterisation,
)


@pytest.fixture
def delta_phase_reparam():
    return create_autospec(DeltaPhaseReparameterisation)


def test_delta_phase_init(delta_phase_reparam):
    """Assert the parent method is called and the parameters are set."""
    parameters = "phase"
    prior_bounds = {"phase": [0, 6.28]}
    with patch(
        "nessai_gw.reparameterisations.phase.Reparameterisation.__init__"
    ) as mock:
        DeltaPhaseReparameterisation.__init__(
            delta_phase_reparam,
            parameters=parameters,
            prior_bounds=prior_bounds,
        )
    mock.assert_called_once_with(
        parameters=parameters, prior_bounds=prior_bounds
    )
    assert delta_phase_reparam.requires == ["psi", "theta_jn"]
    assert delta_phase_reparam.prime_parameters == ["delta_phase"]


def test_delta_phase_reparameterise(delta_phase_reparam):
    """Assert the correct value is returned"""
    delta_phase_reparam.parameters = ["phase"]
    delta_phase_reparam.prime_parameters = ["delta_phase"]

    x = dict(phase=1.0, theta_jn=0.0, psi=0.5)
    x_prime = dict(delta_phase=np.nan, theta_jn=0.0, psi=0.5)
    log_j = 0

    (
        x_out,
        x_prime_out,
        log_j_out,
    ) = DeltaPhaseReparameterisation.reparameterise(
        delta_phase_reparam, x, x_prime, log_j
    )
    assert x_out == x
    assert x_prime_out["delta_phase"] == 1.5
    assert log_j_out == 0


def test_delta_phase_inverse_reparameterise(delta_phase_reparam):
    """Assert the correct value is returned"""
    delta_phase_reparam.parameters = ["phase"]
    delta_phase_reparam.prime_parameters = ["delta_phase"]

    x = dict(phase=np.nan, theta_jn=0.0, psi=0.5)
    x_prime = dict(delta_phase=0.5, theta_jn=0.0, psi=0.5)
    log_j = 0

    (
        x_out,
        x_prime_out,
        log_j_out,
    ) = DeltaPhaseReparameterisation.inverse_reparameterise(
        delta_phase_reparam, x, x_prime, log_j
    )
    assert x_prime_out == x_prime
    assert x["phase"] == 0.0
    assert log_j_out == 0


@pytest.mark.integration_test
def test_delta_phase_inverse_invertible():
    """Assert the reparameterisation is invertible"""
    n = 10
    parameters = ["phase"]
    prior_bounds = {"phase": [0.0, 2 * np.pi]}
    reparam = DeltaPhaseReparameterisation(
        parameters=parameters, prior_bounds=prior_bounds
    )
    x = dict_to_live_points(
        {
            "phase": np.random.uniform(0, 2 * np.pi, n),
            "psi": np.random.uniform(0, np.pi, n),
            "theta_jn": np.random.uniform(0, np.pi, n),
        }
    )
    x_prime = empty_structured_array(
        n, names=["delta_phase", "theta_jn", "psi"]
    )
    x_prime["psi"] = x["psi"]
    x_prime["theta_jn"] = x["theta_jn"]
    log_j = np.zeros(n)
    x_f, x_prime_f, log_j_f = reparam.reparameterise(x, x_prime, log_j)

    x_in = x_f.copy()
    x_in["phase"] = np.nan

    assert_structured_arrays_equal(x_f, x)
    np.testing.assert_array_equal(log_j_f, log_j)
    x_i, x_prime_i, log_j_i = reparam.inverse_reparameterise(
        x_in, x_prime_f.copy(), log_j_f.copy()
    )
    assert_structured_arrays_equal(x_prime_i, x_prime_f)
    assert_structured_arrays_equal(x_i, x, rtol=1e-10)
    np.testing.assert_array_equal(log_j_i, log_j_f)


class TestPolarisationPhase:
    """Tests for :class:`PolarisationPhaseReparameterisation`."""

    prior_bounds = {"phase": [0.0, 2 * np.pi]}

    def _reparam(self):
        return PolarisationPhaseReparameterisation(
            parameters="phase", prior_bounds=self.prior_bounds
        )

    def test_init(self):
        reparam = self._reparam()
        assert reparam.prime_parameters == ["delta_phase"]
        assert reparam.requires == ["psi", "theta_jn"]
        assert reparam.scale == 1.0

    @pytest.mark.parametrize("scale", [1.0, 2.0])
    @pytest.mark.parametrize("theta_jn, sign", [(0.6, 1.0), (2.5, -1.0)])
    def test_reparameterise_values(self, theta_jn, sign, scale):
        reparam = PolarisationPhaseReparameterisation(
            parameters="phase", prior_bounds=self.prior_bounds, scale=scale
        )
        n = 20
        phase = np.random.uniform(0, 2 * np.pi, n)
        psi = np.random.uniform(0, np.pi, n)
        tjn = np.full(n, theta_jn)
        x = dict_to_live_points({"phase": phase, "psi": psi, "theta_jn": tjn})
        x_prime = empty_structured_array(
            n, names=["delta_phase", "psi", "theta_jn"]
        )
        x_prime["psi"] = psi
        x_prime["theta_jn"] = tjn
        log_j = np.zeros(n)
        _, x_prime, log_j = reparam.reparameterise(x, x_prime, log_j)

        # prime coordinate is a single [-1, 1) value
        assert np.all(x_prime["delta_phase"] >= -1.0)
        assert np.all(x_prime["delta_phase"] < 1.0)
        angle = (x_prime["delta_phase"] + 1.0) * np.pi
        np.testing.assert_allclose(
            angle,
            ((phase + sign * psi) * scale) % (2 * np.pi),
            atol=1e-9,
        )
        np.testing.assert_allclose(log_j, np.log(scale / np.pi), atol=1e-12)

    @pytest.mark.integration_test
    @pytest.mark.parametrize("scale", [1.0, 2.0])
    @pytest.mark.parametrize("theta_jn", [0.6, 2.5])
    def test_invertible(self, theta_jn, scale):
        reparam = PolarisationPhaseReparameterisation(
            parameters="phase", prior_bounds=self.prior_bounds, scale=scale
        )
        n = 50
        phase = np.random.uniform(0, 2 * np.pi, n)
        psi = np.random.uniform(0, np.pi, n)
        tjn = np.full(n, theta_jn)
        x = dict_to_live_points(
            {"phase": phase, "psi": psi, "theta_jn": tjn}
        )
        names = ["delta_phase", "psi", "theta_jn"]
        x_prime = empty_structured_array(n, names=names)
        x_prime["psi"] = psi
        x_prime["theta_jn"] = tjn
        log_j = np.zeros(n)

        x_f, x_prime_f, log_j_f = reparam.reparameterise(
            x.copy(), x_prime.copy(), log_j.copy()
        )
        # forward leaves the physical point untouched
        np.testing.assert_array_equal(x_f["phase"], phase)
        x_in = x_f.copy()
        x_in["phase"] = np.nan
        x_i, _, log_j_i = reparam.inverse_reparameterise(
            x_in, x_prime_f.copy(), log_j_f.copy()
        )
        if scale == 1.0:
            np.testing.assert_allclose(x_i["phase"], phase, rtol=1e-9)
        np.testing.assert_allclose(log_j_i, 0.0, atol=1e-10)


class TestArgAlphaBeta:
    """Tests for :class:`ArgAlphaBetaReparameterisation`."""

    prior_bounds = {"psi": [0.0, np.pi], "phase": [0.0, 2 * np.pi]}

    def _reparam(self):
        return ArgAlphaBetaReparameterisation(
            parameters=["psi", "phase"], prior_bounds=self.prior_bounds
        )

    def test_init(self):
        reparam = self._reparam()
        assert reparam.parameters == ["psi", "phase"]
        assert reparam.prime_parameters == ["arg_alpha", "arg_beta"]
        np.testing.assert_allclose(reparam._log_j, np.log(2.0 / np.pi**2))

    def test_bad_parameters(self):
        with pytest.raises(RuntimeError, match="must act on"):
            ArgAlphaBetaReparameterisation(
                parameters=["phase"], prior_bounds={"phase": [0, 6.28]}
            )

    def test_reparameterise_values(self):
        reparam = self._reparam()
        n = 50
        phase = np.random.uniform(0, 2 * np.pi, n)
        psi = np.random.uniform(0, np.pi, n)
        x = dict_to_live_points({"phase": phase, "psi": psi})
        x_prime = empty_structured_array(n, names=["arg_alpha", "arg_beta"])
        log_j = np.zeros(n)
        _, x_prime, log_j = reparam.reparameterise(x, x_prime, log_j)

        for name in ("arg_alpha", "arg_beta"):
            assert np.all(x_prime[name] >= -1.0)
            assert np.all(x_prime[name] < 1.0)
        np.testing.assert_allclose(
            (x_prime["arg_alpha"] + 1.0) * np.pi,
            np.mod(phase - psi, 2 * np.pi),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            (x_prime["arg_beta"] + 1.0) * np.pi,
            np.mod(phase + psi, 2 * np.pi),
            atol=1e-9,
        )
        np.testing.assert_allclose(log_j, np.log(2.0 / np.pi**2), atol=1e-12)

    @pytest.mark.integration_test
    def test_bijective_on_the_full_cell(self):
        """Exact round-trip for every (psi in [0, pi), phase in [0, 2 pi))."""
        reparam = self._reparam()
        n = 2000
        phase = np.random.uniform(0, 2 * np.pi, n)
        psi = np.random.uniform(0, np.pi, n)
        x = dict_to_live_points({"phase": phase, "psi": psi})
        x_prime = empty_structured_array(n, names=["arg_alpha", "arg_beta"])
        log_j = np.zeros(n)
        x_f, x_prime_f, log_j_f = reparam.reparameterise(
            x.copy(), x_prime.copy(), log_j.copy()
        )
        np.testing.assert_array_equal(x_f["phase"], phase)
        x_in = x_f.copy()
        x_in["phase"] = np.nan
        x_in["psi"] = np.nan
        x_i, _, log_j_i = reparam.inverse_reparameterise(
            x_in, x_prime_f.copy(), log_j_f.copy()
        )
        np.testing.assert_allclose(x_i["phase"], phase, atol=1e-8)
        np.testing.assert_allclose(x_i["psi"], psi, atol=1e-8)
        np.testing.assert_allclose(log_j_i, 0.0, atol=1e-10)


class TestFittedPhaseRotation:
    """Tests for :class:`FittedPhaseRotation`."""

    def _reparam(self, angle=0.3, psi0=0.5, phase0=1.2):
        return FittedPhaseRotation(
            parameters=["psi", "phase"], prior_bounds=None,
            angle=angle, psi0=psi0, phase0=phase0,
        )

    def test_init(self):
        reparam = self._reparam(angle=0.7, psi0=0.1, phase0=-0.2)
        assert reparam.parameters == ["psi", "phase"]
        assert reparam.prime_parameters == ["phase_rot_1", "phase_rot_2"]
        assert reparam.angle == pytest.approx(0.7)
        assert reparam.psi0 == pytest.approx(0.1)
        assert reparam.phase0 == pytest.approx(-0.2)

    def test_bad_parameters(self):
        with pytest.raises(RuntimeError, match="must act on"):
            FittedPhaseRotation(parameters=["phase"])

    def test_zero_angle_is_a_plain_shift(self):
        """angle=0 should reduce to p1=phase-phase0, p2=psi-psi0."""
        reparam = self._reparam(angle=0.0, psi0=0.5, phase0=1.2)
        n = 20
        phase = np.random.uniform(0, 2 * np.pi, n)
        psi = np.random.uniform(0, np.pi, n)
        x = dict_to_live_points({"phase": phase, "psi": psi})
        x_prime = empty_structured_array(n, names=["phase_rot_1", "phase_rot_2"])
        log_j = np.zeros(n)
        _, x_prime, log_j = reparam.reparameterise(x, x_prime, log_j)
        np.testing.assert_allclose(x_prime["phase_rot_1"], phase - 1.2)
        np.testing.assert_allclose(x_prime["phase_rot_2"], psi - 0.5)
        np.testing.assert_allclose(log_j, 0.0)

    def test_jacobian_is_unity(self):
        """A pure rotation + shift has unit Jacobian regardless of angle."""
        reparam = self._reparam(angle=1.234)
        n = 10
        x = dict_to_live_points({
            "phase": np.random.uniform(0, 2 * np.pi, n),
            "psi": np.random.uniform(0, np.pi, n),
        })
        x_prime = empty_structured_array(n, names=["phase_rot_1", "phase_rot_2"])
        log_j_in = np.random.uniform(-1, 1, n)
        _, _, log_j_out = reparam.reparameterise(x, x_prime, log_j_in.copy())
        np.testing.assert_allclose(log_j_out, log_j_in)

    @pytest.mark.parametrize("angle", [0.0, 0.7, -1.9, 3.0])
    def test_round_trip(self, angle):
        reparam = self._reparam(angle=angle)
        n = 500
        phase = np.random.uniform(0, 2 * np.pi, n)
        psi = np.random.uniform(0, np.pi, n)
        x = dict_to_live_points({"phase": phase, "psi": psi})
        x_prime = empty_structured_array(n, names=["phase_rot_1", "phase_rot_2"])
        log_j = np.zeros(n)
        x_f, x_prime_f, log_j_f = reparam.reparameterise(
            x.copy(), x_prime.copy(), log_j.copy()
        )
        x_in = x_f.copy()
        x_in["phase"] = np.nan
        x_in["psi"] = np.nan
        x_i, _, log_j_i = reparam.inverse_reparameterise(
            x_in, x_prime_f.copy(), log_j_f.copy()
        )
        np.testing.assert_allclose(x_i["phase"], phase, atol=1e-10)
        np.testing.assert_allclose(x_i["psi"], psi, atol=1e-10)
        np.testing.assert_allclose(log_j_i, 0.0, atol=1e-12)

    def test_rotation_preserves_the_norm(self):
        """A genuine rotation (about the fitted centre) preserves distances --
        a cheap sanity check that it isn't secretly rescaling."""
        reparam = self._reparam(angle=0.9, psi0=0.0, phase0=0.0)
        n = 30
        phase = np.random.uniform(-1, 1, n)
        psi = np.random.uniform(-1, 1, n)
        x = dict_to_live_points({"phase": phase, "psi": psi})
        x_prime = empty_structured_array(n, names=["phase_rot_1", "phase_rot_2"])
        _, x_prime, _ = reparam.reparameterise(x, x_prime, np.zeros(n))
        r_before = np.hypot(phase, psi)
        r_after = np.hypot(x_prime["phase_rot_1"], x_prime["phase_rot_2"])
        np.testing.assert_allclose(r_after, r_before, atol=1e-10)
