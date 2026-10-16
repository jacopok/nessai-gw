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


# --- phase at a reference frequency ---------------------------------------------

_PHASE_GRID = np.geomspace(8.0, 300.0, 33)
_F_TRUE = float(_PHASE_GRID[19])          # ~47 Hz, on the search grid
_DOPPLER = [3e-5, -8e-5, 4e-5]


def _fake_waveform_phase(params, frequencies):
    """A stand-in for the 22-mode phase: a time term (linear in f, which the
    tangent intercept removes), a chirp term whose intercept depends on the
    frequency, and a constant; smooth in the parameters."""
    f = np.atleast_1d(frequencies)[None, :]
    time = 0.02 * params["chi_1"][:, None]
    chirp = (30.0 * (params["chirp_mass"] - 1.2) / 0.01
             + 1e-3 * params["lambda_1"])[:, None]
    const = 5.0 * params["chi_2"][:, None]
    return -2 * np.pi * f * time + chirp * (f / 30.0) ** (-5.0 / 3.0) + const


def _intercept(params, f):
    """Exact tangent intercept of :func:`_fake_waveform_phase` at ``f``."""
    chirp = 30.0 * (params["chirp_mass"] - 1.2) / 0.01 + 1e-3 * params["lambda_1"]
    return 5.0 * params["chi_2"] + chirp * (8.0 / 3.0) * (f / 30.0) ** (-5.0 / 3.0)


def _waveform_reparam(**kwargs):
    kwargs.setdefault("doppler_vector", _DOPPLER)
    return PolarisationPhaseReparameterisation(
        parameters="phase", prior_bounds={"phase": [0, 2 * np.pi]},
        waveform_phase=_fake_waveform_phase, **kwargs,
    )


def _helix_points(n, seed, concentrated=True):
    """Points whose ``phase + sign(cos theta_jn) psi + a(f_true) / 2`` is
    concentrated (as on a posterior), so that ``delta_phase`` at the merger
    winds around the intrinsic parameters."""
    rng = np.random.default_rng(seed)
    x = dict_to_live_points({
        "chirp_mass": rng.uniform(1.19, 1.21, n),
        "mass_ratio": rng.uniform(0.7, 1.0, n),
        "chi_1": rng.uniform(-0.05, 0.05, n),
        "chi_2": rng.uniform(-0.05, 0.05, n),
        "lambda_1": rng.uniform(0, 2000, n),
        "lambda_2": rng.uniform(0, 2000, n),
        "ra": rng.uniform(0, 2 * np.pi, n),
        "dec": np.arcsin(rng.uniform(-1, 1, n)),
        "psi": rng.uniform(0, np.pi, n),
        "theta_jn": np.arccos(rng.uniform(-1, 1, n)),
        "phase": rng.uniform(0, 2 * np.pi, n),
    })
    if concentrated:
        reparam = _waveform_reparam()
        a = _intercept(reparam._params(x), _F_TRUE)
        sign = np.sign(np.cos(x["theta_jn"]))
        x["phase"] = np.mod(np.pi - sign * x["psi"] - a / 2
                            + rng.normal(0, 0.1, n), 2 * np.pi)
    return x


def _forward_phase(reparam, x):
    x_prime = empty_structured_array(len(x), names=reparam.prime_parameters)
    return reparam.reparameterise(x.copy(), x_prime, np.zeros(len(x)))


def _concentration(delta_phase_prime):
    return np.abs(np.mean(np.exp(1j * np.pi * (delta_phase_prime + 1))))


def test_waveform_phase_requires_the_intrinsic_parameters():
    reparam = _waveform_reparam()
    assert set(reparam.requires) == {
        "psi", "theta_jn", "chirp_mass", "mass_ratio", "chi_1", "chi_2",
        "lambda_1", "lambda_2", "ra", "dec",
    }
    plain = PolarisationPhaseReparameterisation(parameters="phase")
    assert plain.requires == ["psi", "theta_jn"]
    assert plain.reference_frequency is None


def test_waveform_phase_unwinds_the_helix_and_inverts():
    reparam = _waveform_reparam()
    plain = PolarisationPhaseReparameterisation(parameters="phase")
    x = _helix_points(3000, 1)
    _, at_merger, log_j_plain = _forward_phase(plain, x)
    assert _concentration(at_merger["delta_phase"]) < 0.2

    reparam.update(x)
    assert reparam.reference_frequency == pytest.approx(_F_TRUE)
    _, at_f, log_j = _forward_phase(reparam, x)
    assert _concentration(at_f["delta_phase"]) > 0.95
    # the offset depends only on the intrinsic parameters: same Jacobian
    np.testing.assert_array_equal(log_j, log_j_plain)

    x_in = x.copy()
    x_in["phase"] = np.nan
    x_out, _, log_j_back = reparam.inverse_reparameterise(
        x_in, at_f.copy(), log_j.copy())
    d = np.angle(np.exp(1j * (x_out["phase"] - x["phase"])))
    np.testing.assert_allclose(d, 0.0, atol=1e-10)
    np.testing.assert_allclose(log_j_back, 0.0, atol=1e-12)


def test_waveform_phase_left_out_when_it_does_not_help():
    """Uniform phases (early in a run): no frequency concentrates
    delta_phase, so the phase stays at the merger."""
    reparam = _waveform_reparam()
    x = _helix_points(3000, 2, concentrated=False)
    reparam.update(x)
    assert reparam.reference_frequency is None
    assert reparam._shift(x) == 0.0


def test_waveform_phase_offset_is_group_invariant():
    """The offset reads the chirp mass in the Earth's frame: moving the sky
    (as the triangular group does) with that chirp mass held fixed leaves it
    unchanged, so the group's translations of the phase commute with it."""
    from nessai_gw.reparameterisations.mass import doppler_factor

    reparam = _waveform_reparam()
    x = _helix_points(3000, 3)
    reparam.update(x)
    moved = x.copy()
    rng = np.random.default_rng(4)
    moved["ra"] = rng.uniform(0, 2 * np.pi, len(x))
    moved["dec"] = np.arcsin(rng.uniform(-1, 1, len(x)))
    moved["psi"] = rng.uniform(0, np.pi, len(x))
    moved["theta_jn"] = np.pi - x["theta_jn"]
    moved["chirp_mass"] = (
        x["chirp_mass"] * doppler_factor(x["ra"], x["dec"], _DOPPLER)
        / doppler_factor(moved["ra"], moved["dec"], _DOPPLER)
    )
    np.testing.assert_allclose(reparam._shift(moved), reparam._shift(x),
                               atol=1e-9)
    # without the Doppler factor it would not be invariant
    sky_only = x.copy()
    sky_only["ra"], sky_only["dec"] = moved["ra"], moved["dec"]
    assert not np.allclose(reparam._shift(sky_only), reparam._shift(x))


def test_waveform_phase_pickles_by_name_and_keeps_its_offset():
    import pickle

    reparam = PolarisationPhaseReparameterisation(
        parameters="phase",
        waveform_phase="nessai_gw.reparameterisations.phase:no_such_function",
    )
    assert reparam._waveform() is None     # warns
    reparam.update(_helix_points(100, 5))  # nothing to fit with

    fitted = _waveform_reparam()
    x = _helix_points(2000, 6)
    fitted.update(x)
    state = fitted.__getstate__()
    assert state["_waveform_phase"].endswith(":_fake_waveform_phase")
    assert state["_waveform_phase_fn"] is None
    fitted._waveform_phase = "no_such_module:phase"   # e.g. a monitor's venv
    restored = pickle.loads(pickle.dumps(fitted))
    np.testing.assert_allclose(restored._shift(x), fitted._shift(x))
    assert restored._waveform() is None
    restored.update(x)                                # keeps the offset
    np.testing.assert_allclose(restored._shift(x), fitted._shift(x))


def test_waveform_phase_old_pickles_load():
    plain = PolarisationPhaseReparameterisation(parameters="phase")
    state = dict(plain.__dict__)
    for key in ("_waveform_phase", "_waveform_phase_fn", "_offset",
                "reference_frequency", "doppler_vector"):
        state.pop(key)
    restored = PolarisationPhaseReparameterisation.__new__(
        PolarisationPhaseReparameterisation)
    restored.__setstate__(state)
    x = _helix_points(50, 7)
    _, a, _ = _forward_phase(restored, x)
    _, b, _ = _forward_phase(plain, x)
    np.testing.assert_array_equal(a["delta_phase"], b["delta_phase"])


def test_triangular_wiring_waveform_phase():
    from nessai_gw.group_mixture import triangular_group_reparameterisations

    names = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "lambda_1",
             "lambda_2", "ra", "dec", "theta_jn", "psi", "phase",
             "geocent_time"]
    reps = triangular_group_reparameterisations(
        names, 1187008882.4, effective_spin=True,
        effective_tidal_deformability=True, doppler_vector=_DOPPLER,
        time_reference_frequency="adaptive", waveform_phase="pkg.mod:phase",
    )
    entry = reps["phase"]
    assert entry["reparameterisation"] == "polarisation-phase"
    assert entry["waveform_phase"] == "pkg.mod:phase"
    assert entry["spins"] == ["chi_1", "chi_2"]
    assert entry["tides"] == ["lambda_1", "lambda_2"]
    assert entry["doppler_vector"] == _DOPPLER
    # ahead of the intrinsic reparameterisations it reads on the inverse
    order = list(reps)
    assert order.index("phase") < order.index("chi_1")
    assert order.index("phase") < order.index("lambda_1")
    assert "waveform_phase" not in triangular_group_reparameterisations(
        names, 1187008882.4)["phase"]
    with pytest.raises(ValueError, match="waveform_phase needs"):
        triangular_group_reparameterisations(
            names, 1187008882.4, waveform_phase="pkg.mod:phase",
            phase_coordinates="arg-alpha-beta")
    with pytest.raises(ValueError, match="waveform_phase needs"):
        triangular_group_reparameterisations(
            [n for n in names if n != "mass_ratio"], 1187008882.4,
            waveform_phase="pkg.mod:phase")
