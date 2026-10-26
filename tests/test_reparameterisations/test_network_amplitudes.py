"""Tests for the two-site amplitude reparameterisation."""

import numpy as np
import pytest

from nessai_gw._effective_distance import response_R
from nessai_gw._geometry import greenwich_mean_sidereal_time
from nessai_gw.reparameterisations import (
    NetworkAmplitudesReparameterisation,
    known_reparameterisations,
)
from nessai_gw.group_mixture import ET_EMR_DETECTOR_TENSORS
from nessai_gw.reparameterisations.network_amplitudes import (
    NETWORK_AMPLITUDE_PARAMETERS,
    triangle_channel_tensors,
)

REFERENCE_TIME = 1187008882.4
NAMES = NETWORK_AMPLITUDE_PARAMETERS + ["ra", "dec", "chirp_mass"]
PRIME = ["ln_amp_1", "arg_amp_1", "ln_amp_2", "arg_amp_2"]


def l_tensor(u, v):
    u, v = (np.asarray(a, float) / np.linalg.norm(a) for a in (u, v))
    v = v - (v @ u) * u
    v /= np.linalg.norm(v)
    return 0.5 * (np.outer(u, u) - np.outer(v, v))


@pytest.fixture
def tensors():
    # two L-shaped detectors with unrelated orientations
    return np.stack([
        l_tensor([1.0, 0.2, 0.1], [0.0, 1.0, 0.3]),
        l_tensor([0.3, -0.5, 0.8], [0.9, 0.4, -0.1]),
    ])


@pytest.fixture
def reparam(tensors):
    return NetworkAmplitudesReparameterisation(
        parameters=NETWORK_AMPLITUDE_PARAMETERS,
        tensors=tensors,
        reference_time=REFERENCE_TIME,
        rng=np.random.default_rng(3),
    )


def samples(n, seed=1):
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=[(k, "f8") for k in NAMES])
    x["luminosity_distance"] = rng.uniform(100, 3000, n)
    x["theta_jn"] = np.arccos(rng.uniform(-0.95, 0.95, n))
    x["psi"] = rng.uniform(0, np.pi, n)
    x["phase"] = rng.uniform(0, 2 * np.pi, n)
    x["ra"] = rng.uniform(0, 2 * np.pi, n)
    x["dec"] = np.arcsin(rng.uniform(-1, 1, n))
    x["chirp_mass"] = rng.uniform(1.1, 1.3, n)
    return x


def forward(reparam, x):
    x_prime = np.zeros(len(x), dtype=[(k, "f8") for k in PRIME])
    return reparam.reparameterise(x.copy(), x_prime, np.zeros(len(x)))


def inverse(reparam, x_prime, x):
    out = x.copy()
    for k in NETWORK_AMPLITUDE_PARAMETERS:
        out[k] = np.nan
    return reparam.inverse_reparameterise(out, x_prime, np.zeros(len(x)))


def test_registered():
    assert "network-amplitudes" in known_reparameterisations


def test_amplitudes_are_the_response(reparam, tensors):
    x = samples(50)
    gmst = greenwich_mean_sidereal_time(REFERENCE_TIME)
    r = response_R(tensors, x["ra"], x["dec"], x["psi"], x["theta_jn"], gmst)
    expected = (r * np.exp(2j * x["phase"])[:, None]
                * (x["chirp_mass"] ** (5 / 6) / x["luminosity_distance"])[:, None])
    np.testing.assert_allclose(reparam.amplitudes(x), expected, rtol=1e-12)


def test_invariant_under_quarter_turn(reparam):
    x = samples(50)
    y = x.copy()
    y["psi"] = np.mod(x["psi"] + np.pi / 2, np.pi)
    y["phase"] = np.mod(x["phase"] - np.pi / 2, 2 * np.pi)
    np.testing.assert_allclose(
        reparam.amplitudes(y), reparam.amplitudes(x), rtol=1e-10, atol=1e-14
    )


def test_round_trip_up_to_the_quarter_turn(reparam):
    x = samples(500)
    _, x_prime, log_j = forward(reparam, x)
    y, _, log_j_inv = inverse(reparam, x_prime, x)
    np.testing.assert_allclose(y["luminosity_distance"], x["luminosity_distance"],
                               rtol=1e-8)
    np.testing.assert_allclose(y["theta_jn"], x["theta_jn"], atol=1e-8)
    # (psi, phase) is one of (psi, phase) + m (pi/2, -pi/2)
    m = np.round(np.mod(y["psi"] - x["psi"], np.pi) / (np.pi / 2)) % 2
    dphase = np.mod(y["phase"] - x["phase"] + np.pi / 2 * (m + 2 * np.arange(4)[:, None]),
                    2 * np.pi)
    close = np.minimum(dphase, 2 * np.pi - dphase).min(axis=0)
    assert np.all(close < 1e-7)
    np.testing.assert_allclose(log_j_inv, -log_j, rtol=1e-8)
    # the four branches are all drawn
    branch = np.round(np.mod(y["phase"] - x["phase"], 2 * np.pi) / (np.pi / 2)) % 4
    assert set(branch.astype(int)) == {0, 1, 2, 3}


def test_jacobian(reparam):
    x = samples(20, seed=4)
    _, x_prime, log_j = forward(reparam, x)
    eps = 1e-6
    for i in range(len(x)):
        jac = np.empty((4, 4))
        for j, name in enumerate(NETWORK_AMPLITUDE_PARAMETERS):
            cols = []
            for sign in (1, -1):
                y = x[i:i + 1].copy()
                y[name] += sign * eps * (x[name][i] if name == "luminosity_distance" else 1)
                cols.append(forward(reparam, y)[1])
            hi, lo = cols
            step = 2 * eps * (x[name][i] if name == "luminosity_distance" else 1)
            for k, p in enumerate(PRIME):
                diff = hi[p][0] - lo[p][0]
                if p.startswith("arg"):
                    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
                jac[k, j] = diff / step
        assert np.log(abs(np.linalg.det(jac))) == pytest.approx(log_j[i], abs=1e-5)


def test_tensor_shapes(tensors):
    with pytest.raises(ValueError, match="one detector per site"):
        NetworkAmplitudesReparameterisation(
            parameters=NETWORK_AMPLITUDE_PARAMETERS,
            tensors=np.concatenate([tensors, tensors]),
            reference_time=REFERENCE_TIME,
        )


def test_triangle_channels():
    a, e = triangle_channel_tensors(ET_EMR_DETECTOR_TENSORS)
    norm = np.sum(ET_EMR_DETECTOR_TENSORS[0] ** 2)
    # orthogonal, each with an interferometer's norm (to the ~1e-3 by which
    # the real triangle is not planar)
    assert abs(np.sum(a * e)) < 1e-3 * norm
    assert np.sum(a * a) == pytest.approx(norm, rel=5e-3)
    assert np.sum(e * e) == pytest.approx(norm, rel=5e-3)


def test_triangle_amplitudes_are_the_channels():
    reparam = NetworkAmplitudesReparameterisation(
        parameters=NETWORK_AMPLITUDE_PARAMETERS,
        tensors=ET_EMR_DETECTOR_TENSORS,
        reference_time=REFERENCE_TIME,
    )
    x = samples(50)
    gmst = greenwich_mean_sidereal_time(REFERENCE_TIME)
    single = response_R(ET_EMR_DETECTOR_TENSORS, x["ra"], x["dec"], x["psi"],
                        x["theta_jn"], gmst)
    channels = np.stack([
        (2 * single[:, 0] - single[:, 1] - single[:, 2]) / 3,
        (single[:, 2] - single[:, 1]) / np.sqrt(3.0),
    ], axis=-1)
    expected = (channels * np.exp(2j * x["phase"])[:, None]
                * (x["chirp_mass"] ** (5 / 6) / x["luminosity_distance"])[:, None])
    np.testing.assert_allclose(reparam.amplitudes(x), expected, rtol=1e-10)
    # round trip off the detector plane
    _, x_prime, log_j = forward(reparam, x)
    y, _, log_j_inv = inverse(reparam, x_prime, x)
    np.testing.assert_allclose(y["luminosity_distance"],
                               x["luminosity_distance"], rtol=1e-7)
    np.testing.assert_allclose(y["theta_jn"], x["theta_jn"], atol=1e-7)
    np.testing.assert_allclose(log_j_inv, -log_j, rtol=1e-8)
