import numpy as np
import pytest
from nessai.livepoint import (
    dict_to_live_points,
    empty_structured_array,
)

from nessai_gw.reparameterisations import (
    AlignedSpinReparameterisation,
)


def test_aligned_spin_init():
    """Assert a_max is taken from the (symmetric) prior bounds."""
    reparam = AlignedSpinReparameterisation(
        parameters="chi_1", prior_bounds={"chi_1": [-0.99, 0.99]}
    )
    assert reparam.a_max == pytest.approx(0.99)
    assert reparam.prime_parameters == ["chi_1_prime"]


def test_aligned_spin_init_multiple_parameters():
    """Assert an error is raised if more than one parameter is given."""
    with pytest.raises(RuntimeError, match="only supports one parameter"):
        AlignedSpinReparameterisation(
            parameters=["chi_1", "chi_2"],
            prior_bounds={"chi_1": [-1, 1], "chi_2": [-1, 1]},
        )


def test_aligned_spin_asymmetric_bounds_warns(caplog):
    """Assert a warning is emitted for asymmetric bounds."""
    reparam = AlignedSpinReparameterisation(
        parameters="chi_1", prior_bounds={"chi_1": [-0.5, 0.99]}
    )
    assert reparam.a_max == pytest.approx(0.99)
    assert "not symmetric" in caplog.text


def test_cdf_matches_numerical_integral():
    """Assert the analytic CDF matches a numerical integral of the PDF."""
    a_max = 0.99
    reparam = AlignedSpinReparameterisation(
        parameters="chi_1", prior_bounds={"chi_1": [-a_max, a_max]}
    )
    chi = np.linspace(-a_max, a_max, 5000)
    pdf = np.exp(reparam._log_pdf(chi))
    numerical_cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(chi))]
    )
    np.testing.assert_allclose(reparam._cdf(chi), numerical_cdf, atol=1e-3)
    assert reparam._cdf(-a_max) == pytest.approx(0.0)
    assert reparam._cdf(a_max) == pytest.approx(1.0)
    assert reparam._cdf(0.0) == pytest.approx(0.5)


@pytest.mark.integration_test
def test_prime_prior_is_standard_normal():
    """Samples drawn from the aligned-spin prior map to a standard normal."""
    a_max = 0.99
    reparam = AlignedSpinReparameterisation(
        parameters="chi_1", prior_bounds={"chi_1": [-a_max, a_max]}
    )
    rng = np.random.default_rng(1234)
    chi = reparam._inverse_cdf(rng.uniform(0, 1, 100_000))
    x = dict_to_live_points({"chi_1": chi})
    x_prime = empty_structured_array(chi.size, names=["chi_1_prime"])
    _, x_prime, log_j = reparam.reparameterise(x, x_prime, np.zeros(chi.size))
    u = x_prime["chi_1_prime"]
    assert abs(np.mean(u)) < 0.02
    assert abs(np.std(u) - 1.0) < 0.02
    # log_j should equal log N(u) - log p(chi)
    expected = -0.5 * u**2 - 0.5 * np.log(2 * np.pi) - reparam._log_pdf(chi)
    np.testing.assert_allclose(log_j, -expected, rtol=1e-6)


@pytest.mark.integration_test
@pytest.mark.parametrize("a_max", [0.99, 0.05])
def test_aligned_spin_invertible(a_max):
    """Assert the reparameterisation round-trips to well within nessai's
    verify_rescaling tolerance (np.allclose, atol=1e-8), for uniform and
    prior-distributed spins (clustered at zero)."""
    n = 1000
    reparam = AlignedSpinReparameterisation(
        parameters="chi_1", prior_bounds={"chi_1": [-a_max, a_max]}
    )
    rng = np.random.default_rng(0)
    chi = np.concatenate(
        [
            rng.uniform(-a_max, a_max, n // 2),
            rng.uniform(0, a_max, n // 2) * rng.uniform(-1, 1, n // 2),
        ]
    )
    x = dict_to_live_points({"chi_1": chi})
    x_prime = empty_structured_array(n, names=["chi_1_prime"])
    log_j = np.zeros(n)
    x_f, x_prime_f, log_j_f = reparam.reparameterise(x, x_prime, log_j)

    x_in = x_f.copy()
    x_in["chi_1"] = np.nan
    x_i, _, log_j_i = reparam.inverse_reparameterise(
        x_in, x_prime_f.copy(), np.zeros(n)
    )
    np.testing.assert_allclose(x_i["chi_1"], chi, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(log_j_i, -log_j_f, rtol=0, atol=1e-9)


@pytest.mark.integration_test
def test_aligned_spin_jacobian_finite_difference():
    """Assert the analytic Jacobian matches a finite-difference estimate."""
    a_max = 0.99
    reparam = AlignedSpinReparameterisation(
        parameters="chi_1", prior_bounds={"chi_1": [-a_max, a_max]}
    )
    chi = np.array([-0.7, -0.2, 0.05, 0.3, 0.8])
    h = 1e-7

    def forward(c):
        x = dict_to_live_points({"chi_1": c})
        x_prime = empty_structured_array(c.size, names=["chi_1_prime"])
        _, x_prime, log_j = reparam.reparameterise(
            x, x_prime, np.zeros(c.size)
        )
        return x_prime["chi_1_prime"], log_j

    u0, log_j = forward(chi)
    u1, _ = forward(chi + h)
    fd = np.log(np.abs((u1 - u0) / h))
    np.testing.assert_allclose(log_j, fd, rtol=1e-4)
