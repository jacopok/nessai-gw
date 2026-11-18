"""The tabulated effective-spin transform's cell lookups."""
import numpy as np

from nessai_gw._effective_spin_table import (
    SpinOrbitChart,
    TabulatedEffectiveSpinTransform,
    _bisect,
)


def test_bisect_matches_full_row_scan():
    rng = np.random.default_rng(0)
    rows = np.sort(rng.normal(size=(500, 37)), axis=1)
    values = np.concatenate([rng.normal(size=480) * 2, rows[:10, 0], rows[10:20, -1]])
    expected = np.clip(np.sum(rows <= values[:, None], axis=1) - 1, 0, 35)
    n = np.arange(len(values))
    assert np.array_equal(_bisect(lambda k: rows[n, k], values, 37), expected)


def test_round_trip():
    rng = np.random.default_rng(1)
    tr = TabulatedEffectiveSpinTransform(0.05, 0.05)
    q = rng.uniform(0.05, 1, 20000)
    chi1, chi2 = rng.uniform(-0.05, 0.05, (2, 20000))
    u, w, lj = tr.forward(chi1, chi2, q)
    c1, c2, lj2 = tr.inverse(u, w, q)
    np.testing.assert_allclose(c1, chi1, atol=1e-12)
    np.testing.assert_allclose(c2, chi2, atol=1e-11)
    np.testing.assert_allclose(lj2, lj, atol=1e-8)
    chart = SpinOrbitChart(tr)
    y, w, lj = chart.forward(chi1, chi2, q)
    c1, c2, lj2 = chart.inverse(y, w, q)
    np.testing.assert_allclose(c1, chi1, atol=1e-12)
    np.testing.assert_allclose(lj2, lj, atol=1e-8)
