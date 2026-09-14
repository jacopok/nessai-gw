"""Tests for the heteroscedasticity-based plateau/clump split."""

import numpy as np
import pytest

from nessai_gw._polarisation_branch import (
    best_changepoint_split, best_gap_split, best_polarisation_split,
    diagonal_split_mask, find_density_gap, fit_distance_floor_split,
    fit_phase_rotation, heteroscedasticity_ratio, local_scale_profile,
    polarisation_branch_flow_kwargs, robust_std, scan_split_thresholds,
)
from nessai_gw.reparameterisations.phase import FittedPhaseRotation


def synthetic(n=4000, seed=0, gap=True):
    """A toy sheet: either a genuine two-population split, or a continuum
    where the residual/sky width grows smoothly with ``a`` -- so *no*
    threshold produces a clean std jump, the way a partway-through-a-run or
    poorly-localised shell doesn't."""
    rng = np.random.default_rng(seed)
    if gap:
        n_p = n // 2
        a = np.concatenate([rng.uniform(0.1, 0.2, n_p), rng.uniform(0.4, 0.9, n - n_p)])
        theta = np.concatenate([rng.normal(0, 0.01, n_p), rng.normal(0, 0.2, n - n_p)])
        sky = np.concatenate([rng.normal(0, 0.6, n_p), rng.normal(0, 0.05, n - n_p)])
    else:
        a = rng.uniform(0.1, 0.9, n)
        theta = rng.normal(0, 0.05 + 0.05 * (a - 0.1) / 0.8)
        sky = rng.normal(0, 0.3 - 0.15 * (a - 0.1) / 0.8)
    return a, theta, sky


def test_robust_std_matches_gaussian_std_roughly():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2.5, 20000)
    assert robust_std(x) == pytest.approx(2.5, rel=0.05)


def test_robust_std_too_few_points():
    assert np.isnan(robust_std(np.array([1.0])))


def test_heteroscedasticity_ratio_detects_scale_mismatch():
    a, theta, _ = synthetic()
    ratio = heteroscedasticity_ratio(a, theta, threshold=0.3, min_count=100)
    assert ratio > 5  # true std ratio is 0.2 / 0.01 = 20


def test_heteroscedasticity_ratio_nan_below_min_count():
    a, theta, _ = synthetic()
    assert np.isnan(heteroscedasticity_ratio(a, theta, threshold=0.3, min_count=10**6))


def test_scan_recovers_the_true_gap():
    a, theta, sky = synthetic()
    scan = scan_split_thresholds(a, theta, sky, n_grid=100, min_count=100)
    best = scan["threshold"][np.nanargmax(scan["score"])]
    assert 0.2 <= best <= 0.4  # the true gap is [0.2, 0.4)


def test_best_split_accepts_a_real_split():
    a, theta, sky = synthetic(gap=True)
    decision = best_polarisation_split(a, theta, sky, min_score=3.0)
    assert decision.split
    assert 0.2 <= decision.threshold <= 0.4
    assert decision.score > 3.0


def test_best_split_rejects_a_continuum():
    a, theta, sky = synthetic(gap=False)
    decision = best_polarisation_split(a, theta, sky, min_score=3.0)
    assert not decision.split


def test_flow_kwargs_needs_a_split():
    a, theta, sky = synthetic(gap=False)
    decision = best_polarisation_split(a, theta, sky, min_score=3.0)
    with pytest.raises(ValueError):
        polarisation_branch_flow_kwargs(decision)


def test_flow_kwargs_shape():
    a, theta, sky = synthetic(gap=True)
    decision = best_polarisation_split(a, theta, sky, min_score=3.0)
    kwargs = polarisation_branch_flow_kwargs(decision, reference_time=1234.0)
    assert set(kwargs) == {"plateau", "clump"}
    assert kwargs["plateau"]["phase_coordinates"] == "independent"
    assert kwargs["clump"]["phase_coordinates"] == "arg-alpha-beta"
    assert kwargs["plateau"]["reference_time"] == 1234.0


def test_flow_kwargs_forces_plateau_to_physical_space():
    a, theta, sky = synthetic(gap=True)
    decision = best_polarisation_split(a, theta, sky, min_score=3.0)
    kwargs = polarisation_branch_flow_kwargs(decision, prime_space=True)
    assert kwargs["plateau"]["prime_space"] is False
    assert kwargs["clump"]["prime_space"] is True  # untouched


def test_flow_kwargs_default_polarisation_offset_is_zero():
    a, theta, sky = synthetic(gap=True)
    decision = best_polarisation_split(a, theta, sky, min_score=3.0)
    kwargs = polarisation_branch_flow_kwargs(decision)
    assert kwargs["plateau"]["polarisation_offset"] == 0.0
    assert kwargs["clump"]["polarisation_offset"] == 0.0


def test_flow_kwargs_fits_clump_offset_when_psi_given():
    a, theta, sky = synthetic(gap=True)
    decision = best_polarisation_split(a, theta, sky, min_score=3.0)
    rng = np.random.default_rng(4)
    # a psi population straddling the pi/2 seam
    clump_psi = np.mod(rng.normal(0.5 * np.pi, 0.1, 2000), np.pi)
    kwargs = polarisation_branch_flow_kwargs(decision, clump_psi=clump_psi)
    assert kwargs["plateau"]["polarisation_offset"] == 0.0
    assert kwargs["clump"]["polarisation_offset"] != 0.0
    # re-folding with the fitted offset should recentre it near pi/4
    from nessai_gw.group_mixture import recommended_polarisation_offset
    shifted = np.mod(clump_psi - kwargs["clump"]["polarisation_offset"], np.pi)
    d = recommended_polarisation_offset(shifted)
    assert d["circ_mean_lambda"] == pytest.approx(np.pi / 4, abs=0.05)


# ---------------------------------------------------------------------------
# find_density_gap / best_gap_split -- the recommended entry point
# ---------------------------------------------------------------------------
def test_find_density_gap_finds_the_true_gap():
    a, _, _ = synthetic(gap=True)
    gap = find_density_gap(a)
    assert gap is not None
    assert 0.2 <= gap <= 0.4


def test_find_density_gap_none_on_a_continuum():
    a, _, _ = synthetic(gap=False)
    assert find_density_gap(a) is None


def test_find_density_gap_ignores_a_thin_tail():
    """A gap with almost nothing past it (a near-empty tail) should not count."""
    rng = np.random.default_rng(3)
    a = np.concatenate([rng.uniform(0.1, 0.5, 3990), rng.uniform(0.9, 0.95, 10)])
    assert find_density_gap(a, min_fraction=0.15) is None


def test_best_gap_split_accepts_a_real_split():
    a, theta, sky = synthetic(gap=True)
    decision = best_gap_split(a, theta, sky)
    assert decision.split
    assert 0.2 <= decision.threshold <= 0.4


def test_best_gap_split_rejects_a_continuum():
    a, theta, sky = synthetic(gap=False)
    decision = best_gap_split(a, theta, sky)
    assert not decision.split


# ---------------------------------------------------------------------------
# best_changepoint_split
# ---------------------------------------------------------------------------
def test_local_scale_profile_shape():
    a, theta, _ = synthetic(gap=True)
    centers, log_std = local_scale_profile(a, theta, n_bins=20, min_bin=30)
    assert len(centers) == len(log_std)
    assert len(centers) > 0
    assert np.all(np.diff(centers) > 0)  # bin centres are ordered


def test_changepoint_split_accepts_a_real_split():
    a, theta, sky = synthetic(gap=True)
    decision = best_changepoint_split(a, theta, sky, min_improvement=0.5)
    assert decision.split
    assert 0.15 <= decision.threshold <= 0.45


# ---------------------------------------------------------------------------
# fit_distance_floor_split / diagonal_split_mask
# ---------------------------------------------------------------------------
def synthetic_wedge(n=40000, seed=0, quarter_turn=False):
    """A toy deterministic sky sample with a quarter-circle distance-floor
    curve from ``(a, v) = (0.15, 0.2)`` to ``(0.6, 0.6)`` (arc centred on
    ``(0.15, 0.6)``), ``d_L^\\ast`` = 25 exactly on the curve and rising with
    distance from it -- the same qualitative shape as the real v70 map."""
    rng = np.random.default_rng(seed)
    centre = np.array([0.15, 0.6])
    radius = 0.4
    theta = rng.uniform(0.0, 0.5 * np.pi, n)
    r = radius + rng.normal(0, 0.15, n)
    r = np.clip(r, 0.01, None)
    a = centre[0] + r * np.sin(theta)
    v = centre[1] - r * np.cos(theta)
    dl_star = 25.0 * (r / radius)  # = 25 exactly on the arc, grows outward
    if quarter_turn:
        a, v = v, a
    return a, v, dl_star


def test_fit_distance_floor_split_recovers_the_midpoint():
    a, v, dl_star = synthetic_wedge()
    split = fit_distance_floor_split(a, v, dl_star, floor=25.0, tol=1.0)
    # arc-length midpoint of a quarter circle is at 45 degrees
    expected = np.array([0.15 + 0.4 * np.sin(np.pi / 4),
                         0.6 - 0.4 * np.cos(np.pi / 4)])
    assert np.allclose(split.midpoint, expected, atol=0.03)


def test_fit_distance_floor_split_raises_when_the_floor_barely_bites():
    a, v, dl_star = synthetic_wedge()
    dl_star = dl_star + 1000.0  # nothing anywhere near the 25 Mpc floor
    with pytest.raises(ValueError):
        fit_distance_floor_split(a, v, dl_star, floor=25.0, tol=1.0)


def test_diagonal_split_separates_a_synthetic_two_population_sheet():
    a_det, v_det, dl_star = synthetic_wedge()
    split = fit_distance_floor_split(a_det, v_det, dl_star, floor=25.0, tol=1.0)

    rng = np.random.default_rng(7)
    n = 4000
    # plateau: low a, broad v; clump: high a, tight v -- straddling the arc
    plateau = np.stack([rng.normal(0.15, 0.02, n), rng.uniform(0.2, 0.6, n)], axis=1)
    clump = np.stack([rng.normal(0.6, 0.05, n), rng.normal(0.6, 0.02, n)], axis=1)
    mask_c = diagonal_split_mask(clump[:, 0], clump[:, 1], split)
    mask_p = diagonal_split_mask(plateau[:, 0], plateau[:, 1], split)
    assert mask_c.mean() > 0.9
    assert mask_p.mean() < 0.1


def test_diagonal_split_degenerate_when_curve_is_out_of_range():
    """A curve fit far from where the live points sit puts everyone on one
    side (the v96 failure mode) -- silent, not an error."""
    a_det, v_det, dl_star = synthetic_wedge()
    split = fit_distance_floor_split(a_det, v_det, dl_star, floor=25.0, tol=1.0)
    rng = np.random.default_rng(9)
    far_a = rng.normal(0.9, 0.02, 2000)
    far_v = rng.normal(0.05, 0.02, 2000)
    mask = diagonal_split_mask(far_a, far_v, split)
    assert mask.all() or (~mask).all()


# ---------------------------------------------------------------------------
# fit_phase_rotation / FittedPhaseRotation
# ---------------------------------------------------------------------------
def correlated_psi_phase(n=6000, seed=0, true_angle=0.6, psi0=0.7, phase0=1.3,
                         ratio=6.0):
    """Synthetic (psi, phase) with a known decorrelating angle: draw in a
    frame rotated by -true_angle from (psi, phase), so PCA should recover
    +true_angle (mod pi)."""
    rng = np.random.default_rng(seed)
    p1 = rng.normal(0, 1.0, n)
    p2 = rng.normal(0, 1.0 / ratio, n)
    c, s = np.cos(true_angle), np.sin(true_angle)
    # inverse of FittedPhaseRotation's forward map
    dphase = p1 * c + p2 * s
    dpsi = -p1 * s + p2 * c
    return dpsi + psi0, dphase + phase0


def test_fit_phase_rotation_decorrelates():
    psi, phase = correlated_psi_phase()
    assert abs(np.corrcoef(psi, phase)[0, 1]) > 0.3  # genuinely correlated first
    fit = fit_phase_rotation(psi, phase)
    reparam = FittedPhaseRotation(
        parameters=["psi", "phase"], angle=fit.angle, psi0=fit.psi0,
        phase0=fit.phase0,
    )
    x = np.zeros(len(psi), dtype=[("psi", "f8"), ("phase", "f8")])
    x["psi"], x["phase"] = psi, phase
    xp = np.zeros(len(psi), dtype=[("phase_rot_1", "f8"), ("phase_rot_2", "f8")])
    _, xp, _ = reparam.reparameterise(x, xp, np.zeros(len(psi)))
    assert abs(np.corrcoef(xp["phase_rot_1"], xp["phase_rot_2"])[0, 1]) < 1e-8


def test_fit_phase_rotation_eigenvalues_ordered_descending():
    psi, phase = correlated_psi_phase(ratio=6.0)
    fit = fit_phase_rotation(psi, phase)
    assert fit.eigenvalues[0] > fit.eigenvalues[1]
    # true variance ratio was 6**2 = 36
    assert (fit.eigenvalues[0] / fit.eigenvalues[1]) == pytest.approx(36.0, rel=0.15)


def test_fit_phase_rotation_recovers_the_centre():
    psi, phase = correlated_psi_phase(psi0=0.42, phase0=2.1)
    fit = fit_phase_rotation(psi, phase)
    assert fit.psi0 == pytest.approx(0.42, abs=0.05)
    assert fit.phase0 == pytest.approx(2.1, abs=0.05)
