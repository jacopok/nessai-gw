"""Split the triangular-detector degeneracy sheet into a plateau/clump pair.

Motivation
----------
A single co-sited triangle is exactly degenerate over the whole sky for the
dominant (2, 2) mode (see :mod:`nessai_gw._ellipse`): fixing a sky direction
``n`` forces ``cos iota*(n)``, and the folded posterior is a thin sheet
winding across the sphere.  ``a = |cos iota*(n)|`` splits that sheet into two
regimes with very different local curvature:

* **plateau** (``a`` small, near the run's plateau value ``c_pl``): a broad
  patch of sky maps to nearly the same inclination -- diffuse sky, tight
  inclination.
* **clump** (``a`` large, near the eight circularly-polarised directions):
  a tiny patch of sky spans the whole range down to the plateau -- compact
  sky, broad inclination.

Trying to fit both regimes with one flow and one fixed coordinate map is what
makes the sheet look bimodal/heteroscedastic (see the note,
``triangle_degeneracy/triangle_degeneracy.tex`` Sec. 4).  This module finds
whether, for the live points actually in hand, splitting on a threshold in
``a`` buys anything -- rather than assuming it always does, or detecting only
whether a gap exists in the ``a`` histogram (a gap can be transient: it opens
partway through a run and closes again once the plateau population empties
out, and never opens at all for a poorly-localised run).  The criterion here
is direct: does the split make the two populations differ enough in scale
(on the axes that actually suffer -- the inclination residual and the sky
latitude) that giving them one flow each, instead of one flow for both, is
worth the extra machinery.

Three ways to answer that, in increasing order of how well they held up
against actual corner plots of the result (see ``triangle_degeneracy``'s
``corner_hetero_cases.py`` / ``corner_gap_*.pdf``):

* :func:`heteroscedasticity_ratio` / :func:`scan_split_thresholds` /
  :func:`best_polarisation_split` -- maximise the std ratio of the two raw
  halves directly.  Looked sound (a real ET-Delta shell clears a chosen
  ``min_score`` by 5-25x, and the fully converged posterior falls below it),
  but the threshold it finds does **not** visually match two coherent
  populations: it carves the single tightest sliver of points off the tail,
  because a global std ratio does not care whether either side is internally
  homogeneous.  Kept for the diagnostic value of the ratio itself, not
  recommended for choosing the threshold.
* :func:`best_changepoint_split` -- fit a kink (two lines vs. one) in
  ``log(local std))`` vs. ``a`` instead.  Better, but still over-triggers: on
  a shell with a single genuine population (``-lnX=55``, well past the
  point the plateau empties out) the inclination residual's width saturates
  near ``a=1`` for reasons unrelated to population structure (a measurement
  floor, not a second regime), and that alone is enough of a kink to pass a
  lenient threshold.
* :func:`find_density_gap` / :func:`best_gap_split` -- require an actual
  empty bin in the ``a`` histogram, with a healthy population on each side,
  and use the ratio only to confirm the gap is worth exploiting rather than
  to search for it.  This reproduces the visually clean plateau/clump split
  on the shell it is known to hold for, and correctly declines everywhere it
  does not -- but it turns out to be *too* conservative: real cases with two
  differently-scaled populations (``-lnX=35``, well before the shell is
  fully localised) have no empty bin yet, because the two populations are
  still touching, not gapped.
* :func:`fit_distance_floor_split` / :func:`diagonal_split_mask` -- use the
  actual geometry instead of reading it off a live-point histogram.  In the
  ``(a, sky_lat)`` plane the populated region is bounded by a curve traced by
  the 25 Mpc distance-prior floor (``d_L^\ast(n) = 25``) -- see the note's
  Sec. 4.2 -- running from the plateau corner out towards the high-``a``
  clump, where a smooth increase in ``d_L^\ast`` (hence volumetric prior
  weight) makes the clump population matter even where it is not gapped from
  the plateau.  The split orthogonal to this curve's arc-length midpoint,
  computed **once** from the run's deterministic sky map (not from any one
  live-point shell), correctly separates plateau from clump on every v70
  shell tested (``-lnX`` 35, 45) with no tunable threshold at all: shells
  where the plateau has already been excluded put *zero* points on the
  plateau side, which is its own "don't split" signal.  **Use this one when
  the deterministic map is available** (it needs an auxiliary sky sample with
  ``d_L^\ast(n)`` computed, e.g. ``triangle_degeneracy/bimodality.py``'s
  ``ellipse_locus``); fall back to :func:`best_gap_split` otherwise.  Only
  validated so far on a run (v70) where the floor actually bites a long
  stretch of the sheet -- on a run whose injected distance is far above the
  floor (v96, 600 Mpc) the floor only clips a small corner, and the fit does
  not fail loudly there: it happily returns a curve fit to that corner, whose
  midpoint then sits nowhere near where the live points actually are, so
  :func:`diagonal_split_mask` puts every point on one side (silently
  degenerate, not an error) rather than raising.  Check for that -- e.g.
  reject the split if either side is implausibly small a priori, not just
  after the fact -- before trusting this on a run you have not looked at.
  That case needs the general valid/invalid boundary instead, not just
  implemented here.
"""
from __future__ import annotations

import numpy as np

from . import nessai_logger

logger = nessai_logger.getChild(__name__)


def robust_std(x: np.ndarray) -> float:
    """Median-absolute-deviation standard-deviation estimate.

    Robust to the heavy tails these residuals have close to a split boundary;
    plain ``std`` is dominated by a handful of stragglers there.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float("nan")
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad)


def heteroscedasticity_ratio(a, values, threshold, min_count=100, min_fraction=0.2):
    """``max/min`` of the robust std of ``values`` on either side of ``threshold``.

    Parameters
    ----------
    a : array_like
        The split indicator, ``|cos iota*(n)|``, one entry per live point.
    values : array_like
        The quantity whose scale you want to compare across the split (the
        inclination residual, or the sky-latitude coordinate).
    threshold : float
        Candidate cut: ``lo = a < threshold``, ``hi = a >= threshold``.
    min_count : int
        Return ``nan`` if either side has fewer than this many points -- the
        std estimate is too noisy to trust below that.
    min_fraction : float
        Return ``nan`` if either side has less than this fraction of all the
        points.  Needed on top of ``min_count``: with a smooth (non-gapped)
        trend in ``values`` vs. ``a``, a threshold near the edge of the range
        can still pass ``min_count`` while carving off a thin, extreme-``a``
        sliver whose width is unrepresentative of that whole side -- an
        artefactually large ratio that has nothing to do with there being two
        real populations.  Requiring a healthy fraction on each side rules
        that out.

    Returns
    -------
    float
        ``nan`` if either side is too small or either std is (numerically)
        zero; otherwise a number ``>= 1``, larger when the split matters more.
    """
    a = np.asarray(a, dtype=float)
    values = np.asarray(values, dtype=float)
    lo, hi = a < threshold, a >= threshold
    n_min = max(min_count, min_fraction * a.size)
    if lo.sum() < n_min or hi.sum() < n_min:
        return float("nan")
    s_lo, s_hi = robust_std(values[lo]), robust_std(values[hi])
    if not (s_lo > 0 and s_hi > 0):
        return float("nan")
    return max(s_lo, s_hi) / min(s_lo, s_hi)


def scan_split_thresholds(a, theta_resid, sky_lat, n_grid=200, min_count=100,
                          min_fraction=0.2, lo=None, hi=None):
    """Heteroscedasticity ratio vs. candidate threshold, on a grid.

    Parameters
    ----------
    a, theta_resid, sky_lat : array_like
        ``|cos iota*(n)|``, the plain ellipse-residual coordinate
        (:class:`~nessai_gw.reparameterisations.inclination.PolarisationEllipseReparameterisation`
        with ``adaptive_width=False``), and the sky-latitude coordinate
        (``sky_v`` of :class:`~nessai_gw.reparameterisations.sky.EqualAreaSky`),
        one entry per live point.
    n_grid : int
        Number of candidate thresholds.
    min_count, min_fraction : as in :func:`heteroscedasticity_ratio`.
    lo, hi : float, optional
        Range to scan; default ``(a.min(), a.max())``.

    Returns
    -------
    dict with keys ``threshold``, ``ratio_theta``, ``ratio_sky``, ``score``
        (the geometric mean of the two ratios, ``nan`` where either is
        ``nan``), each an ``(n_grid,)`` array.
    """
    a = np.asarray(a, dtype=float)
    lo_ = float(a.min()) if lo is None else lo
    hi_ = float(a.max()) if hi is None else hi
    thresholds = np.linspace(lo_, hi_, n_grid + 2)[1:-1]
    ratio_theta = np.array([
        heteroscedasticity_ratio(a, theta_resid, t, min_count, min_fraction)
        for t in thresholds
    ])
    ratio_sky = np.array([
        heteroscedasticity_ratio(a, sky_lat, t, min_count, min_fraction)
        for t in thresholds
    ])
    score = np.sqrt(ratio_theta * ratio_sky)
    return dict(threshold=thresholds, ratio_theta=ratio_theta,
               ratio_sky=ratio_sky, score=score)


def local_scale_profile(a, values, n_bins=20, min_bin=30):
    """Robust std of ``values`` in quantile bins of ``a``.

    Parameters
    ----------
    a, values : array_like
    n_bins : int
        Number of quantile bins to try (bins with fewer than ``min_bin``
        points are dropped, so the returned arrays can be shorter).
    min_bin : int
        Minimum points for a bin's std to be trusted.

    Returns
    -------
    centers, log_std : ndarray
        Per-bin median ``a`` and ``log`` robust std, only for bins with a
        finite, positive std.  Not necessarily the same length as ``n_bins``.
    """
    a = np.asarray(a, dtype=float)
    values = np.asarray(values, dtype=float)
    edges = np.quantile(a, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(a, edges) - 1, 0, n_bins - 1)
    centers, log_std = [], []
    for i in range(n_bins):
        m = idx == i
        if int(m.sum()) < min_bin:
            continue
        s = robust_std(values[m])
        if s > 0:
            centers.append(float(np.median(a[m])))
            log_std.append(float(np.log(s)))
    return np.asarray(centers), np.asarray(log_std)


def _line_sse(x, y):
    """Residual sum of squares of the best-fit line through ``(x, y)``."""
    if len(x) < 2:
        return 0.0
    if len(x) == 2 or np.ptp(x) == 0:
        return float(np.sum((y - y.mean()) ** 2))
    coef = np.polyfit(x, y, 1)
    resid = y - np.polyval(coef, x)
    return float(np.sum(resid ** 2))


def best_changepoint_split(a, theta_resid, sky_lat, n_bins=20, min_bin=30,
                           min_improvement=0.6, min_segment=3):
    r"""Find the ``a`` where the local scale genuinely *kinks*, vs. where it
    is just a smooth, single monotone trend.

    Two failure modes this avoids:

    * :func:`heteroscedasticity_ratio` (the ``score`` in
      :func:`best_polarisation_split`) compares the std of the two raw
      halves, which is maximised by carving off the single tightest sliver of
      points -- even when the remaining "other side" is itself a mix of two
      different scales.  Visually (see ``triangle_degeneracy``'s
      ``corner_hetero_cases.py``) this reliably picks a threshold in the
      *middle* of the clump's own continuum, not the plateau/clump boundary.
    * A piecewise-*constant* changepoint test (flat segment vs. flat segment)
      does better, but still over-triggers: the clump-only regime is a smooth
      monotone trend in ``log(local std))`` vs. ``a``, and *any* monotone
      trend is better fit by two flat segments than by one, so a constant-fit
      test finds a "changepoint" even in runs with a single, genuinely
      homogeneous-up-to-a-gradient population (the fully converged v70
      posterior, or the late-run ``-lnX=55`` shell).

    This compares one straight *line* through ``log(local std)`` vs. ``a``
    (the natural model for a smooth single regime, where the scale shrinks
    roughly as a power law all the way to the circular points) against two
    independent lines, one per side of a candidate changepoint.  A genuine
    plateau/clump boundary is a *kink*, not just a slope change on a
    monotone curve -- the plateau side is close to flat while the clump side
    keeps climbing -- so the two-line fit wins by a large margin there and
    only marginally on a single smooth regime.

    Parameters
    ----------
    a, theta_resid, sky_lat : array_like
    n_bins, min_bin : as in :func:`local_scale_profile`.
    min_improvement : float
        Minimum fractional reduction in total sum-of-squares,
        ``(sse1 - sse2) / sse1``, to accept the split.  Higher than
        :func:`best_polarisation_split`'s analogous threshold because a
        smooth single-line regime already explains most of the variance;
        what is left to explain is exactly the kink a real split should find.
    min_segment : int
        Minimum bins per side (need >= 2 to fit a line; 3 gives it one
        residual degree of freedom).

    Returns
    -------
    SplitDecision
        ``score`` is the fractional SSE improvement (not the ratio
        :func:`best_polarisation_split` reports).
    """
    ca, lt = local_scale_profile(a, theta_resid, n_bins, min_bin)
    _, ls = local_scale_profile(a, sky_lat, n_bins, min_bin)
    n = min(len(ca), len(ls))
    if n < 2 * min_segment:
        return SplitDecision(False, None, float("nan"), float("nan"), float("nan"))
    ca, lt, ls = ca[:n], lt[:n], ls[:n]

    sse1 = _line_sse(ca, lt) + _line_sse(ca, ls)
    best_k, best_sse2 = None, None
    for k in range(min_segment, n - min_segment + 1):
        s2 = (_line_sse(ca[:k], lt[:k]) + _line_sse(ca[k:], lt[k:])
              + _line_sse(ca[:k], ls[:k]) + _line_sse(ca[k:], ls[k:]))
        if best_sse2 is None or s2 < best_sse2:
            best_sse2, best_k = s2, k
    if best_k is None or sse1 <= 0:
        return SplitDecision(False, None, float("nan"), float("nan"), float("nan"))
    improvement = (sse1 - best_sse2) / sse1
    threshold = 0.5 * (ca[best_k - 1] + ca[best_k])
    ratio_theta = float(np.exp(lt[best_k:].mean() - lt[:best_k].mean()))
    ratio_sky = float(np.exp(ls[:best_k].mean() - ls[best_k:].mean()))
    if not (improvement >= min_improvement):
        return SplitDecision(False, threshold, improvement,
                             abs(ratio_theta), abs(ratio_sky))
    return SplitDecision(True, threshold, improvement,
                         abs(ratio_theta), abs(ratio_sky))


def find_density_gap(a, n_bins=80, min_fraction=0.15):
    """A real low-density valley in the ``a`` histogram, if there is one.

    Unlike :func:`best_polarisation_split` and :func:`best_changepoint_split`
    (both of which search for wherever the *scale* of ``theta_resid`` /
    ``sky_lat`` changes most, and both of which turn out to have a failure
    mode -- see their docstrings and ``triangle_degeneracy``'s
    ``corner_hetero_cases.py`` for the visual evidence), this looks for the
    much more basic and much harder to fool signature of two populations:
    an actual absence of live points at some ``a``, with a healthy fraction
    of the points on either side.  On the one shell where a hard split is
    known (by direct inspection) to produce two visually coherent, unimodal
    populations -- ET-Delta v70 at ``-lnX=45`` -- this is also where the gap
    sits (``a`` in ``[0.25, 0.32)``, no points at all); the shells that
    should *not* split (the converged v70 posterior, and ``-lnX=55``, where
    the plateau population has already been excluded by the likelihood) have
    no such gap.  Recommended as the split trigger; use
    :func:`heteroscedasticity_ratio` at the returned threshold only as a
    secondary sanity check on the scale mismatch it buys, not to search for
    the threshold itself.

    Parameters
    ----------
    a : array_like
    n_bins : int
        Resolution of the histogram the gap is read off.
    min_fraction : float
        A candidate gap only counts if at least this fraction of the points
        sit on each side of it -- rules out the near-empty tail past the
        last handful of points at ``a`` close to 1, which is a gap by the
        letter of the definition but not a second population.

    Returns
    -------
    float or None
        The midpoint of the first (lowest-``a``) qualifying gap, or ``None``.
    """
    a = np.asarray(a, dtype=float)
    counts, edges = np.histogram(a, bins=n_bins, range=(a.min(), a.max()))
    n_min = min_fraction * a.size
    i = 1
    while i < len(counts) - 1:
        if counts[i] == 0 and counts[i - 1] > 0:
            j = i
            while j < len(counts) and counts[j] == 0:
                j += 1
            lo_count = counts[:i].sum()
            hi_count = counts[j:].sum() if j < len(counts) else 0
            if lo_count >= n_min and hi_count >= n_min:
                return 0.5 * (edges[i] + edges[j])
            i = j
        else:
            i += 1
    return None


def best_gap_split(a, theta_resid, sky_lat, n_bins=80, min_fraction=0.15,
                   min_count=100, min_score=2.0):
    """:func:`find_density_gap` gated by a minimum heteroscedasticity ratio.

    The recommended entry point: split only where there is an actual gap in
    the ``a`` histogram (see :func:`find_density_gap`) *and* that gap buys a
    real scale difference (``score >= min_score``, evaluated at the gap, not
    searched for).  ``min_score`` is deliberately lower than
    :func:`best_polarisation_split`'s -- the gap has already done the work of
    ruling out spurious edge slivers and within-population kinks, so this is
    just confirming the gap is not, say, splitting off two populations that
    happen to have coincidentally similar scale (in which case a split buys
    engineering complexity for no benefit).

    Returns
    -------
    SplitDecision, or None if :func:`find_density_gap` finds no candidate.
    """
    threshold = find_density_gap(a, n_bins=n_bins, min_fraction=min_fraction)
    if threshold is None:
        return SplitDecision(False, None, float("nan"), float("nan"), float("nan"))
    ratio_theta = heteroscedasticity_ratio(a, theta_resid, threshold, min_count, 0.0)
    ratio_sky = heteroscedasticity_ratio(a, sky_lat, threshold, min_count, 0.0)
    if not (np.isfinite(ratio_theta) and np.isfinite(ratio_sky)):
        return SplitDecision(False, threshold, float("nan"), ratio_theta, ratio_sky)
    score = float(np.sqrt(ratio_theta * ratio_sky))
    return SplitDecision(score >= min_score, threshold, score, ratio_theta, ratio_sky)


class DiagonalSplit:
    """A line in the ``(a, sky_lat)`` plane: point + direction.

    ``midpoint`` and ``tangent`` are given in the *original* curve's own
    sense (``tangent`` along the fitted boundary curve, not along the split
    line itself -- the split line is orthogonal to it, through ``midpoint``).
    """

    def __init__(self, midpoint, tangent):
        self.midpoint = np.asarray(midpoint, dtype=float)
        t = np.asarray(tangent, dtype=float)
        self.tangent = t / np.linalg.norm(t)

    @property
    def normal(self):
        """Direction of the split line itself (orthogonal to ``tangent``)."""
        return np.array([-self.tangent[1], self.tangent[0]])

    def __repr__(self):
        return (f"DiagonalSplit(midpoint=({self.midpoint[0]:.3f}, "
               f"{self.midpoint[1]:.3f}), tangent=({self.tangent[0]:.3f}, "
               f"{self.tangent[1]:.3f}))")


def fit_distance_floor_split(a, sky_lat, dl_star, floor=25.0, tol=1.0,
                             n_bins=25, min_bin=5):
    r"""Fit the ``(a, sky_lat)`` boundary traced by the distance-prior floor.

    Parameters
    ----------
    a, sky_lat, dl_star : array_like
        ``|cos iota*(n)|``, the sky-latitude coordinate and ``d_L^\ast(n)``
        (see :mod:`nessai_gw._ellipse` / the note's Sec. 2-3) from an
        auxiliary deterministic sky sample -- *not* live points.  Typically a
        few times ``10^4`` directions drawn uniformly on the sphere, folded
        into the same canonical ``(a, sky_lat)`` representation as the live
        points (e.g. ``triangle_degeneracy/bimodality.py``'s
        ``ellipse_locus``).  Needs enough directions that ``tol`` catches a
        few thousand of them near the floor; too coarse a sample gives a
        noisy curve fit or ``ValueError``.
    floor, tol : float
        Select directions with ``|d_L^\ast(n) - floor| < tol`` (Mpc).
    n_bins, min_bin : int
        Bin the selected directions by ``a`` into ``n_bins`` bins (dropping
        bins with fewer than ``min_bin`` points) and take the median
        ``sky_lat`` in each -- a clean, monotone curve out of a noisy point
        cloud.

    Returns
    -------
    DiagonalSplit
        The line orthogonal to the fitted curve at its arc-length midpoint.

    Raises
    ------
    ValueError
        If fewer than 4 bins survive -- the floor does not bite enough of
        this run's sheet for a stable fit (see the module docstring: this
        happens when the injected distance is far above the floor).
    """
    a = np.asarray(a, dtype=float)
    sky_lat = np.asarray(sky_lat, dtype=float)
    dl_star = np.asarray(dl_star, dtype=float)
    near = np.abs(dl_star - floor) < tol
    if near.sum() < min_bin:
        raise ValueError(
            f"only {int(near.sum())} directions within {tol} Mpc of the "
            f"{floor} Mpc floor; increase tol or the sample size."
        )
    a_near, v_near = a[near], sky_lat[near]
    edges = np.linspace(a_near.min(), a_near.max(), n_bins + 1)
    idx = np.clip(np.digitize(a_near, edges) - 1, 0, n_bins - 1)
    curve = []
    for i in range(n_bins):
        m = idx == i
        if int(m.sum()) < min_bin:
            continue
        curve.append((float(np.median(a_near[m])), float(np.median(v_near[m]))))
    if len(curve) < 4:
        raise ValueError(
            f"only {len(curve)} bins survived min_bin={min_bin}; the floor "
            "curve is too short/sparse to fit here (see the module "
            "docstring's note on runs where the floor barely bites)."
        )
    curve = np.asarray(curve)
    seg = np.diff(curve, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    mid_s = 0.5 * s[-1]
    i = np.clip(np.searchsorted(s, mid_s) - 1, 0, len(curve) - 2)
    frac = 0.0 if s[i + 1] == s[i] else (mid_s - s[i]) / (s[i + 1] - s[i])
    midpoint = curve[i] + frac * (curve[i + 1] - curve[i])
    tangent = curve[i + 1] - curve[i]
    return DiagonalSplit(midpoint, tangent)


def diagonal_split_mask(a, sky_lat, split: DiagonalSplit):
    """Boolean mask: which side of ``split`` each ``(a, sky_lat)`` point is on.

    ``True`` is the side the curve's fitted tangent points towards (the
    higher-``a`` / clump side, for a curve fit with
    :func:`fit_distance_floor_split` -- the curve runs from low ``a``/low
    ``sky_lat`` towards high ``a``/high ``sky_lat``, so its tangent already
    points clump-ward).  A shell where the plateau population has been fully
    excluded by the likelihood puts every point on this side (mask all
    ``True``) -- read as "nothing to split", the same as
    :class:`SplitDecision`'s ``split=False``, just without a separate flag.

    Parameters
    ----------
    a, sky_lat : array_like
        Live-point coordinates (unlike :func:`fit_distance_floor_split`,
        these are the actual points to classify).
    split : DiagonalSplit

    Returns
    -------
    ndarray of bool
    """
    a = np.asarray(a, dtype=float)
    sky_lat = np.asarray(sky_lat, dtype=float)
    signed = ((a - split.midpoint[0]) * split.tangent[0]
             + (sky_lat - split.midpoint[1]) * split.tangent[1])
    return signed >= 0.0


class SplitDecision:
    """Result of :func:`best_polarisation_split`."""

    def __init__(self, split, threshold, score, ratio_theta, ratio_sky):
        self.split = bool(split)
        self.threshold = float(threshold) if threshold is not None else None
        self.score = float(score) if score is not None else None
        self.ratio_theta = ratio_theta
        self.ratio_sky = ratio_sky

    def __repr__(self):
        if not self.split:
            return f"SplitDecision(split=False, best_score={self.score:.2f})"
        return (f"SplitDecision(split=True, threshold={self.threshold:.3f}, "
               f"score={self.score:.2f}, ratio_theta={self.ratio_theta:.2f}, "
               f"ratio_sky={self.ratio_sky:.2f})")


def best_polarisation_split(a, theta_resid, sky_lat, n_grid=200, min_count=100,
                            min_fraction=0.2, min_score=3.0):
    """Scan for the best plateau/clump split, and decide whether to use it.

    The threshold maximising :func:`scan_split_thresholds`'s geometric-mean
    ``score`` is the best *candidate* split; it is only worth using -- two
    flows/canonicalisations instead of one -- if that score clears
    ``min_score``.  Below it, the live points are closer to the continuum
    regime (early in a run, or a poorly-localised run like a low-SNR
    injection) where a split just cuts one homogeneous population in half for
    no benefit; ``in_fundamental_domain``-style hard assignment would then
    only add training-round-to-training-round jitter.

    Parameters
    ----------
    a, theta_resid, sky_lat : array_like
        See :func:`scan_split_thresholds`.
    n_grid, min_count, min_fraction : as there.  ``min_fraction`` matters more
        here than its docstring alone suggests: without it the scan is biased
        towards the edge of the ``a`` range, because the local scale keeps
        shrinking monotonically all the way to the eight exactly-circular
        points, so an unbalanced split that isolates just the tightest sliver
        near ``a = 1`` always beats a more even one on raw ratio -- not the
        two-population structure this function is meant to find.  ``0.2``
        (used here and in :func:`scan_split_thresholds`) requires each side to
        hold at least a fifth of the points, which in practice moves the
        selected threshold back towards the genuine plateau/clump boundary.
    min_score : float
        Minimum acceptable ``score`` (a std ratio) to actually split.  The
        default, 3, is calibrated against the ET-Delta v70/v96 runs (see
        ``triangle_degeneracy``'s ``hetero_threshold.py``): the mid-run v70
        shells clear it comfortably (score 6-25, threshold drifting with
        ``min_fraction`` -- the scale mismatch is not confined to one crisp
        boundary, it keeps growing all the way towards the circular points),
        the fully converged v70 posterior falls clearly below it (score ~2,
        single population, plateau already excluded by the likelihood), and
        the low-SNR v96 posterior sits just above it (score ~5) *despite*
        showing no gap at all in the raw ``a`` histogram -- evidence that the
        variance criterion catches real splits a gap search would miss, not
        only ones a gap search would also find.

    Returns
    -------
    SplitDecision
    """
    scan = scan_split_thresholds(a, theta_resid, sky_lat, n_grid=n_grid,
                                 min_count=min_count, min_fraction=min_fraction)
    score = scan["score"]
    if not np.any(np.isfinite(score)):
        return SplitDecision(False, None, float("nan"), float("nan"), float("nan"))
    i = int(np.nanargmax(score))
    best_score = float(score[i])
    if not (best_score >= min_score):
        return SplitDecision(False, scan["threshold"][i], best_score,
                             scan["ratio_theta"][i], scan["ratio_sky"][i])
    return SplitDecision(True, scan["threshold"][i], best_score,
                         scan["ratio_theta"][i], scan["ratio_sky"][i])


#: Per-branch ``phase_coordinates`` choice for
#: :func:`nessai_gw.group_mixture.make_triangular_group_flow_proposal`.
#: The two branches end up on *different* bases, for related but distinct
#: reasons -- worth spelling out since both took more than one try.
#:
#: **Clump**: raw ``(psi, phase)`` was two parallel ridges -- the un-recentred
#: ``polarisation_quarter`` seam, see
#: :func:`nessai_gw.group_mixture.recommended_polarisation_offset`.  Fed the
#: *offset-corrected* ``psi`` instead, the two ridges merge into one
#: (corr -0.76 -> -0.999 on v70), and once merged, ``arg_alpha``/``arg_beta``
#: is the better basis: it is the physically-motivated pair (the actual
#: circular-polarisation phases), and clump's covariance is by then so
#: anisotropic (eigenvalue ratio ~1300x) that a numerically-fit rotation
#: (:func:`fit_phase_rotation`) buys only a little over the standard
#: 45-degree shear.  So: ``polarisation_offset`` (fit from clump's own
#: ``psi`` -- see :func:`polarisation_branch_flow_kwargs`) *then*
#: ``arg-alpha-beta``.
#:
#: **Plateau**: raw ``(psi, phase)`` is *also* two clusters, but scanning
#: every ``polarisation_offset`` never merges them (max |corr| ~0.73, see the
#: module docstring) -- a different, still-unidentified degeneracy, not this
#: seam.  The discreteness sits almost entirely on ``phase`` (two narrow,
#: separated clusters, e.g. [0.31, 0.47] and [1.89, 2.04] rad on v70
#: ``-lnX=45`` -- a clean gap, nothing in between) while ``psi`` is broad and
#: single-valued (no comparable gap anywhere in its own marginal).
#: ``arg_alpha``/``arg_beta`` mixes ``psi`` into both flow axes and smears
#: that clean one-axis discreteness diagonally across two -- worse, not
#: better, despite giving a visually "tighter" ridge (corr -0.78, actually
#: *more* correlated than plain ``(psi, phase)``'s -0.55).  So: plain,
#: independent ``phase_coordinates="independent"`` --
#: :class:`~nessai_gw.reparameterisations.phase.FittedPhaseRotation` /
#: ``ArgAlphaBetaReparameterisation`` were both tried and both worse here.
#: Needs ``prime_space=False`` on this branch's proposal (see
#: ``phase_coordinates``'s docstring on
#: :func:`~nessai_gw.group_mixture.make_triangular_group_flow_proposal`).
BRANCH_PHASE_COORDINATES = {
    "plateau": "independent",
    "clump": "arg-alpha-beta",
}


class PhaseRotationFit:
    """Result of :func:`fit_phase_rotation`: an angle and a centre."""

    def __init__(self, angle, psi0, phase0, eigenvalues=None):
        self.angle = float(angle)
        self.psi0 = float(psi0)
        self.phase0 = float(phase0)
        self.eigenvalues = eigenvalues

    def __repr__(self):
        deg = np.degrees(self.angle)
        return (f"PhaseRotationFit(angle={self.angle:.4f} rad ({deg:.1f} deg), "
               f"psi0={self.psi0:.4f}, phase0={self.phase0:.4f})")


def fit_phase_rotation(psi, phase):
    r"""The rotation that diagonalises a branch's own ``(psi, phase)`` covariance.

    Neither ``ArgAlphaBetaReparameterisation``'s fixed 45-degree shear nor
    plain ``(psi, phase)`` (a 0-degree rotation) is the right basis for a
    given plateau/clump branch in general -- the correlation they leave
    behind is set by that branch's own local ``c_-/c_+`` ratio (see
    :class:`~nessai_gw.reparameterisations.phase.FittedPhaseRotation`'s
    docstring for the numbers).  This finds the angle that actually
    decorrelates it: centre ``(psi, phase)`` on their means and take the
    eigenvectors of the 2x2 covariance matrix -- the leading eigenvector's
    angle is the rotation :class:`~nessai_gw.reparameterisations.phase.FittedPhaseRotation`
    should use.

    Parameters
    ----------
    psi, phase : array_like
        A single branch's live points (radians, already restricted to that
        branch -- e.g. via :func:`diagonal_split_mask` or
        :func:`best_gap_split`).

    Returns
    -------
    PhaseRotationFit
        ``.angle`` (radians), ``.psi0``, ``.phase0`` -- feed straight into
        :class:`~nessai_gw.reparameterisations.phase.FittedPhaseRotation`.
        ``.eigenvalues`` (descending) says how anisotropic the branch is:
        a large ratio means one direction is much better measured than the
        other (on v70, plateau ~5.7x, clump ~19x), which the flow's own
        per-dimension standardisation handles on top of this rotation.

    Notes
    -----
    This decorrelates the *linear* structure only.  A branch whose
    ``(psi, phase)`` scatter is two parallel, offset ridges rather than one
    blob (a real discrete near-degeneracy, not a basis problem -- see the
    module docstring) will still show two ridges after rotating; the
    rotation just makes each ridge axis-aligned instead of diagonal.  Folding
    that degeneracy, if it turns out to be a genuine, foldable symmetry of
    the group action, is separate work this function does not attempt.
    """
    psi = np.asarray(psi, dtype=float)
    phase = np.asarray(phase, dtype=float)
    psi0, phase0 = float(psi.mean()), float(phase.mean())
    # (dphase, dpsi) order, matching the v = (dphase, dpsi) FittedPhaseRotation
    # rotates: p1 = v . (cos, -sin), p2 = v . (sin, cos).
    x = np.stack([phase - phase0, psi - psi0], axis=1)
    cov = np.cov(x.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    e1_phase, e1_psi = eigvecs[:, 0]
    # want (cos, -sin) proportional to the leading eigenvector (dphase, dpsi).
    angle = float(np.arctan2(-e1_psi, e1_phase))
    return PhaseRotationFit(angle, psi0, phase0, eigenvalues=eigvals)


def polarisation_branch_flow_kwargs(decision: SplitDecision, clump_psi=None,
                                    **common):
    """Build the ``make_triangular_group_flow_proposal`` kwargs for each branch.

    Parameters
    ----------
    decision : SplitDecision
        Must have ``decision.split`` True.
    clump_psi : array_like, optional
        The clump branch's own ``psi`` (already restricted to that branch,
        e.g. via :func:`diagonal_split_mask`).  If given,
        :func:`~nessai_gw.group_mixture.recommended_polarisation_offset` is
        fit to it and passed as that branch's ``polarisation_offset`` --
        needed to merge clump's ``(psi, phase)`` seam before
        ``arg-alpha-beta`` sees it (see the module docstring and
        :data:`BRANCH_PHASE_COORDINATES`).  Without it, ``polarisation_offset``
        defaults to ``0`` for both branches, which is fine for plateau but
        leaves clump's seam unfixed -- pass this whenever you have clump's
        live points in hand, which is always, since you need them to reach
        this function's caller in the first place.
    **common
        Extra kwargs (``reference_time``, ``plane_normal``, ``vertex``, ...)
        forwarded to both branches, with one exception: ``prime_space`` is
        always forced to ``False`` for plateau regardless of what is passed
        here (its ``phase_coordinates="independent"`` requires it -- see
        :data:`BRANCH_PHASE_COORDINATES`); clump keeps whatever ``prime_space``
        ``common`` sets (default ``True``, the fast path -- clump's
        ``arg-alpha-beta`` supports it).

    Returns
    -------
    dict
        ``{"plateau": kwargs, "clump": kwargs}``, each a dict of kwargs for
        :func:`nessai_gw.group_mixture.make_triangular_group_flow_proposal`
        (or :func:`~nessai_gw.group_mixture.make_et_group_flow_proposal`).
        ``phase_coordinates`` is ``"independent"`` for plateau,
        ``"arg-alpha-beta"`` for clump (see :data:`BRANCH_PHASE_COORDINATES`
        for why); ``polarisation_offset`` is ``0`` for plateau, fit from
        ``clump_psi`` for clump if given.  Routing live points to the right
        branch (``a(n) < decision.threshold``) and combining the two
        proposals into the sampler's live-point pool is the caller's job --
        this only decides *whether* to split and *how* each branch should be
        configured once it has its own live points.
    """
    if not decision.split:
        raise ValueError("decision.split is False; use a single flow instead "
                         "of building per-branch kwargs.")
    kwargs = {
        branch: dict(common, phase_coordinates=coords, polarisation_offset=0.0)
        for branch, coords in BRANCH_PHASE_COORDINATES.items()
    }
    kwargs["plateau"]["prime_space"] = False
    if clump_psi is not None:
        from .group_mixture import recommended_polarisation_offset
        rec = recommended_polarisation_offset(clump_psi)
        kwargs["clump"]["polarisation_offset"] = rec["recommended_azimuth_offset"]
    return kwargs
