"""Clustering strategies for the ET-Delta triangle's plateau/clump split,
plugged into nessai's clustered group-mixture flow.

Both are ``flow_model_factory`` callables for
:func:`nessai_gw.group_mixture.make_et_group_flow_proposal` (with
``n_clusters_max=2``), e.g.::

    make_et_group_flow_proposal(
        names, reference_time, n_clusters_max=2,
        flow_model_factory=functools.partial(
            make_diagonal_split_cluster_flow, activate_min_round=3))

Two alternatives to the free-GMM clustering in ``nessai.flowmodel.group_mixture``
live here:

* :func:`make_diagonal_split_cluster_flow` -- a straight-line split in the
  canonical ``(cos theta_jn, sky_v)`` plane, re-fit from the live points every
  training round (:func:`fit_live_diagonal_split`) and switched on/off by how
  much a two-expert representation improves on one
  (:func:`gaussian_split_gain`).
* :func:`make_constrained_group_flow` -- a tuned fixed threshold on the
  folded ``theta_jn_prime`` coordinate, validated on the finished v70 run
  (see ``ConstrainedTwoModeWrapper``'s docstring below).

Background
----------
A single co-sited triangle is exactly degenerate over the whole sky for the
dominant (2, 2) mode; the folded posterior is a thin sheet whose local scale
in sky latitude and inclination differs sharply between the plateau regime
(broad sky, tight inclination) and the clump regime near the eight
circularly-polarised directions (compact sky, broad inclination).

``nessai.flowmodel.group_mixture.ClusteredGroupMixtureFlowWrapper`` already
implements everything a mixture-of-experts base flow needs -- per-expert
neural flows, per-expert canonical standardisation, closed-form mixture
weights, importance-weight-correct ``log_prob``/``sample_and_log_prob``,
warm-starting, hysteresis across training rounds -- driven by two overridable
hooks, ``_k_want`` (how many clusters this round) and ``_labels_for_k``
(which cluster each point belongs to).  The default hooks fit a GMM; both
strategies below replace them with a deterministic split instead.

One thing neither strategy attempts: the plateau and clump branches also want
different **phase coordinates** (plateau: plain, independent ``psi``/
``phase``; clump: ``arg_alpha``/``arg_beta`` with a fitted
``polarisation_offset`` -- see :mod:`nessai_gw._polarisation_branch`'s module
docstring).  The clustered mixture shares one ``param_names``/reparameterisation
set across every expert (only the per-expert flow weights and canonical
standardisation differ), so that per-branch difference cannot be expressed
here; both experts see the same coordinates, and the per-expert flows are
relied on to absorb the rest.  A future two-proposal architecture (see the
session notes) would be needed to also vary the coordinates per branch.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from nessai.flowmodel.group_mixture import (
    ClusteredGroupMixtureFlowModel,
    ClusteredGroupMixtureFlowWrapper,
    make_clustered_group_mixture_flow,
    make_group_mixture_flow,
)
from nessai.flows import reset_permutations, reset_weights

# A nessai child logger, so these lines land in the run's BNS.log.
logger = logging.getLogger("nessai.nessai_gw.clustering")

__all__ = [
    "DiagonalSplitClusterWrapper",
    "fit_live_diagonal_split",
    "gaussian_split_gain",
    "make_diagonal_split_cluster_flow",
    "ConstrainedTwoModeWrapper",
    "ConstrainedClusteredModel",
    "make_constrained_group_flow",
]

#: Default minimum fraction of the training points on each side of the line
#: for the split to activate (and for a candidate line to be considered).
MIN_BRANCH_FRACTION = 0.05


def split_coordinates(canon, param_names):
    """``(cos theta_jn, sky_v)`` from canonical prime coordinates.

    ``theta_jn_prime`` is the fixed-bounds ``[0, pi] -> [-1, 1]`` rescaling
    (the same convention :class:`nessai_gw.group_mixture.PrimeSpaceTriangularGroupAction`
    relies on), so ``cos theta_jn = -sin(pi * theta_jn_prime / 2)``.
    ``sky_v`` is the folded equal-area latitude in ``[0, 0.5)`` (0 at the
    detector zenith, 0.5 in the detector plane).
    """
    canon = np.asarray(canon, dtype=float)
    p = canon[:, param_names.index("theta_jn_prime")]
    return -np.sin(0.5 * np.pi * p), canon[:, param_names.index("sky_v")]


def _half_logdet(cov):
    sign, ld = np.linalg.slogdet(cov)
    return 0.5 * ld if sign > 0 else np.inf


def gaussian_split_gain(t, labels, min_count=2):
    """Per-point log-likelihood gain (nats) of fitting each side of a binary
    split with its own Gaussian, over one Gaussian for everything.

    ``0.5 logdet(S) - sum_j w_j [0.5 logdet(S_j) - log w_j]`` with ML
    covariances in the base-flow frame ``t``.  A closed-form proxy for how
    much two separately-standardised experts beat a single flow: it is large
    exactly when the two sides have very different scales/orientations
    (heteroscedasticity), not merely different means.  Returns ``-inf`` if
    either side has fewer than ``min_count`` points.
    """
    t = np.asarray(t, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n = t.shape[0]
    out = _half_logdet(np.cov(t, rowvar=False, bias=True))
    for m in (labels, ~labels):
        nj = int(m.sum())
        if nj < max(min_count, t.shape[1] + 1):
            return -np.inf
        w = nj / n
        out -= w * (
            _half_logdet(np.cov(t[m], rowvar=False, bias=True)) - np.log(w)
        )
    return float(out)


def fit_live_diagonal_split(
    c,
    v,
    t,
    n_angles=36,
    n_offsets=48,
    min_fraction=MIN_BRANCH_FRACTION,
    min_count=100,
):
    """Best straight line in the ``(c, v) = (cos theta_jn, sky_v)`` plane by
    :func:`gaussian_split_gain` in the base-flow frame ``t``.

    Scans ``n_angles`` orientations (in the standardised ``(c, v)`` plane)
    times ``n_offsets`` positions between the ``min_fraction`` and
    ``1 - min_fraction`` quantiles, using cumulative first/second moments so
    each orientation costs one sort.

    Returns
    -------
    dict or None
        ``normal`` (2,), ``offset`` (label 1 is ``normal . (c, v) > offset``,
        oriented so label 1 is the higher-``sky_v`` side), ``gain``,
        ``fraction`` (of label 1), ``angle`` (deg, standardised plane);
        ``None`` if no admissible line exists.
    """
    c = np.asarray(c, dtype=float)
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    n, d = t.shape
    lo = max(int(np.ceil(min_fraction * n)), int(min_count), d + 1)
    if 2 * lo > n:
        return None
    c0, v0 = c.mean(), v.mean()
    cs, vs = max(c.std(), 1e-12), max(v.std(), 1e-12)
    cz, vz = (c - c0) / cs, (v - v0) / vs
    tc = t - t.mean(axis=0)
    half_ld_all = _half_logdet(tc.T @ tc / n)
    splits = np.unique(np.linspace(lo, n - lo, n_offsets).round().astype(int))

    best = None
    for phi in np.linspace(0.0, np.pi, n_angles, endpoint=False):
        s = cz * np.cos(phi) + vz * np.sin(phi)
        order = np.argsort(s, kind="stable")
        x = tc[order]
        s1 = np.cumsum(x, axis=0)
        s2 = np.cumsum(x[:, :, None] * x[:, None, :], axis=0)
        for m in splits:
            n0, n1 = m, n - m
            mu0 = s1[m - 1] / n0
            mu1 = (s1[-1] - s1[m - 1]) / n1
            c0v = s2[m - 1] / n0 - np.outer(mu0, mu0)
            c1v = (s2[-1] - s2[m - 1]) / n1 - np.outer(mu1, mu1)
            w0, w1 = n0 / n, n1 / n
            g = (
                half_ld_all
                - w0 * (_half_logdet(c0v) - np.log(w0))
                - w1 * (_half_logdet(c1v) - np.log(w1))
            )
            if np.isfinite(g) and (best is None or g > best[0]):
                thr = 0.5 * (s[order[m - 1]] + s[order[m]])
                best = (float(g), float(phi), float(thr))
    if best is None:
        return None
    g, phi, thr = best
    normal = np.array([np.cos(phi) / cs, np.sin(phi) / vs])
    offset = thr + normal @ np.array([c0, v0])
    lab = c * normal[0] + v * normal[1] > offset
    if v[lab].mean() < v[~lab].mean():
        normal, offset, lab = -normal, -offset, ~lab
    return dict(
        normal=normal,
        offset=float(offset),
        gain=g,
        fraction=float(lab.mean()),
        angle=float(np.degrees(phi)),
    )


class DiagonalSplitClusterWrapper(ClusteredGroupMixtureFlowWrapper):
    """:class:`ClusteredGroupMixtureFlowWrapper` split by a live-fitted line
    in the canonical ``(cos theta_jn, sky_v)`` plane instead of a GMM.

    Every clustering round (:meth:`_k_want`) re-fits the line to the current
    training points with :func:`fit_live_diagonal_split` and scores it with
    :func:`gaussian_split_gain`.  The split activates once the gain reaches
    :attr:`gain_on` with at least :attr:`activate_fraction` of the points on
    each side, and collapses once the gain drops below :attr:`gain_off` or a
    side falls below :attr:`min_branch_fraction_floor`; the base class's
    ``k_grow_patience``/``k_shrink_patience`` add the round-to-round
    persistence.  While inactive the line simply tracks the best fit; while
    active it is only replaced when a new fit beats the current line's gain
    by :attr:`refit_margin`, so the experts' populations do not reshuffle
    every round.  The line lives in registered buffers, so it survives
    ``--resume`` and ``--reset-flow`` and ``populate()`` routes by exactly the
    line the experts were trained on.

    Tunables are class attributes, set by :func:`make_diagonal_split_cluster_flow`.
    """

    #: Gain (nats per training point) needed to activate the split.
    gain_on = 0.5
    #: Once active, the split collapses when the gain falls below this.
    gain_off = 0.2
    #: A new line replaces the active one only if it beats it by this much.
    refit_margin = 0.05
    #: Minimum fraction of points on each side to activate.
    activate_fraction = MIN_BRANCH_FRACTION
    #: Once active, the split collapses when either side falls below this.
    min_branch_fraction_floor = 0.01
    #: Minimum number of clustering rounds before the split may activate
    #: (a coarse config-time guard; 0 disables it).
    activate_min_round = 0
    #: Resolution of the line scan in :func:`fit_live_diagonal_split`.
    n_angles = 36
    n_offsets = 48

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dtype = torch.get_default_dtype()
        self.register_buffer(
            "_split_normal", torch.zeros(2, dtype=torch.float64)
        )
        self.register_buffer(
            "_split_offset", torch.zeros((), dtype=torch.float64)
        )
        self.register_buffer(
            "_split_fitted", torch.zeros((), dtype=torch.bool)
        )
        self.register_buffer(
            "_split_gain", torch.full((), -np.inf, dtype=dtype)
        )

    def _carry_over_group_state(self, old):
        super()._carry_over_group_state(old)
        if isinstance(old, DiagonalSplitClusterWrapper):
            with torch.no_grad():
                self._split_normal.copy_(old._split_normal)
                self._split_offset.copy_(old._split_offset)
                self._split_fitted.copy_(old._split_fitted)
                self._split_gain.copy_(old._split_gain)

    #: How far an expert's live population may shrink (relative to the
    #: largest it has been since its flow weights were last (re)initialised)
    #: before those weights get reset. ``None`` disables this entirely
    #: (the old behaviour: warm-start forever, never reset).
    #:
    #: MLE flow training only ever pulls density *towards* the current
    #: training points -- nothing in the objective pushes density *away*
    #: from a region once no data lands there. Every expert starts from a
    #: shared, broad ancestor (at a *first* split, ``experts[1]`` is a
    #: literal copy of the pre-split single flow, which was itself trained
    #: on the *entire*, much wider, live population up to that point -- see
    #: :meth:`_cluster`'s warm-start block, unchanged here). Continued
    #: per-round fine-tuning on an ever-shrinking, ever-narrower routed
    #: subset (as the branch localises) never gets a training signal to
    #: unlearn that inherited breadth, so it can persist for tens of
    #: thousands of iterations as free-standing, catastrophically-low-
    #: likelihood density that ``populate()`` keeps sampling and rejecting
    #: (see the ET-Delta v40split investigation: expert 0 shrank from
    #: ~5000 live points at first activation to ~200, and a from-scratch
    #: (non-warm-started) flow trained on just its current ~200 points
    #: showed none of the stale-density lobe the live, warm-started
    #: version did).
    #:
    #: Checked every clustering round, per active expert, against a
    #: running high-water mark of that expert's own population size since
    #: its last reset (:meth:`_maybe_reset_shrunk_experts`); tripping the
    #: factor reinitialises that expert's flow weights (and its canonical
    #: standardisation) so its next few training rounds rebuild density
    #: from only the live points it actually owns now, instead of carrying
    #: forward a wide ancestor. A *growing* population never triggers this
    #: -- growth just means more data for the flow to (correctly) spread
    #: over, not stale unmatched density.
    reset_shrink_factor = 2.0

    def _cluster(self, x):
        labels = super()._cluster(x)
        self._maybe_reset_shrunk_experts(labels)
        return labels

    def _maybe_reset_shrunk_experts(self, labels):
        """Reset any active expert whose live population has shrunk more
        than :attr:`reset_shrink_factor` since its own high-water mark (see
        that attribute's docstring for why). ``labels`` is this round's
        per-row expert assignment, as returned by :meth:`_cluster`.
        """
        if self.reset_shrink_factor is None:
            return
        k = int(self._n_active.item())
        if k < 1:
            return
        ref = getattr(self, "_expert_ref_size", None)
        if ref is None or len(ref) != self.n_experts:
            # Not a buffer: a resume restarts high-water tracking, which
            # only delays detecting a shrink that happened within the gap.
            ref = np.zeros(self.n_experts)
            self._expert_ref_size = ref

        counts = np.bincount(
            labels.detach().cpu().numpy(), minlength=self.n_experts
        )
        for j in range(k):
            n_j = float(counts[j])
            if n_j <= 0:
                continue
            if ref[j] <= 0:
                ref[j] = n_j
                continue
            ref[j] = max(ref[j], n_j)
            if n_j * self.reset_shrink_factor < ref[j]:
                expert = self.experts[j]
                expert.apply(reset_weights)
                expert.apply(reset_permutations)
                expert._canon_seen.zero_()
                expert._domain_mass_seen = False
                logger.info(
                    "DiagonalSplitClusterWrapper: expert %d population "
                    "shrank %.0f -> %.0f (>%gx since its last reset) -- "
                    "resetting its flow weights so it rebuilds density "
                    "from its current live points instead of carrying "
                    "forward a wider ancestor",
                    j,
                    ref[j],
                    n_j,
                    self.reset_shrink_factor,
                )
                ref[j] = n_j

    def _split_coords(self, t):
        """Base-flow-frame ``t`` -> ``(cos theta_jn, sky_v)`` numpy arrays."""
        e = self.experts[0]
        t = torch.as_tensor(np.asarray(t), dtype=self._base_mu.dtype)
        with torch.no_grad():
            canon, _ = e._from_base(t)
        return split_coordinates(canon.cpu().numpy(), list(e.param_names))

    def _split_labels(self, c, v):
        """1 on the (higher-``sky_v``) clump side of the stored line, else 0."""
        normal = self._split_normal.cpu().numpy()
        return (
            c * normal[0] + v * normal[1] > float(self._split_offset)
        ).astype(int)

    def _store_split(self, fit):
        with torch.no_grad():
            self._split_normal.copy_(torch.as_tensor(fit["normal"]))
            self._split_offset.fill_(fit["offset"])
            self._split_fitted.fill_(True)

    def _k_want(self, t, ts, mu, sd):
        """2 while a two-expert split pays for itself, else 1 (see the class
        docstring)."""
        if self.n_experts < 2:
            return 1
        self._cluster_round_count = (
            getattr(self, "_cluster_round_count", 0) + 1
        )
        active = (
            bool(self._clustering_seen) and int(self._n_active.item()) >= 2
        )

        c, v = self._split_coords(t)
        fit = fit_live_diagonal_split(
            c,
            v,
            t,
            n_angles=self.n_angles,
            n_offsets=self.n_offsets,
            min_fraction=self.activate_fraction,
        )
        cur_gain = -np.inf
        if bool(self._split_fitted):
            cur_gain = gaussian_split_gain(t, self._split_labels(c, v))
        if fit is not None and (
            not active or fit["gain"] > cur_gain + self.refit_margin
        ):
            self._store_split(fit)
            cur_gain = fit["gain"]
        if not bool(self._split_fitted):
            return 1

        frac = float(self._split_labels(c, v).mean())
        self._split_gain.fill_(cur_gain)
        normal = self._split_normal.cpu().numpy()
        logger.info(
            "Diagonal split (cos theta_jn, sky_v): gain %.3f nat/pt "
            "(on %.2f / off %.2f), clump-side fraction %.3f, line "
            "%.3f*cos + %.3f*sky_v > %.3f, %s",
            cur_gain,
            self.gain_on,
            self.gain_off,
            frac,
            normal[0],
            normal[1],
            float(self._split_offset),
            "active" if active else "inactive",
        )
        f_min = min(frac, 1.0 - frac)
        if active:
            keep = (
                cur_gain >= self.gain_off
                and f_min >= self.min_branch_fraction_floor
            )
            return 2 if keep else 1
        if self._cluster_round_count < self.activate_min_round:
            return 1
        return (
            2
            if (cur_gain >= self.gain_on and f_min >= self.activate_fraction)
            else 1
        )

    def _labels_for_k(self, t, ts, mu, sd, k, k_cur, prev_raw):
        if k < 2:
            labels = np.zeros(ts.shape[0], dtype=int)
            return labels, ts.mean(0, keepdims=True)
        labels = self._split_labels(*self._split_coords(t))
        centroids = np.stack(
            [
                ts[labels == j].mean(axis=0)
                if np.any(labels == j)
                else ts.mean(axis=0)
                for j in range(2)
            ]
        )
        return labels, centroids

    @torch.no_grad()
    def _route(self, x):
        """Route by the stored line (not nearest-centroid), so training,
        clustering and ``populate()`` all agree on which expert owns a
        point."""
        act = int(self._n_active.item())
        if act <= 1 or not bool(self._clustering_seen):
            return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        t = self._fold_to_base(x).detach().cpu().numpy()
        labels = self._split_labels(*self._split_coords(t))
        return torch.as_tensor(labels, dtype=torch.long, device=x.device)


def make_diagonal_split_cluster_flow(
    gain_on=None,
    gain_off=None,
    activate_fraction=None,
    activate_min_round=0,
    reset_shrink_factor=None,
    **kwargs,
):
    """:func:`~nessai.flowmodel.group_mixture.make_clustered_group_mixture_flow`
    with :class:`DiagonalSplitClusterWrapper` as the clustering strategy.

    Parameters
    ----------
    gain_on, gain_off : float, optional
        Activation / collapse thresholds on :func:`gaussian_split_gain`
        (nats per point).  ``None`` keeps the class defaults (0.5 / 0.2).
    activate_fraction : float, optional
        Minimum fraction of points on each side to activate (and to consider
        a candidate line).  ``None`` keeps :data:`MIN_BRANCH_FRACTION`.
    activate_min_round : int, optional
        Clustering rounds before the split may activate at all (0: off).
    reset_shrink_factor : float or "off", optional
        See :attr:`DiagonalSplitClusterWrapper.reset_shrink_factor`.
    **kwargs
        Forwarded to
        :func:`~nessai.flowmodel.group_mixture.make_clustered_group_mixture_flow`.
    """
    kwargs.setdefault("n_clusters_max", 2)
    base_cls = make_clustered_group_mixture_flow(**kwargs)

    class DiagonalSplitCustom(base_cls):
        pass

    attrs = {"activate_min_round": int(activate_min_round)}
    if gain_on is not None:
        attrs["gain_on"] = float(gain_on)
    if gain_off is not None:
        attrs["gain_off"] = float(gain_off)
    if activate_fraction is not None:
        attrs["activate_fraction"] = float(activate_fraction)
    if reset_shrink_factor is not None:
        attrs["reset_shrink_factor"] = (
            None
            if reset_shrink_factor == "off"
            else float(reset_shrink_factor)
        )
    DiagonalSplitCustom.wrapper_cls = type(
        "BoundDiagonalSplitClusterWrapper",
        (DiagonalSplitClusterWrapper,),
        attrs,
    )
    return DiagonalSplitCustom


class ConstrainedTwoModeWrapper(ClusteredGroupMixtureFlowWrapper):
    """``k=2`` wrapper whose split is a fixed threshold on one folded prime
    coordinate instead of a GMM.  ``n_experts`` must be 2.

    A *tuned, not general* alternative to the free-GMM clustering: the free
    GMM at ``k=2`` splits the two compact localised inclination sub-modes
    into the two experts and orphans the diffuse **edge-on / broad-sky**
    mode -- neither compact-sheet-shaped expert proposes there, so it starves
    and the run drops a co-equal posterior peak (v26 / v27; see the
    ``v27-dropped-edge-on-mode`` note).  ``k=3`` works but uses a third
    expert.

    This wrapper pins the two clusters to the *known* regions the two modes
    occupy:

      * expert 0 -- "localised": face-on OR face-off inclination, tight sky
      * expert 1 -- "broad": edge-on inclination, sky opens up

    Routing is a fixed threshold on the folded ``theta_jn_prime`` coordinate
    (``angle-sine`` folds face-on <-> face-off, so edge-on is the coordinate's
    extreme near 0).  Threshold ``-0.30`` was picked on the finished v70 run
    (``compare_runs_at_compression`` / the threshold sweep): completeness 1.00
    (no edge-on point ever mis-routed), purity 0.999, and the broad cluster
    never drops below ~330 live points across the whole ``-lnX`` window where
    the mode exists, then empties cleanly when it dies.

    Overrides the strategy hooks added to
    :class:`ClusteredGroupMixtureFlowWrapper` (``_k_want`` / ``_labels_for_k``)
    and ``_route``; every other piece -- the expert warm-start on activation,
    ``_pending_split_train``, the per-cluster training loss, the background
    expert -- is inherited unchanged.
    """

    #: folded base-frame coordinate the rule reads
    split_coord = "theta_jn_prime"
    #: ``t[:, i] > split_threshold`` -> expert 1 ("broad" / edge-on)
    split_threshold = -0.30
    #: both sides must clear this many live points to *activate* the split
    split_min_size = 100
    #: once active (``sticky``), the split only collapses when a side drops
    #: below this (i.e. the mode has genuinely died)
    split_empty_floor = 10
    sticky = True

    # -- geometry --------------------------------------------------------
    def _split_index(self):
        return list(self.param_names).index(self.split_coord)

    def _broad_mask(self, t):
        """Boolean mask (numpy): rows that belong to expert 1."""
        return np.asarray(t)[:, self._split_index()] > self.split_threshold

    # -- strategy hooks -------------------------------------------------
    def _k_want(self, t, ts, mu, sd):
        broad = self._broad_mask(t)
        n1 = int(broad.sum())
        n0 = int(broad.size - n1)
        active = (
            self.sticky
            and bool(self._clustering_seen)
            and int(self._n_active.item()) == 2
        )
        floor = self.split_empty_floor if active else self.split_min_size
        return 2 if (n0 >= floor and n1 >= floor) else 1

    def _labels_for_k(self, t, ts, mu, sd, k, k_cur, prev_raw):
        if k < 2:
            return np.zeros(ts.shape[0], dtype=int), ts.mean(0, keepdims=True)
        labels = self._broad_mask(t).astype(int)  # 0 localised, 1 broad
        cent = np.stack(
            [
                ts[labels == c].mean(0) if np.any(labels == c) else ts.mean(0)
                for c in range(2)
            ]
        )
        # EMA the stored routing centroids for continuity (labels are
        # rule-based, so this only smooths diagnostics / the _route fallback).
        if (
            prev_raw is not None
            and k == k_cur
            and prev_raw.shape[0] == 2
            and 0.0 <= self.centroid_ema < 1.0
        ):
            b = self.centroid_ema
            cent = ((1.0 - b) * prev_raw + b * (cent * sd + mu) - mu) / sd
        return labels, cent

    @torch.no_grad()
    def _route(self, x):
        """Route training points by the same fixed rule (not nearest-centroid),
        so the per-cluster training loss and the clustering agree exactly."""
        act = int(self._n_active.item())
        if act <= 1 or not bool(self._clustering_seen):
            return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        t = self._fold_to_base(x)
        i = self._split_index()
        return (t[:, i] > self.split_threshold).long()


class ConstrainedClusteredModel(ClusteredGroupMixtureFlowModel):
    """:class:`ClusteredGroupMixtureFlowModel` that builds a
    :class:`ConstrainedTwoModeWrapper` (fixed ``n_clusters_max = 2``)."""

    n_clusters_max = 2
    wrapper_cls = ConstrainedTwoModeWrapper


def make_constrained_group_flow(
    *,
    split_coord="theta_jn_prime",
    split_threshold=-0.30,
    split_min_size=100,
    split_empty_floor=10,
    sticky=True,
    bg_weight=0.2,
    k_shrink_patience=3,
    k_grow_patience=1,
    centroid_ema=None,
    **kwargs,
):
    """A clustered group-mixture ``FlowModel`` class with the constrained
    two-mode split.  ``**kwargs`` are forwarded to
    :func:`nessai.flowmodel.group_mixture.make_group_mixture_flow` (the group
    action, ``param_names``, ``canonical_transform``, ``mode_factor_sizes``,
    ...), exactly as :func:`make_clustered_group_mixture_flow` forwards them --
    so this drops in as a ``flow_model_factory`` for
    :func:`nessai_gw.group_mixture.make_et_group_flow_proposal`.
    """
    base_cls = make_group_mixture_flow(**kwargs)

    class _Wrapper(ConstrainedTwoModeWrapper):
        pass

    _Wrapper.split_coord = str(split_coord)
    _Wrapper.split_threshold = float(split_threshold)
    _Wrapper.split_min_size = int(split_min_size)
    _Wrapper.split_empty_floor = int(split_empty_floor)
    _Wrapper.sticky = bool(sticky)

    class ConstrainedCustom(ConstrainedClusteredModel, base_cls):
        pass

    ConstrainedCustom.n_clusters_max = 2
    ConstrainedCustom.wrapper_cls = _Wrapper
    ConstrainedCustom.bg_weight = float(bg_weight)
    ConstrainedCustom.min_cluster_size = int(split_min_size)
    ConstrainedCustom.k_shrink_patience = int(k_shrink_patience)
    ConstrainedCustom.k_grow_patience = int(k_grow_patience)
    ConstrainedCustom.centroid_ema = (
        None if centroid_ema is None else float(centroid_ema)
    )
    return ConstrainedCustom
