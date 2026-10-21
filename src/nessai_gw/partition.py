"""Split coordinates and splitters for nessai's partitioned nested sampling
(``NestedSampler(partition=...)``, :mod:`nessai.partition`) in the GW sky.

The multimodalities the partition is meant for come from known places:

* two L-shaped detectors: the ring of sky positions around the baseline,
  whose azimuth is ``sky_u`` in the baseline frame -- one periodic
  coordinate, cut at its empty gaps (:class:`nessai.partition.GapSplitter`);
* the ET triangle: the plateau and clump branches in the canonical
  ``(cos theta_jn, sky_v)`` plane, cut by the straight line of the diagonal
  split (:class:`nessai_gw.clustering.DiagonalSplitClusterWrapper`), frozen
  once it fires: :class:`DiagonalSplitTrigger` takes the line and the moment
  from the proposal's own diagonal split, :class:`LineSplitter` refits them
  from the live points.

:class:`SkyCoordinates` computes them from the physical parameters with the
proposal's sky frame (:class:`nessai_gw.reparameterisations.sky.EqualAreaSky`)
copied at the split, so a region stays a fixed set of parameters whatever
the proposal does to its frame later.
"""

from __future__ import annotations

import logging

import numpy as np
from nessai.partition import LinearCut, SplitCoordinates

from .clustering import (
    MIN_BRANCH_FRACTION,
    fit_live_diagonal_split,
    gaussian_frame,
)

logger = logging.getLogger("nessai.nessai_gw.partition")

__all__ = [
    "SkyCoordinates",
    "LineSplitter",
    "DiagonalSplitTrigger",
    "base_frame",
]

_COORDINATES = ("sky_u", "sky_v", "cos_theta_jn")


def _sky_rotation(proposal):
    """The rotation of the proposal's ``EqualAreaSky`` (equatorial -> sky
    frame), or None."""
    reparams = getattr(proposal, "_reparameterisation", None)
    if reparams is None:
        return None
    for rep in reparams.values():
        R = getattr(rep, "_R", None)
        if R is not None and getattr(rep, "prime_parameters", None) == [
            "sky_u",
            "sky_v",
        ]:
            return np.array(R, dtype=float)
    return None


class SkyCoordinates(SplitCoordinates):
    """``sky_u``, ``sky_v`` and ``cos_theta_jn`` as split coordinates.

    ``sky_u`` (the azimuth in the sky frame, in ``[0, 1)``, periodic) and
    ``sky_v`` (``(1 - z) / 2``, ``z`` along the frame's pole) are those of
    the proposal's :class:`~nessai_gw.reparameterisations.sky.EqualAreaSky`,
    with its rotation copied by :meth:`freeze`.

    Parameters
    ----------
    names : list of str
        Some of ``"sky_u"``, ``"sky_v"``, ``"cos_theta_jn"``.
    fold : {None, "plane"}
        ``"plane"``: the triangle's plane reflection ``(sky_v, cos theta_jn)
        -> (1 - sky_v, -cos theta_jn)`` applied where ``sky_v >= 1/2``, so
        ``sky_v`` is the folded latitude in ``[0, 1/2)`` of the diagonal
        split. A region is then a union of the images of the fold.
    rotation : array_like, optional
        The frame rotation; set by :meth:`freeze`.
    """

    def __init__(self, names=("sky_u",), fold=None, rotation=None):
        names = tuple(names)
        unknown = set(names) - set(_COORDINATES)
        if unknown:
            raise ValueError(f"Unknown sky coordinates {unknown}")
        if fold not in (None, "plane"):
            raise ValueError(f"Unknown fold {fold!r}")
        self.names = names
        self.fold = fold
        self.rotation = None if rotation is None else np.array(rotation)

    @property
    def periods(self):
        return [1.0 if n == "sky_u" else None for n in self.names]

    def configure(self, model):
        needed = {"ra", "dec"}
        if "cos_theta_jn" in self.names or self.fold:
            needed.add("theta_jn")
        missing = needed - set(model.names)
        if missing:
            raise ValueError(f"SkyCoordinates needs the parameters {missing}")

    def freeze(self, proposal=None):
        if proposal is None:
            if self.rotation is None:
                raise RuntimeError("SkyCoordinates needs the proposal's frame")
            return self
        R = _sky_rotation(proposal)
        if R is None:
            raise RuntimeError(
                "SkyCoordinates: the proposal has no EqualAreaSky frame"
            )
        return SkyCoordinates(self.names, self.fold, rotation=R)

    def __call__(self, points):
        if self.rotation is None:
            raise RuntimeError("SkyCoordinates is not frozen")
        ra = np.asarray(points["ra"], dtype=float)
        dec = np.asarray(points["dec"], dtype=float)
        cd = np.cos(dec)
        w = self.rotation @ np.stack(
            [cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=0
        )
        out = {
            "sky_u": np.mod(np.arctan2(w[1], w[0]), 2 * np.pi) / (2 * np.pi),
            "sky_v": 0.5 * (1.0 - np.clip(w[2], -1.0, 1.0)),
        }
        if "cos_theta_jn" in self.names or self.fold:
            out["cos_theta_jn"] = np.cos(
                np.asarray(points["theta_jn"], dtype=float)
            )
        if self.fold == "plane":
            reflected = out["sky_v"] >= 0.5
            out["sky_v"] = np.where(
                reflected, 1.0 - out["sky_v"], out["sky_v"]
            )
            out["cos_theta_jn"] = np.where(
                reflected, -out["cos_theta_jn"], out["cos_theta_jn"]
            )
        return np.column_stack([out[n] for n in self.names]).reshape(
            len(points), len(self.names)
        )

    def __repr__(self):
        return f"SkyCoordinates({list(self.names)}, fold={self.fold!r})"


def base_frame(proposal, points):
    """The group-mixture base-flow frame ``t`` of ``points`` (the canonical
    image, through the canonical transform), as the diagonal split routes
    by, with the frame's parameter names and circular columns; None if the
    proposal has no group-mixture flow."""
    import torch

    flow = getattr(getattr(proposal, "flow", None), "model", None)
    if flow is None or not hasattr(proposal, "_training_data_as_prime_tensor"):
        return None
    experts = getattr(flow, "experts", None)
    e = experts[0] if experts else flow
    if not hasattr(e, "_assign_branch") or not hasattr(e, "_to_base"):
        return None
    n = len(points)
    x = proposal._training_data_as_prime_tensor(points)[:n]
    with torch.no_grad():
        assigned, pre, _ = e._assign_branch(x)
        canon = pre[assigned, torch.arange(n, device=x.device)]
        t, _ = e._to_base(canon)
    return (
        t.cpu().numpy().astype(float),
        list(getattr(e, "param_names", [])),
        getattr(e, "_circular_idx", None),
    )


class LineSplitter:
    """The diagonal split as a partition: the straight line in a plane of
    two split coordinates (by default ``(cos theta_jn, sky_v)`` folded,
    :class:`SkyCoordinates` ``fold="plane"``) that most improves two
    Gaussians on its sides over one, in the group-mixture base frame
    (:func:`nessai_gw.clustering.fit_live_diagonal_split`,
    :func:`nessai_gw.clustering.gaussian_split_gain`). It is proposed once
    the gain reaches ``gain_on`` with at least ``activate_fraction`` of the
    points on either side -- the diagonal split's activation. Unlike the
    diagonal split, the partition freezes the line: it never moves or
    collapses afterwards.

    Parameters
    ----------
    gain_on : float
        Gain (nats per point) to split.
    activate_fraction : float
        Minimum fraction of the points on each side.
    min_count : int
        Minimum number of points on each side.
    rank_gaussian_params : tuple of str
        Base-frame columns rank-Gaussianised before the gain.
    n_angles, n_offsets : int
        Resolution of the line scan.
    """

    def __init__(
        self,
        gain_on=0.6,
        activate_fraction=MIN_BRANCH_FRACTION,
        min_count=100,
        rank_gaussian_params=("chirp_distance",),
        n_angles=36,
        n_offsets=48,
    ):
        self.gain_on = float(gain_on)
        self.activate_fraction = float(activate_fraction)
        self.min_count = int(min_count)
        self.rank_gaussian_params = tuple(rank_gaussian_params)
        self.n_angles = int(n_angles)
        self.n_offsets = int(n_offsets)

    def _frame(self, coords, points, proposal):
        frame = None
        if points is not None and proposal is not None:
            try:
                frame = base_frame(proposal, points)
            except Exception as e:  # fall back to the plane itself
                logger.debug(f"LineSplitter: no base frame ({e})")
        if frame is None:
            # the plane itself, standardised (rank-Gaussianising each
            # coordinate would hide the difference in shape the gain
            # measures)
            sd = coords.std(axis=0)
            return (coords - coords.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        t, names, circular = frame
        return gaussian_frame(t, names, circular, self.rank_gaussian_params)

    def propose(
        self,
        coords,
        periods,
        points=None,
        proposal=None,
        max_labels=None,
        force=False,
    ):
        coords = np.asarray(coords, dtype=float)
        if coords.shape[1] != 2:
            raise ValueError("LineSplitter needs two split coordinates")
        if max_labels is not None and max_labels < 2:
            return None
        t = self._frame(coords, points, proposal)
        fit = fit_live_diagonal_split(
            coords[:, 0],
            coords[:, 1],
            t,
            n_angles=self.n_angles,
            n_offsets=self.n_offsets,
            min_fraction=self.activate_fraction,
            min_count=self.min_count,
        )
        if fit is None:
            return None
        f_min = min(fit["fraction"], 1.0 - fit["fraction"])
        if not force and (
            fit["gain"] < self.gain_on or f_min < self.activate_fraction
        ):
            return None
        cut = LinearCut(fit["normal"], fit["offset"])
        cut.info = dict(
            gain=float(fit["gain"]),
            fraction=float(fit["fraction"]),
            angle=float(fit["angle"]),
            gain_on=self.gain_on,
        )
        return cut


class DiagonalSplitTrigger:
    """The proposal's own diagonal split as a partition: the cut is the line
    of its :class:`~nessai_gw.clustering.DiagonalSplitClusterWrapper`, used
    once the split is active and its two experts have trained -- the same
    activation (gain, side fractions, patience) and line as a run without
    the partition, so the regions are the experts' two sides. The coordinates
    are :class:`SkyCoordinates` ``(cos theta_jn, sky_v)`` with
    ``fold="plane"``, the canonical plane the line lives in; the cut is only
    used if it labels at least ``min_agreement`` of the points as the
    mixture routes them. Paired with ``max_regions=2``: the split is made
    once, and each region's copy of the proposal keeps the expert of its
    side (:meth:`nessai.flowmodel.group_mixture.GroupFlowProposalMixin.restrict_to_region`)
    and its own diagonal split, which may split that flow again.

    Parameters
    ----------
    min_agreement : float
        Minimum fraction of the points the cut and the mixture's routing
        put on the same side.
    """

    def __init__(self, min_agreement=0.99):
        self.min_agreement = float(min_agreement)

    @staticmethod
    def _wrapper(proposal):
        from .clustering import DiagonalSplitClusterWrapper

        model = getattr(getattr(proposal, "flow", None), "model", None)
        if not isinstance(model, DiagonalSplitClusterWrapper):
            return None
        return model

    def propose(
        self,
        coords,
        periods,
        points=None,
        proposal=None,
        max_labels=None,
        force=False,
    ):
        import torch

        coords = np.asarray(coords, dtype=float)
        if coords.shape[1] != 2:
            raise ValueError("DiagonalSplitTrigger needs two split coordinates")
        if max_labels is not None and max_labels < 2:
            return None
        w = self._wrapper(proposal)
        if w is None:
            raise RuntimeError(
                "DiagonalSplitTrigger needs a proposal with the diagonal split"
            )
        if not (
            bool(w._clustering_seen)
            and int(w._n_active.item()) >= 2
            and bool(w._split_fitted)
            and not bool(w._pending_split_train.item())
        ):
            return None
        cut = LinearCut(
            w._split_normal.cpu().numpy().copy(), float(w._split_offset)
        )
        labels = cut.classify(coords)
        # the training tensor's leading rows are the points (any augmented
        # copies follow), as in base_frame
        x = proposal._training_data_as_prime_tensor(points)[: len(points)]
        with torch.no_grad():
            routed = w._route(x).cpu().numpy()
        agreement = float(np.mean(labels == routed))
        cut.info = dict(
            gain=float(w._split_gain),
            fraction=float(labels.mean()),
            agreement=agreement,
        )
        if agreement < self.min_agreement:
            logger.warning(
                f"DiagonalSplitTrigger: the line {cut.describe()} agrees "
                f"with the mixture's routing at {agreement:.4f} of the live "
                f"points (< {self.min_agreement}): not splitting"
            )
            return None
        return cut
