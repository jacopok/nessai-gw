"""A mixture of two flow proposals in different coordinates, with ordinary
nested-sampling bookkeeping.

The pool is drawn from ``q(x) = (1 - w) q_1(x) + w q_2(x)``: ``q_1`` the
run's own proposal (e.g. the network group proposal), ``q_2`` a second
proposal with its own coordinates (e.g. the ring chart of
:mod:`nessai_gw.sheet_chart`). Both are trained on the same live points and
every draw is weighted with the mixture density, so the support of the pool
is the union of theirs: where one flow under-covers the live points (the
ring chart at face-on, where its prior piles up) the other still proposes.
There is no partition of the prior: one set of live points, one evidence.

With ``split_activate`` the mixture starts at the run's diagonal split
(:class:`nessai_gw.partition.DiagonalSplitTrigger`'s condition and line, as
a partitioned run's split): until then the pools are the run's proposal's
alone; from then on the run's proposal keeps the expert of the face-on side
of the line and trains on that side's points, the secondary (the sheet chart)
trains on the edge-on side's, and the pools are the mixture's. The same
components as the partitioned run's two regions, without the partition.
Once a side holds less than ``split_collapse_share`` of the training points
(the clustered diagonal split's ``min_branch_fraction_floor``, at which its
two experts collapse to one) the mixture collapses for good to the other
side's component, trained on all the points.

Rejection sampling is against the clip-mode weight cap ``c`` (the
``quantile`` of ``log prior - log q`` over the current live points, as the
prior-relative log-proposal threshold): a draw is kept with probability
``min(w, c) / c``, so the pool follows ``min(prior, c q)``. The other
component's density is needed only for the draws that pass a first test
against the bound ``q >= w_k q_k`` of the component that drew them, which
keeps the costly density (the network's) off almost all of the cheap
component's draws; the two-stage test is exact.
"""

from __future__ import annotations

import copy
import datetime
import os

import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.special import logsumexp

from nessai import config as nessai_config
from . import nessai_logger

logger = nessai_logger.getChild(__name__)


def _component_log_q(proposal, x):
    """Pool density of ``proposal`` at the physical points ``x`` (its draws'
    density: summed over the sides of a boundary inversion with
    ``sum_mirror_density``)."""
    n = len(x)
    if not n:
        return np.empty(0)
    if n == 1:
        # some reparameterisations' table lookups fail on a single point
        x = np.concatenate([x, x])
    sides = np.asarray(proposal.mirror_log_q(x), dtype=float)[:, :n]
    if sides.shape[0] > 1 and not getattr(proposal, "sum_mirror_density", False):
        raise RuntimeError(
            f"{type(proposal).__name__}: a mixture component with boundary-inversion "
            "copies needs sum_mirror_density (each draw is otherwise scored with its "
            "own side's density only)"
        )
    log_q = logsumexp(sides, axis=0)
    return np.where(np.isnan(log_q), -np.inf, log_q)


class MixtureProposalMixin:
    """Mixed into the run's proposal class (see :func:`mixture_proposal_class`):
    the instance is component 1 and owns component 2 (``self.secondary``).

    Class attributes set by :func:`mixture_proposal_class`:
    ``secondary_class`` (a proposal class taking ``model`` and keywords),
    ``secondary_kwargs``, ``mixture_weight`` (the secondary's share ``w``;
    ``None`` fits it to the live points after every training, by EM,
    within ``[min_weight, 1 - min_weight]``), ``min_weight``,
    ``mixture_quantile`` (the cap's quantile; ``None`` takes the run's
    log-proposal-threshold quantile, else 0.005), ``reset_secondary`` (reset
    the secondary's flow when the run resets its own; off: warm, as the
    chart runs), ``switch_lnx`` (see below).

    With ``switch_lnx`` (a ln X, negative) there is no mixture: the run's
    proposal draws the pools on its own (its own population, unchanged)
    until the live points' ln X (``-max(it) / nlive``) first reaches
    ``switch_lnx``, the secondary on its own (its own population, with a
    copy of the run's truncation rules) from then on. The secondary is
    trained from ``switch_warmup`` nats before the switch. With
    ``switch_spread`` instead, the switch is when the live points' ln L
    spread (max - min) first falls to ``switch_spread`` (the secondary
    trained from ``2 * switch_spread``): a live set in the posterior bulk,
    whatever the signal's loudness. With ``switch_race`` instead, the two
    race from ln X ``race_start_lnx`` on: every ``race_every``-th pool is the
    secondary's (trained from then on), the others the run's, each timed
    (population with the pool's likelihoods) per pool point above the
    threshold; the switch is when the secondary's cost per new point is
    below the run's latest at ``race_wins`` of its pools in a row.
    """

    secondary_class = None
    secondary_kwargs = {}
    mixture_weight = 0.5
    min_weight = 0.05
    mixture_quantile = None
    reset_secondary = False
    switch_lnx = None
    switch_warmup = 2.0
    switch_spread = None
    switch_race = False
    race_start_lnx = -5.0
    race_every = 4
    race_wins = 2
    #: a race pool of the secondary draws at most this many times (a pool
    #: that has lost by far costs seconds, not minutes)
    race_max_samples = 200_000
    #: after a secondary pool that cost over ``race_backoff`` times the run's
    #: the gap to its next pool doubles (up to 16 times ``race_every``)
    race_backoff = 4.0
    #: the race's pools: dicts of ln X, component, seconds, new points, cost
    race_history = ()
    #: ln X at which the pools switched to the secondary (None: not yet)
    switched_at = None
    #: start the mixture at the run's diagonal split (see the module notes)
    split_activate = False
    #: the secondary's side: the one whose live points' median |cos
    #: theta_jn| is below this (as :class:`nessai_gw.sheet_chart.SheetChartRegions`)
    split_max_abs_cos = 0.5
    #: the split is used only if each side holds this share of the live
    #: points (as the partition's ``n_min = nlive // 5``)
    split_min_share = 0.2
    #: the frozen split: ``(coordinates, cut, label of the run's side)``
    split = None
    #: collapse the split once a side holds less than this share of the
    #: training points (DiagonalSplitClusterWrapper.min_branch_fraction_floor)
    split_collapse_share = 0.01
    #: the component left after the collapse (0: the run's proposal, 1: the
    #: secondary; None: not collapsed)
    split_survivor = None
    #: settings of the run's proposal the secondary shares
    copied = ("drawsize", "accumulate_weights", "check_acceptance",
              "threshold_from_live_points", "clip_population_weights",
              "population_weight_cap_mass")

    # -- set-up -----------------------------------------------------------
    def _build_secondary(self):
        kwargs = {k: getattr(self, k) for k in self.copied if hasattr(self, k)}
        kwargs.update(self.secondary_kwargs)
        kwargs.setdefault("plot", False)
        kwargs.setdefault("poolsize", getattr(self, "_poolsize", 1000))
        kwargs.setdefault("update_poolsize", False)
        self.secondary = self.secondary_class(self.model, **kwargs)
        self.secondary.output = os.path.join(self.output, "mixture_secondary", "")
        self.secondary.rng = np.random.default_rng(int(self.rng.integers(2**63)))
        self.mixture_log_weights = self._log_weights(
            0.5 if self.mixture_weight is None else self.mixture_weight
        )
        self.mixture_history = []

    @staticmethod
    def _log_weights(w):
        return np.log(np.array([1.0 - w, w]))

    def initialise(self, resumed=False):
        super().initialise(resumed=resumed)
        if getattr(self, "secondary", None) is None:
            self._build_secondary()
        if not resumed:
            self.secondary.initialise()
            self._copy_truncation()

    def _copy_truncation(self):
        """The secondary's truncation rules: a copy of the run's (for its
        own population, after a switch)."""
        scheme = getattr(self, "_truncation_scheme", None)
        if scheme is None:
            return
        self.secondary._truncation_scheme = copy.deepcopy(scheme)
        self.secondary._truncation_scheme.configure(self.secondary)
        if hasattr(self.secondary, "_sync_truncation_state"):
            self.secondary._sync_truncation_state()

    def update_output(self, output):
        super().update_output(output)
        if getattr(self, "secondary", None) is not None:
            self.secondary.update_output(os.path.join(output, "mixture_secondary", ""))

    def resume(self, model, flow_config, weights_file=None):
        super().resume(model, flow_config, weights_file)
        secondary = self.secondary
        config = getattr(secondary, "sheet_flow_config", None) or self.secondary_kwargs.get(
            "flow_config", {})
        if hasattr(secondary, "sheet_flow_config"):
            secondary.resume(model, config)
        else:
            secondary.resume(model, dict(config))
        self._copy_truncation()

    def reset_model_weights(self, **kwargs):
        super().reset_model_weights(**kwargs)
        if self.reset_secondary:
            self.secondary.reset_model_weights(**kwargs)

    # -- training ---------------------------------------------------------
    def train(self, x, plot=True):
        live = getattr(self, "current_live_points", None)
        if self.split_activate:
            return self._train_split(x, live, plot)
        if self.switched_at is not None:
            # after the switch only the secondary's flow is used
            self.secondary.current_live_points = live
            self.secondary.train(x, plot=False)
            self.training_count += 1
            self.populated = False
            return
        super().train(x, plot=plot)
        if self._staged:
            if self._near_switch(warmup=True):
                self.secondary.current_live_points = live
                self.secondary.train(x, plot=False)
            return
        self.secondary.current_live_points = live
        self.secondary.train(x, plot=False)
        self._fit_mixture_weight(live if live is not None and len(live) else x)

    # -- split activation -------------------------------------------------
    def _side(self, x):
        """Whether each point of ``x`` is on the run's proposal's side of the
        frozen split."""
        coordinates, cut, label = self.split
        return cut.classify(coordinates(x)) == label

    def _train_split(self, x, live, plot):
        if self.split is None:
            super().train(x, plot=plot)
            if live is not None and len(live):
                self._try_split(live, x, plot)
            return
        if self.split_survivor is None:
            self._check_collapse(x)
        if self.split_survivor == 0:
            super().train(x, plot=plot)
            return
        if self.split_survivor == 1:
            self.secondary.current_live_points = live
            self.secondary.train(x, plot=False)
            self.training_count += 1
            self.populated = False
            return
        mine = self._side(x)
        live_mine = None if live is None else self._side(live)
        # each component trains as a partitioned run's region would: on its
        # side's points, seeing its side's live points (the population cap
        # and the mixture weight then use all of them)
        if live is not None:
            self.current_live_points = live[live_mine]
        try:
            super().train(x[mine], plot=plot)
        finally:
            self.current_live_points = live
        self.secondary.current_live_points = None if live is None else live[~live_mine]
        self.secondary.train(x[~mine], plot=False)
        self._fit_mixture_weight(live if live is not None and len(live) else x)

    def _check_collapse(self, x):
        """Collapse to one component once a side of the split holds less
        than ``split_collapse_share`` of the training points ``x``."""
        n_mine = int(self._side(x).sum())
        counts = (n_mine, len(x) - n_mine)
        if min(counts) >= self.split_collapse_share * len(x):
            return
        self.split_survivor = 0 if counts[0] >= counts[1] else 1
        self.mixture_log_weights = np.where(np.arange(2) == self.split_survivor, 0.0, -np.inf)
        lnx = self._live_lnx()
        logger.info(
            "Mixture: collapses at ln X %.2f to the %s alone (%d and %d training points "
            "on the run's and the secondary's sides, < %.2f of them)",
            np.nan if lnx is None else lnx,
            "secondary" if self.split_survivor else "run's proposal",
            counts[0], counts[1], self.split_collapse_share)

    def _try_split(self, live, x, plot):
        """Start the mixture once the diagonal split is active with both
        experts trained (the partition's ``DiagonalSplitTrigger``)."""
        from .partition import DiagonalSplitTrigger, SkyCoordinates

        if getattr(self, "populating", False):
            return
        try:
            coordinates = SkyCoordinates(["cos_theta_jn", "sky_v"], fold="plane")
            coordinates.configure(self.model)
            coordinates = coordinates.freeze(self)
            cut = DiagonalSplitTrigger().propose(
                coordinates(live), coordinates.period_list(), points=live, proposal=self)
        except Exception as e:  # the check changes nothing: no mixture yet
            logger.warning("Mixture: split check failed, not starting: %r", e)
            return
        if cut is None:
            return
        labels = cut.classify(coordinates(live))
        counts = np.bincount(labels, minlength=2)
        if counts.min() < self.split_min_share * len(live):
            logger.info("Mixture: diagonal split leaves %s live points per side "
                        "(< %.2f of them): not starting", counts.tolist(),
                        self.split_min_share)
            return
        cos = np.abs(np.cos(np.asarray(live["theta_jn"], float)))
        median = [float(np.median(cos[labels == k])) for k in (0, 1)]
        edge = int(np.argmin(median))
        if median[edge] >= self.split_max_abs_cos:
            logger.info("Mixture: neither side of the diagonal split is edge-on "
                        "(median |cos theta_jn| %.2f, %.2f): not starting", *median)
            return
        mine = 1 - edge
        self.split = (coordinates, cut, mine)
        lnx = self._live_lnx()
        logger.info(
            "Mixture: starts at the diagonal split %s (ln X %.2f): the run's proposal "
            "keeps the side of %d live points (median |cos theta_jn| %.2f), the "
            "secondary the edge-on side's %d (%.2f)", cut.describe(),
            np.nan if lnx is None else lnx, counts[mine], median[mine],
            counts[edge], median[edge])
        restrict = getattr(self, "restrict_to_region", None)
        if restrict is not None:
            restrict(live[labels == mine])
        self.populated = False
        self.indices = []
        self._train_split(x, live, plot)

    @property
    def _staged(self):
        return (self.switch_lnx is not None
                or getattr(self, "switch_spread", None) is not None
                or getattr(self, "switch_race", False))

    def _near_switch(self, warmup=False):
        """Whether the live points have reached the switch (or, with
        ``warmup``, its warm-up)."""
        live = getattr(self, "current_live_points", None)
        if live is None or not len(live):
            return False
        if getattr(self, "switch_race", False):
            # the race decides the switch (in populate); its pools need the
            # secondary trained from its start
            lnx = self._live_lnx()
            return bool(warmup) and lnx is not None and lnx <= self.race_start_lnx
        spread = getattr(self, "switch_spread", None)
        if spread is not None:
            logl = np.asarray(live["logL"], float)
            return float(logl.max() - logl.min()) <= spread * (2.0 if warmup else 1.0)
        lnx = self._live_lnx()
        return lnx is not None and lnx <= self.switch_lnx + (self.switch_warmup if warmup else 0.0)

    def _live_lnx(self):
        """ln X of the current live points, ``-max(it) / nlive``."""
        live = getattr(self, "current_live_points", None)
        if live is None or not len(live) or "it" not in (live.dtype.names or ()):
            return None
        return -float(np.max(live["it"])) / len(live)

    def _fit_mixture_weight(self, x):
        entry = dict(training_count=int(self.training_count))
        lq = np.stack([_component_log_q(self, x), _component_log_q(self.secondary, x)])
        ok = np.isfinite(lq).any(axis=0)
        lq = lq[:, ok]
        if self.mixture_weight is None:
            w = float(np.exp(self.mixture_log_weights[1]))
            for _ in range(50):
                r = lq + self._log_weights(w)[:, None]
                resp = np.exp(r[1] - logsumexp(r, axis=0))
                w = float(np.clip(np.mean(resp), self.min_weight, 1 - self.min_weight))
            self.mixture_log_weights = self._log_weights(w)
        r = lq + self.mixture_log_weights[:, None]
        resp = np.exp(r[1] - logsumexp(r, axis=0))
        entry.update(
            weight=float(np.exp(self.mixture_log_weights[1])),
            live_responsibility=float(np.mean(resp)),
            live_secondary_better=float(np.mean(lq[1] > lq[0])),
        )
        self.mixture_history.append(entry)
        logger.info(
            "Mixture: secondary weight %.3f (live points' mean responsibility %.3f, "
            "secondary denser at %.3f of them)",
            entry["weight"], entry["live_responsibility"], entry["live_secondary_better"],
        )

    # -- density ----------------------------------------------------------
    def mixture_log_q(self, x):
        """``log q`` of the mixture at the physical points ``x`` and the
        components' ``log q_k`` (``(2, n)``)."""
        lq = np.stack([_component_log_q(self, x), _component_log_q(self.secondary, x)])
        return logsumexp(lq + self.mixture_log_weights[:, None], axis=0), lq

    def _cap(self):
        live = self.current_live_points
        log_q, _ = self.mixture_log_q(live)
        log_w = self.compute_weights(live, log_q)
        quantile = self.mixture_quantile
        if quantile is None:
            scheme = getattr(self, "_truncation_scheme", None)
            rule = scheme.get_rule("log_proposal_threshold") if scheme else None
            quantile = rule.quantile if rule is not None else 0.005
        finite = log_w[np.isfinite(log_w)]
        return float(np.quantile(finite, 1.0 - quantile)), float(np.mean(~np.isfinite(log_w)))

    def _draw_component(self, k, n):
        """``n`` draws of component ``k`` (0: self, 1: secondary): physical
        points and that component's ``log q``."""
        proposal = self if k == 0 else self.secondary
        if getattr(proposal, "map_to_unit_hypercube", False):
            raise RuntimeError("the mixture proposal needs map_to_unit_hypercube = False")
        z = proposal.sample_latent_distribution(n)
        x, log_q = proposal.backward_pass(z, rescale=True)[:2]
        if not len(x):
            return x, np.empty(0)
        finite = np.isfinite(log_q)
        x = x[finite]
        names = self.model.names + nessai_config.livepoints.non_sampling_parameters
        return rfn.repack_fields(x[names]), np.asarray(log_q, float)[finite]

    # -- population -------------------------------------------------------
    def populate(self, worst_point, n_samples=10000, plot=True, r=None, max_samples=1_000_000):
        """Fill the pool with ``n_samples`` draws of the mixture, kept by
        rejection against the weight cap (see the module notes)."""
        if self._staged:
            return self._populate_stage(worst_point, n_samples, plot, r, max_samples)
        if self.split_activate and (self.split is None or self.split_survivor == 0):
            return super().populate(worst_point, n_samples=n_samples, plot=plot, r=r,
                                    max_samples=max_samples)
        st = datetime.datetime.now()
        if not self.initialised:
            raise RuntimeError("Proposal has not been initialised.")
        log_cap, live_lost = self._cap()
        log_w_k = self.mixture_log_weights
        accepted, n_proposed, n_accepted = [], 0, 0
        counts = np.zeros((2, 3), dtype=int)  # drawn, past the bound, kept
        n_clipped = 0
        batch = min(int(getattr(self, "drawsize", None) or n_samples), n_samples)
        while n_accepted < n_samples:
            n2 = self.rng.binomial(batch, np.exp(log_w_k[1]))
            for k, n in ((0, batch - n2), (1, n2)):
                if not n:
                    continue
                n_proposed += n
                counts[k, 0] += n
                x, log_q_own = self._draw_component(k, n)
                if not len(x):
                    continue
                log_p = self.log_prior(x)
                # first test against the bound q >= w_k q_k
                log_a1 = np.minimum(log_p - log_w_k[k] - log_q_own - log_cap, 0.0)
                keep = np.log(self.rng.random(len(x))) < log_a1
                if not keep.any():
                    continue
                x, log_q_own, log_p, log_a1 = x[keep], log_q_own[keep], log_p[keep], log_a1[keep]
                counts[k, 1] += len(x)
                log_q = log_w_k[k] + log_q_own
                if np.isfinite(log_w_k[1 - k]):
                    other = self.secondary if k == 0 else self
                    log_q = np.logaddexp(log_q, log_w_k[1 - k] + _component_log_q(other, x))
                log_w = log_p - log_q
                log_a = np.minimum(log_w - log_cap, 0.0)
                keep = np.log(self.rng.random(len(x))) < log_a - log_a1
                counts[k, 2] += int(keep.sum())
                n_clipped += int((log_w[keep] > log_cap).sum())
                accepted.append(x[keep])
                n_accepted += int(keep.sum())
            if n_proposed > max_samples:
                logger.warning("Mixture: reached max_samples (%d) while populating", max_samples)
                break
        x = np.concatenate(accepted) if accepted else accepted
        self.x = x[:n_samples]
        self.population_clipped_share = n_clipped / max(n_accepted, 1)
        self.samples = self.convert_to_samples(self.x, plot=False)
        self.population_time += datetime.datetime.now() - st
        self.samples["logL"] = self.model.batch_evaluate_log_likelihood(self.samples)
        if self.check_acceptance:
            self.acceptance.append(self.compute_acceptance(worst_point["logL"]))
        self.indices = self.rng.permutation(self.samples.size).tolist()
        self.population_acceptance = n_accepted / n_proposed
        entry = dict(
            populated_count=int(self.populated_count),
            log_cap=log_cap,
            live_without_density=live_lost,
            weight=float(np.exp(log_w_k[1])),
            drawn=counts[:, 0].tolist(),
            past_bound=counts[:, 1].tolist(),
            kept=counts[:, 2].tolist(),
            clipped_share=self.population_clipped_share,
        )
        if not hasattr(self, "mixture_population_history"):
            self.mixture_population_history = []
        self.mixture_population_history.append(entry)
        logger.info(
            "Mixture pool %d: acceptance %.3g, drawn %s, past the bound %s, kept %s "
            "(network, chart); clipped %.3f; cap %.2f",
            entry["populated_count"], self.population_acceptance, entry["drawn"],
            entry["past_bound"], entry["kept"], entry["clipped_share"], log_cap,
        )
        self.populated_count += 1
        self.populated = True
        self._checked_population = False


    def _populate_stage(self, worst_point, n_samples, plot, r, max_samples):
        """``switch_lnx`` / ``switch_spread`` / ``switch_race``: the pool of the
        run's proposal, or of the secondary once switched (or for a race)."""
        if self.switched_at is None and not getattr(self, "switch_race", False) \
                and self._near_switch():
            self._switch()
        if self.switched_at is None and getattr(self, "switch_race", False) \
                and self._near_switch(warmup=True):
            return self._race(worst_point, n_samples, plot, r, max_samples)
        if self.switched_at is None:
            return super().populate(worst_point, n_samples=n_samples, plot=plot, r=r,
                                    max_samples=max_samples)
        self._secondary_pool(worst_point, n_samples, r, max_samples)
        if not self.samples.size:
            logger.warning("Mixture: the secondary's pool is empty (max_samples %d); "
                           "this pool is the run's proposal's", max_samples)
            super().populate(worst_point, n_samples=n_samples, plot=plot, r=r,
                             max_samples=max_samples)

    def _switch(self):
        lnx = self._live_lnx()
        self.switched_at = lnx if lnx is not None else np.nan
        logl = np.asarray(self.current_live_points["logL"], float)
        logger.info("Mixture: the pools are the secondary's from ln X %.2f on (live ln L "
                    "spread %.1f)", self.switched_at, logl.max() - logl.min())

    def _race_gap(self, history):
        """Pools of the run's proposal before the secondary's next: ``race_every
        - 1``, doubled after each secondary pool that lost by over
        ``race_backoff`` times (up to 16 times), reset by a closer one."""
        gap, last_run = self.race_every - 1, None
        for e in history:
            if e["component"] == 0:
                last_run = e["cost"]
            elif last_run is not None and e["cost"] > self.race_backoff * last_run:
                gap = min(2 * gap + 1, 16 * self.race_every - 1)
            else:
                gap = self.race_every - 1
        return gap

    def _race(self, worst_point, n_samples, plot, r, max_samples):
        """One pool of the race: the secondary's after ``_race_gap`` of the
        run's."""
        history = list(self.race_history)
        since = 0
        for e in reversed(history):
            if e["component"] == 1:
                break
            since += 1
        k = 1 if since >= self._race_gap(history) else 0
        st = datetime.datetime.now()
        if k:
            self._secondary_pool(worst_point, n_samples, r,
                                 min(max_samples, getattr(self, "race_max_samples", max_samples)))
        else:
            super().populate(worst_point, n_samples=n_samples, plot=plot, r=r,
                             max_samples=max_samples)
        seconds = (datetime.datetime.now() - st).total_seconds()
        n_new = int(np.sum(self.samples["logL"] > worst_point["logL"]))
        entry = dict(lnx=self._live_lnx(), component=k, seconds=seconds, n_new=n_new,
                     n_pool=int(self.samples.size), cost=seconds / max(n_new, 1))
        history.append(entry)
        self.race_history = history
        logger.info("Mixture race at ln X %.2f: %s pool %.1f s, %d of %d above the threshold, "
                    "%.3f s per new point", entry["lnx"], ("network", "secondary")[k], seconds,
                    n_new, entry["n_pool"], entry["cost"])
        if k and not self.samples.size:
            # nothing kept within race_max_samples: a lost pool (the backoff
            # spaces the next), and the draws come from the run's proposal
            super().populate(worst_point, n_samples=n_samples, plot=plot, r=r,
                             max_samples=max_samples)
        elif k:
            costs = {}
            wins = 0
            for e in history:
                if e["component"] == 0:
                    costs[0] = e["cost"]
                elif 0 in costs:
                    wins = wins + 1 if e["cost"] < costs[0] else 0
            if wins >= self.race_wins:
                self._switch()

    def _secondary_pool(self, worst_point, n_samples, r, max_samples):
        st = datetime.datetime.now()
        sec = self.secondary
        sec.current_live_points = self.current_live_points
        if not sec.training_count:
            sec.train(self.current_live_points, plot=False)
        sec.populate(worst_point, n_samples=n_samples, plot=False, r=r, max_samples=max_samples)
        self.x = sec.x
        self.samples = sec.samples
        self.indices = self.rng.permutation(self.samples.size).tolist()
        sec.indices, sec.populated = [], False
        self.population_acceptance = sec.population_acceptance
        if self.check_acceptance and getattr(sec, "acceptance", None):
            self.acceptance.append(sec.acceptance[-1])
        self.population_time += datetime.datetime.now() - st
        self.populated_count += 1
        self.populated = True
        self._checked_population = False

def mixture_proposal_class(primary_class, secondary_class, secondary_kwargs=None,
                           mixture_weight=0.5, min_weight=0.05, mixture_quantile=None,
                           reset_secondary=False, switch_lnx=None, switch_warmup=2.0,
                           switch_spread=None, switch_race=False, race_start_lnx=-5.0,
                           race_every=4, race_wins=2, split_activate=False,
                           split_max_abs_cos=0.5, split_min_share=0.2,
                           class_name="MixtureFlowProposal"):
    """``primary_class`` (the run's proposal) mixed with a second proposal
    (see :class:`MixtureProposalMixin`), published at module scope as
    ``class_name`` so checkpoints unpickle (build it again before resuming).

    Parameters
    ----------
    primary_class : type
        The run's proposal class (a nessai ``FlowProposal`` subclass).
    secondary_class : type
        The second component's class, e.g. a bound ring-chart proposal
        (:func:`nessai_gw.sheet_chart.ring_chart_proposal_class`).
    secondary_kwargs : dict, optional
        Its keywords (e.g. ``flow_config``, ``training_config``,
        ``latent_temperature``).
    mixture_weight : float or None
        The secondary's share of the draws; ``None`` fits it to the live
        points after every training.
    switch_lnx, switch_warmup, switch_spread : float, optional
        Instead of a mixture, the run's proposal until ln X reaches
        ``switch_lnx`` (or the live ln L spread falls to ``switch_spread``),
        the secondary after (see :class:`MixtureProposalMixin`).
    switch_race, race_start_lnx, race_every, race_wins : optional
        Instead, a race of the two from ln X ``race_start_lnx`` on, the
        switch when the secondary's measured cost per new point beats the
        run's at ``race_wins`` of its pools in a row (see
        :class:`MixtureProposalMixin`).
    split_activate, split_max_abs_cos, split_min_share : optional
        Instead, the mixture from the run's diagonal split on, each
        component on its side of the line (see the module notes).
    """
    if split_activate and (switch_lnx is not None or switch_spread is not None
                           or switch_race):
        raise ValueError("split_activate takes no switch or race")
    attrs = dict(
        secondary_class=secondary_class,
        secondary_kwargs=dict(secondary_kwargs or {}),
        mixture_weight=None if mixture_weight is None else float(mixture_weight),
        min_weight=float(min_weight),
        mixture_quantile=mixture_quantile,
        reset_secondary=bool(reset_secondary),
        switch_lnx=None if switch_lnx is None else float(switch_lnx),
        switch_warmup=float(switch_warmup),
        switch_spread=None if switch_spread is None else float(switch_spread),
        switch_race=bool(switch_race),
        race_start_lnx=float(race_start_lnx),
        race_every=int(race_every),
        race_wins=int(race_wins),
        split_activate=bool(split_activate),
        split_max_abs_cos=float(split_max_abs_cos),
        split_min_share=float(split_min_share),
        __module__=__name__,
        __qualname__=class_name,
    )
    cls = type(class_name, (MixtureProposalMixin, primary_class), attrs)
    globals()[class_name] = cls
    return cls
