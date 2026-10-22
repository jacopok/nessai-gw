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
    chart runs).
    """

    secondary_class = None
    secondary_kwargs = {}
    mixture_weight = 0.5
    min_weight = 0.05
    mixture_quantile = None
    reset_secondary = False
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

    def reset_model_weights(self, **kwargs):
        super().reset_model_weights(**kwargs)
        if self.reset_secondary:
            self.secondary.reset_model_weights(**kwargs)

    # -- training ---------------------------------------------------------
    def train(self, x, plot=True):
        super().train(x, plot=plot)
        live = getattr(self, "current_live_points", None)
        self.secondary.current_live_points = live
        self.secondary.train(x, plot=False)
        self._fit_mixture_weight(live if live is not None and len(live) else x)

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
                other = self.secondary if k == 0 else self
                log_q = np.logaddexp(log_w_k[k] + log_q_own,
                                     log_w_k[1 - k] + _component_log_q(other, x))
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


def mixture_proposal_class(primary_class, secondary_class, secondary_kwargs=None,
                           mixture_weight=0.5, min_weight=0.05, mixture_quantile=None,
                           reset_secondary=False, class_name="MixtureFlowProposal"):
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
    """
    attrs = dict(
        secondary_class=secondary_class,
        secondary_kwargs=dict(secondary_kwargs or {}),
        mixture_weight=None if mixture_weight is None else float(mixture_weight),
        min_weight=float(min_weight),
        mixture_quantile=mixture_quantile,
        reset_secondary=bool(reset_secondary),
        __module__=__name__,
        __qualname__=class_name,
    )
    cls = type(class_name, (MixtureProposalMixin, primary_class), attrs)
    globals()[class_name] = cls
    return cls
