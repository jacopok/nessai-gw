r"""Validate the triangular-ET group action against a real BNS likelihood.

This script does two things with a bilby BNS ``Result`` and its (pickled)
likelihood object:

1.  **Consistency check.** For a batch of posterior samples it recomputes
    ``log_likelihood_ratio`` and compares it with the stored
    ``log_likelihood`` column, to confirm the likelihood object and the
    posterior belong together.

2.  **Mode weights.** For ~1k posterior samples it applies every element of
    :class:`nessai_gw.group_mixture.ETTriangleGroupAction` and recomputes the
    likelihood at each of the 8 images.  If the prior is flat and
    group-invariant in the measure-preserving coordinates
    (:data:`~nessai_gw.group_mixture.TRIANGULAR_DETECTOR_PARAMETERS`), then for a
    posterior draw ``x`` the probability that the truth is image ``g`` is

        w_g(x) = p(g.x) / sum_h p(h.x)
               = softmax_g( logL(g.x) + logpi(g.x) )

    and, averaging over ``x`` drawn from the posterior,

        E_x[ w_g(x) ] = \int_{fundamental domain} p(g.y) dy  ==  Q_g,

    the posterior mass carried by mode ``g`` (the ``sum_g Q_g = 1``
    normalisation is exact).  ``Q_g`` is exactly the mixture weight the
    group-mixture flow should learn for that mode.  Error bars come from a
    bootstrap over the sampled points.

The likelihood used here is the pickled
``RelativeBinningGravitationalWaveTransientNextGenerationModebyMode`` from an
``xG_inference`` ET run, so this example needs ``bilby_xG`` (and the run
directory) available; it is not exercised in CI.

Example
-------
::

    python examples/validate_group_mixture.py \
        --result examples/BNS_result.json \
        --likelihood ../xG_inference/ET_Delta_600Mpc_IMRPhenomXPHM_v3_group/likelihood_BNS.pickle \
        --n-samples 1000
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from nessai_gw.group_mixture import (
    TRIANGULAR_DETECTOR_GROUP_SIZE,
    ETTriangleGroupAction,
    detector_plane_normal,
    detector_vertex,
)

#: Physical parameters (besides the acted-on ones) needed to evaluate the
#: likelihood; filtered against the columns actually present.
_EXTRA_PARAMS = [
    "chirp_mass",
    "mass_ratio",
    "luminosity_distance",
    "chi_1",
    "chi_2",
    "lambda_1",
    "lambda_2",
]


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--result",
        type=Path,
        default=here / "BNS_result.json",
        help="bilby Result file (JSON) with the posterior.",
    )
    p.add_argument(
        "--likelihood",
        type=Path,
        default=(
            here.parent.parent
            / "xG_inference"
            / "ET_Delta_600Mpc_IMRPhenomXPHM_v3_group"
            / "likelihood_BNS.pickle"
        ),
        help="Pickled bilby likelihood object.",
    )
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--n-check", type=int, default=50,
                   help="Samples used for the logL consistency check.")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outfile", type=Path, default=here / "group_mixture_weights.json")
    p.add_argument("--plot", type=Path, default=here / "group_mixture_weights.png")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def load_posterior(path: Path):
    """Return the posterior DataFrame from a bilby Result JSON."""
    import bilby

    return bilby.core.result.read_in_result(str(path)).posterior


def to_group_coords(row) -> dict:
    """One posterior row -> the measure-preserving coordinate dict (1-tensors)."""
    return {
        "ra": torch.tensor([row["ra"]], dtype=torch.float64),
        "sin_dec": torch.tensor([np.sin(row["dec"])], dtype=torch.float64),
        "cos_theta_jn": torch.tensor([np.cos(row["theta_jn"])], dtype=torch.float64),
        "psi": torch.tensor([row["psi"]], dtype=torch.float64),
        "phase": torch.tensor([row["phase"]], dtype=torch.float64),
        "geocent_time": torch.tensor([row["geocent_time"]], dtype=torch.float64),
    }


def from_group_coords(out: dict) -> dict:
    """Group-action output -> bilby physical parameters."""
    return {
        "ra": float(out["ra"][0]) % (2.0 * np.pi),
        "dec": float(np.arcsin(np.clip(float(out["sin_dec"][0]), -1.0, 1.0))),
        "theta_jn": float(np.arccos(np.clip(float(out["cos_theta_jn"][0]), -1.0, 1.0))),
        "psi": float(out["psi"][0]),
        "phase": float(out["phase"][0]),
        "geocent_time": float(out["geocent_time"][0]),
    }


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    print(f"loading likelihood   {args.likelihood}")
    with open(args.likelihood, "rb") as f:
        likelihood = pickle.load(f)
    print(f"loading posterior    {args.result}")
    post = load_posterior(args.result)
    print(f"posterior: {len(post)} samples, {post.shape[1]} columns")

    extra = [c for c in _EXTRA_PARAMS if c in post.columns]

    def log_l(params: dict) -> float:
        likelihood.parameters.update({k: float(v) for k, v in params.items()})
        return float(likelihood.log_likelihood_ratio())

    # ------------------------------------------------------------------
    # 1. consistency check: recomputed vs stored log-likelihood
    # ------------------------------------------------------------------
    idx = rng.choice(len(post), size=min(args.n_check, len(post)), replace=False)
    diffs = []
    for i in idx:
        row = post.iloc[i]
        params = {k: row[k] for k in extra}
        params.update(
            {k: row[k] for k in ("ra", "dec", "theta_jn", "psi", "phase", "geocent_time")}
        )
        diffs.append(log_l(params) - float(row["log_likelihood"]))
    diffs = np.asarray(diffs)
    print(
        "\n[1] recomputed - stored log-likelihood over "
        f"{len(diffs)} samples:\n"
        f"    mean {diffs.mean():+.3f}   std {diffs.std():.3f}   "
        f"max|.| {np.abs(diffs).max():.3f}"
    )

    # ------------------------------------------------------------------
    # 2. per-mode weights
    # ------------------------------------------------------------------
    ifos = likelihood.interferometers
    action = ETTriangleGroupAction(
        reference_time=float(post["geocent_time"].mean()),
        plane_normal=detector_plane_normal(ifos),
        vertex=detector_vertex(ifos),
    )

    gt_prior = None
    try:
        import bilby

        priors = bilby.core.result.read_in_result(str(args.result)).priors
        gt_prior = (priors["geocent_time"].minimum, priors["geocent_time"].maximum)
    except Exception:  # pragma: no cover
        pass

    n = min(args.n_samples, len(post))
    sidx = rng.choice(len(post), size=n, replace=False)
    G = TRIANGULAR_DETECTOR_GROUP_SIZE

    # log posterior (up to a constant) at every image of every sample
    log_p = np.full((n, G), -np.inf)
    base_minus_image = []
    for j, i in enumerate(sidx):
        row = post.iloc[i]
        extra_params = {k: row[k] for k in extra}
        coords = to_group_coords(row)
        base = log_l(
            {**extra_params, **from_group_coords(coords)}
        )
        for g in range(G):
            out = action(coords, torch.tensor([g]))
            phys = from_group_coords(out)
            ln_pi = 0.0
            if gt_prior is not None and not (
                gt_prior[0] <= phys["geocent_time"] <= gt_prior[1]
            ):
                ln_pi = -np.inf
            log_p[j, g] = log_l({**extra_params, **phys}) + ln_pi
        base_minus_image.append(base - log_p[j, 0])
        if (j + 1) % 100 == 0:
            print(f"    {j + 1}/{n}")

    # identity element should reproduce the original likelihood
    bmi = np.asarray(base_minus_image)
    print(
        f"\n[2] identity-element check (base - image[0] logL): "
        f"mean {bmi.mean():+.2e}  max|.| {np.abs(bmi).max():.2e}"
    )

    # softmax over modes for each sample -> per-sample mode probabilities
    w = log_p - log_p.max(axis=1, keepdims=True)
    w = np.exp(w)
    w /= w.sum(axis=1, keepdims=True)

    weights = w.mean(axis=0)

    # bootstrap error bars over the sampled points
    boot = np.empty((args.n_bootstrap, G))
    for b in range(args.n_bootstrap):
        take = rng.integers(0, n, size=n)
        boot[b] = w[take].mean(axis=0)
    err = boot.std(axis=0)
    lo, hi = np.percentile(boot, [16, 84], axis=0)

    print(f"\n    per-mode weights (n = {n} samples, bootstrap error bars)")
    print("    mode   (k, reflected)     weight +- err        [16, 84]%")
    for g in range(G):
        k, refl = g % 4, g // 4
        print(
            f"    {g:>2d}     (k={k}, refl={int(refl)})     "
            f"{weights[g]:.4f} +- {err[g]:.4f}     "
            f"[{lo[g]:.4f}, {hi[g]:.4f}]"
        )
    print(f"    sum = {weights.sum():.6f}")

    result = {
        "n_samples": int(n),
        "seed": int(args.seed),
        "group_size": int(G),
        "mode_index_convention": "g = reflected*4 + k, k = rotation by k*pi/2",
        "logl_consistency": {
            "mean": float(diffs.mean()),
            "std": float(diffs.std()),
            "max_abs": float(np.abs(diffs).max()),
        },
        "weights": weights.tolist(),
        "weights_err_bootstrap": err.tolist(),
        "weights_ci68": [lo.tolist(), hi.tolist()],
    }
    args.outfile.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.outfile}")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        g = np.arange(G)
        ax.bar(g, weights, yerr=err, capsize=4, color="#4c72b0")
        ax.axhline(1.0 / G, color="k", ls=":", lw=1, label="uniform (1/8)")
        ax.set_xticks(g)
        ax.set_xticklabels(
            [f"{gi}\n(k={gi % 4}, r={gi // 4})" for gi in g], fontsize=8
        )
        ax.set_xlabel("group element")
        ax.set_ylabel("posterior mode weight $Q_g$")
        ax.set_title(f"triangular-ET group-mixture weights ({n} samples)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
