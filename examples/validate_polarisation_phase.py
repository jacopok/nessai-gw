"""Validate the polarisation-phase (``delta_phase = phase +/- psi``) coordinate.

Folds the octomodal ET BNS posterior
(``tests/data/bns_et_octomodal_posterior.npz``) into a single mode with
:class:`nessai_gw.group_mixture.ETTriangleGroupAction`, maps the folded set into
the flow's prime space several ways, and fits an identical RealNVP to each:

phase coordinate
  * ``phase``       -- the old ``angle-2pi`` pair (baseline)
  * ``phase+psi``   -- the plan's ``delta_phase``
  * ``phase-psi``   -- the opposite sign

fundamental domain (which reflection representative is "canonical")
  * ``beta``   -- detector-plane hemisphere ``beta_f >= 0`` (what
    ``ETTriangleGroupAction.in_fundamental_domain`` / the group-mixture wrapper
    actually use)
  * ``face``   -- ``cos(theta_jn) >= 0`` (face-on)

Every other prime coordinate is identical across runs and the phase-pair
coordinate change has unit Jacobian, so the validation NLLs are directly
comparable.  Writes ``polarisation_phase_validation.{png,json}`` to ``examples/``.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from nessai_gw.group_mixture import (
    ET_TRIANGLE_GROUP_SIZE,
    ETTriangleGroupAction,
    _GEOCENT_SCALE,
)

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "tests" / "data" / "bns_et_octomodal_posterior.npz"
_TWO_PI = 2.0 * np.pi
_KEYS = ETTriangleGroupAction.parameters


def load_posterior():
    d = np.load(DATA)
    return {k: d[k] for k in d.files}


def to_coords(post):
    return {
        "ra": post["ra"],
        "sin_dec": np.sin(post["dec"]),
        "cos_theta_jn": np.cos(post["theta_jn"]),
        "psi": post["psi"],
        "phase": post["phase"],
        "geocent_time": post["geocent_time"],
    }


def _domain_mask(action, tp, domain):
    """Boolean 'is this the canonical representative of its orbit'."""
    dec = torch.asin(torch.clamp(tp["sin_dec"], -1, 1))
    lam_f, beta_f, _ = action._to_frame(tp["ra"], dec, tp["psi"])
    in_quarter = torch.remainder(lam_f, _TWO_PI) < 0.5 * np.pi
    if domain == "beta":
        return in_quarter & (beta_f >= 0.0)
    elif domain == "face":
        return in_quarter & (tp["cos_theta_jn"] >= 0.0)
    raise ValueError(domain)


def fold(action, coords, domain):
    n = len(coords["ra"])
    tp = {k: torch.as_tensor(np.asarray(coords[k], float)) for k in _KEYS}
    folded = {k: np.asarray(tp[k]) for k in _KEYS}
    assigned = _domain_mask(action, tp, domain).numpy()
    for g in range(1, ET_TRIANGLE_GROUP_SIZE):
        modes = torch.full((n,), g, dtype=torch.long)
        image = {k: v.numpy() for k, v in action(tp, modes).items()}
        it = {k: torch.as_tensor(v) for k, v in image.items()}
        take = _domain_mask(action, it, domain).numpy() & ~assigned
        for k in _KEYS:
            folded[k] = np.where(take, image[k], folded[k])
        assigned |= take
    assert assigned.all(), f"{(~assigned).sum()} samples unfolded ({domain})"
    return folded


def build_prime(folded, reference_time, phase_combo, seed):
    """Prime coordinates; only the phase pair depends on ``phase_combo``."""
    rng = np.random.default_rng(seed)
    n = len(folded["ra"])
    dec = np.arcsin(np.clip(folded["sin_dec"], -1, 1))
    ra, psi = folded["ra"], folded["psi"]

    r_sky = np.sqrt(rng.chisquare(3, n))
    cd = np.cos(dec)
    r_psi = np.sqrt(rng.chisquare(2, n))
    r_ph = np.sqrt(rng.chisquare(2, n))

    ct = folded["cos_theta_jn"]
    ascale = 1.0
    if phase_combo == "phase":
        ang, pname = folded["phase"], "phase"
    elif phase_combo == "phase+psi":
        ang, pname = folded["phase"] + psi, "delta_phase"
    elif phase_combo == "phase-psi":
        ang, pname = folded["phase"] - psi, "delta_phase"
    elif phase_combo == "signed":
        ang, pname = folded["phase"] + np.sign(ct) * psi, "delta_phase"
    elif phase_combo == "signed-pi":
        ang, pname, ascale = folded["phase"] + np.sign(ct) * psi, "delta_phase", 2.0
    elif phase_combo == "phase-pi":
        ang, pname, ascale = folded["phase"], "phase", 2.0
    else:
        raise ValueError(phase_combo)
    ang = np.mod(ang * ascale, _TWO_PI)

    cols = {
        "ra_dec_x": r_sky * cd * np.cos(ra),
        "ra_dec_y": r_sky * cd * np.sin(ra),
        "ra_dec_z": r_sky * np.sin(dec),
        "psi_x": r_psi * np.cos(2.0 * psi),
        "psi_y": r_psi * np.sin(2.0 * psi),
        f"{pname}_x": r_ph * np.cos(ang),
        f"{pname}_y": r_ph * np.sin(ang),
        "theta_jn_u": folded["cos_theta_jn"],
        "gt_u": (folded["geocent_time"] - reference_time) / _GEOCENT_SCALE,
    }
    order = list(cols)
    x = np.stack([cols[k] for k in order], axis=1).astype(np.float32)
    return x, order


def train_flow(x_tr, x_vl, *, seed, epochs, batch_size, lr, n_transforms,
               n_neurons):
    from glasflow.flows import RealNVP

    torch.manual_seed(seed)
    flow = RealNVP(
        n_inputs=x_tr.shape[1], n_transforms=n_transforms, n_neurons=n_neurons,
        n_blocks_per_transform=2, batch_norm_between_transforms=True,
    )
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xt, xv = torch.as_tensor(x_tr), torch.as_tensor(x_vl)
    n = xt.shape[0]
    hist = []
    best, best_state = np.inf, None
    for _ in range(epochs):
        flow.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = -flow.log_prob(xt[idx]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
        sched.step()
        flow.eval()
        with torch.no_grad():
            v = -flow.log_prob(xv).mean().item()
        hist.append(v)
        if v < best:
            best = v
            best_state = {k: t.detach().clone()
                          for k, t in flow.state_dict().items()}
    flow.load_state_dict(best_state)
    return flow, hist, best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-transforms", type=int, default=10)
    p.add_argument("--n-neurons", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--domains", nargs="+", default=["beta", "face"])
    p.add_argument(
        "--combos", nargs="+",
        default=["phase", "phase+psi", "phase-psi"],
    )
    args = p.parse_args()

    post = load_posterior()
    reference_time = float(np.mean(post["geocent_time"]))
    action = ETTriangleGroupAction(reference_time=reference_time)
    coords = to_coords(post)
    n = len(coords["ra"])

    rng = np.random.default_rng(1234)
    perm = rng.permutation(n)
    n_val = int(args.val_fraction * n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    folds = {d: fold(action, coords, d) for d in args.domains}

    results = {"reference_time": reference_time, "n": n, "n_val": n_val,
               "config": vars(args), "runs": {}}
    curves = {}
    for domain in args.domains:
        folded = folds[domain]
        for combo in args.combos:
            key = f"{domain}/{combo}"
            best_list, last_list = [], []
            for seed in args.seeds:
                x, order = build_prime(folded, reference_time, combo, seed)
                flow, hist, best = train_flow(
                    x[tr_idx], x[val_idx], seed=seed, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr,
                    n_transforms=args.n_transforms, n_neurons=args.n_neurons,
                )
                best_list.append(best)
                last_list.append(hist[-1])
                if seed == args.seeds[0]:
                    curves[key] = hist
            results["runs"][key] = {
                "best_val_nll_mean": float(np.mean(best_list)),
                "best_val_nll_std": float(np.std(best_list)),
                "best_val_nll_per_seed": best_list,
                "final_val_nll_per_seed": last_list,
            }
            print(f"{key:24s}  best val NLL "
                  f"{np.mean(best_list):+.4f} +/- {np.std(best_list):.4f}")

    # baseline per domain = the 'phase' combo
    for domain in args.domains:
        base = results["runs"][f"{domain}/phase"]["best_val_nll_mean"]
        for combo in args.combos:
            r = results["runs"][f"{domain}/{combo}"]
            r["delta_vs_phase_baseline"] = r["best_val_nll_mean"] - base

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(args.domains), 2, figsize=(13, 5 * len(args.domains)),
        squeeze=False,
    )
    for r, domain in enumerate(args.domains):
        ax = axes[r][0]
        for combo in args.combos:
            ax.plot(curves[f"{domain}/{combo}"][3:], label=combo)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val NLL")
        ax.set_title(f"domain '{domain}': flow fit (log scale, warmup dropped)")
        ax.legend()

        ax = axes[r][1]
        best = {c: results["runs"][f"{domain}/{c}"]["best_val_nll_mean"]
                for c in args.combos}
        bars = ax.bar(list(best), list(best.values()),
                      color=["0.5", "tab:blue", "tab:orange"])
        ax.set_ylabel("best val NLL (lower better)")
        ax.set_title(f"domain '{domain}': best val NLL by phase coordinate")
        for b, v in zip(bars, best.values()):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom" if v < 0 else "top")

    fig.tight_layout()
    fig.savefig(HERE / "polarisation_phase_validation.png", dpi=110)
    (HERE / "polarisation_phase_validation.json").write_text(
        json.dumps(results, indent=2)
    )
    print("\n" + json.dumps(
        {k: {"best": v["best_val_nll_mean"],
             "d_vs_phase": v.get("delta_vs_phase_baseline")}
         for k, v in results["runs"].items()},
        indent=2,
    ))
    print(f"wrote {HERE / 'polarisation_phase_validation.png'}")


if __name__ == "__main__":
    main()
