"""Visualise the eight-mode triangular-ET symmetry on a real BNS posterior.

Uses the octomodal ET BNS posterior from nessai's
``replay_optimization/BNS_result.json`` (a small extract ships in
``tests/data/bns_et_octomodal_posterior.npz``).  The left panel shows the
sky posterior in the ET-EMR detector-plane frame -- four peaks spaced by
``pi / 2`` in azimuth, mirrored across the plane -- and the right panel the
posterior after folding every sample into the fundamental domain with
:class:`nessai_gw.group_mixture.ETTriangleGroupAction`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nessai_gw.group_mixture import (
    ET_TRIANGLE_PARAMETERS,
    ETTriangleGroupAction,
)

DATA = (
    Path(__file__).parents[1]
    / "tests"
    / "data"
    / "bns_et_octomodal_posterior.npz"
)


def main() -> None:
    post = np.load(DATA)
    action = ETTriangleGroupAction(
        reference_time=float(np.mean(post["geocent_time"]))
    )
    keys = ET_TRIANGLE_PARAMETERS
    coords = {
        "ra": np.asarray(post["ra"]),
        "sin_dec": np.sin(post["dec"]),
        "cos_theta_jn": np.cos(post["theta_jn"]),
        "psi": np.asarray(post["psi"]),
        "phase": np.asarray(post["phase"]),
        "geocent_time": np.asarray(post["geocent_time"]),
    }
    tp = {k: torch.as_tensor(coords[k]) for k in keys}

    dec = np.arcsin(np.clip(coords["sin_dec"], -1, 1))
    lam, beta, _ = action._to_frame(
        tp["ra"], torch.as_tensor(dec), tp["psi"]
    )
    lam = np.mod(lam.numpy(), 2 * np.pi)
    beta = beta.numpy()

    # Fold: keep, for each sample, the unique image in the fundamental domain.
    folded = {k: np.array(coords[k]) for k in keys}
    assigned = action.in_fundamental_domain(tp).numpy()
    for g in range(1, action.group_size):
        modes = torch.full((len(assigned),), g, dtype=torch.long)
        image = {k: v.numpy() for k, v in action(tp, modes).items()}
        take = (
            action.in_fundamental_domain(
                {k: torch.as_tensor(v) for k, v in image.items()}
            ).numpy()
            & ~assigned
        )
        for k in keys:
            folded[k] = np.where(take, image[k], folded[k])
        assigned |= take

    fp = {k: torch.as_tensor(v) for k, v in folded.items()}
    dec_f = torch.asin(torch.clamp(fp["sin_dec"], -1.0, 1.0))
    lam_f, beta_f, _ = action._to_frame(fp["ra"], dec_f, fp["psi"])

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    axs[0].hexbin(np.degrees(lam), np.degrees(beta), gridsize=45, cmap="magma")
    axs[0].set_title("posterior in the ET detector-plane frame")
    for k in range(1, 4):
        axs[0].axvline(90 * k, color="c", ls=":", lw=1)
    axs[0].axhline(0, color="c", ls=":", lw=1)
    axs[1].hexbin(
        np.degrees(np.mod(lam_f.numpy(), 2 * np.pi)),
        np.degrees(beta_f.numpy()),
        gridsize=45,
        cmap="magma",
    )
    axs[1].set_title("folded into the fundamental domain")
    for ax in axs:
        ax.set_xlabel(r"detector-frame azimuth $\lambda$ [deg]")
        ax.set_xlim(0, 360)
    axs[0].set_ylabel(r"detector-frame elevation $\beta$ [deg]")
    fig.tight_layout()
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
