import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patches
from torch.nn import functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from package import plot_functions as pf
from package.plot_functions import update_style

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_FILE = "plot/sign_weight.pdf"
D = 10000
T = 400
ETA = 0.05


def load_data():
    ranks = torch.arange(1, D + 1, dtype=torch.float64, device=device)
    pi = 1.0 / ranks
    pi /= torch.sum(pi)
    log_pi = torch.log(pi)
    W = torch.full((D,), -np.log(D), dtype=torch.float64, device=device)
    indices = [x - 1 for x in [10, 100, 1000, 10000]]
    w_traj = {k: np.zeros(T) for k in indices}
    traj = {k: np.zeros(T) for k in indices}
    delta_history_window = []
    with torch.no_grad():
        for t in tqdm(range(T), total=T, leave=False):

            P = F.softmax(W, dim=0)
            Z = torch.sum(torch.exp(W))

            c = torch.mean(W - log_pi)

            delta = W - (log_pi + c)

            for k in indices:
                w_traj[k][t] = W[k].item()
                traj[k][t] = (log_pi[k] + torch.log(Z)).item()

            if t >= T - 30:
                delta_history_window.append(delta.cpu().numpy())

            W += ETA * torch.sign(pi - P)

        delta_history = np.array(delta_history_window)

        deltas1 = delta_history[-1:].flatten()
    result = w_traj, traj, deltas1, indices
    W_traj, target_traj, deltas, track_indices = result

    return {
        "W_traj": W_traj,
        "target_traj": target_traj,
        "deltas": deltas,
        "track_indices": track_indices,
        "T": T,
        "eta": ETA,
    }


def postprocess(data):
    processed = dict(data)
    processed["residual_traj"] = {
        k: data["W_traj"][k] - data["target_traj"][k] for k in data["track_indices"]
    }
    return processed


def settings(plt):
    update_style(plt, nrows=1, ncols=3, rel_width=1.0, height_to_width_ratio=1.0)


def make_figure(fig, data):
    ax1, ax2, ax3 = pf.make_subplots_on_figure(fig, nrows=1, ncols=3)

    W_traj = data["W_traj"]
    target_traj = data["target_traj"]
    deltas = data["deltas"]
    track_indices = data["track_indices"]
    residual_traj = data["residual_traj"]
    total_steps = data["T"]
    eta = data["eta"]

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(track_indices)))

    lines, labels = [], []
    for idx, (k, col) in enumerate(zip(track_indices, colors)):
        (line,) = ax1.plot(
            W_traj[k], color=col, alpha=0.8, lw=1.5, label=rf"$w_{{{k+1}}}$"
        )
        ax1.plot(target_traj[k], color=col, linestyle="--", alpha=0.5, lw=2.5)
        lines.append(line)
        labels.append(rf"$w_{{{k+1}}}$")

    ax1.set_title("Weights and targets")
    ax1.set_ylim([-15, 0])
    ax1.set_xlim(0, 200)
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$w_k(t)$ and target")
    ax1.legend(
        lines,
        labels,
        loc="upper center",
        ncol=2,
        # borderaxespad=-0.5,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.4,
        frameon=False,
        labelspacing=0.0,
    )

    for idx, (k, col) in enumerate(zip(track_indices, colors)):
        ax2.plot(residual_traj[k], color=col, alpha=0.5, lw=1)

    ax2.axhline(2 * eta, color="black", linestyle="--", alpha=0.5, lw=1, zorder=1)
    ax2.axhline(-2 * eta, color="black", linestyle="--", alpha=0.5, lw=1, zorder=1)

    ax2.fill_between(
        [0, total_steps],
        -2 * eta,
        2 * eta,
        color="black",
        alpha=0.1,
        zorder=0,
        label=r"$[-2\eta,2\eta]$",
    )
    ax2.legend(
        loc="lower right",
        ncol=2,
        # borderaxespad=-0.5,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.4,
        frameon=False,
        labelspacing=0.0,
    )

    ax2.set_title(r"Residuals")
    ax2.set_ylabel(r"$\mathrm{res}_k(t)$")
    ax2.set_xlabel("t")
    ax2.set_xlim(0, 200)
    ax2.set_ylim(-4 * eta, 4 * eta)

    ax3.hist(deltas, bins=100, density=True, alpha=0.5)

    rect_uniform = patches.Rectangle(
        (-eta, -1 / (2 * eta)),
        2 * eta,
        2 / (2 * eta),
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        zorder=10,
    )
    ax3.add_patch(rect_uniform)

    theory_handle = mlines.Line2D(
        [], [], color="black", linestyle="--", lw=1.5, label=r"Uniform $[-\eta, \eta]$"
    )
    ax3.set_ylim([0, 1.3 * (0.5 / eta)])
    ax3.set_yticks([0, 0.25 / eta, 0.5 / eta])
    ax3.legend(
        handles=[theory_handle],
        loc="upper center",
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.0,
    )

    ax3.set_title(r"Distribution of $\delta_k$")
    ax3.set_ylabel(r"Density")
    ax3.set_xlabel(r"$\delta_k$")

    ax3.set_xlim(-eta - 0.03, eta + 0.03)

    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major")

    fig.tight_layout(pad=0.2)
    return fig


if __name__ == "__main__":
    settings(plt)
    fig = plt.figure()
    data = postprocess(load_data())
    make_figure(fig, data)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    fig.savefig(OUT_FILE, bbox_inches="tight")
    plt.show()
