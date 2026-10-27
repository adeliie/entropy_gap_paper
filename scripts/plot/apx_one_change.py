"""Simulate and plot Zipf bigram weight and probability trajectories."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import softmax
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from package import plot_functions as pf  # pylint: disable=wrong-import-position
from package.plot_functions import update_style  # pylint: disable=wrong-import-position

OUT_FILE = "plot/apx_one_change.pdf"
DS = [10000]
N_CHECKPOINTS = 500


def make_data():
    """Return raw gradient-flow solutions for each dimension."""

    def simulate(d):
        ranks = np.arange(1, d + 1, dtype=np.float64)
        pi = 1.0 / ranks
        pi /= pi.sum()
        eta = 1.0 / pi[0]
        end_t = d**1.5
        times = np.concatenate(([0.0], np.geomspace(1e-3, end_t, N_CHECKPOINTS)))

        with tqdm(
            total=float(end_t), desc=f"Gradient flow (d={d})", unit="t"
        ) as progress:

            def gradient_flow(t, logits):
                progress.update(max(0.0, min(t, progress.total) - progress.n))
                return eta * (pi - softmax(logits))

            solution = solve_ivp(
                gradient_flow,
                (0.0, float(end_t)),
                np.full(d, -np.log(d)),
                t_eval=times,
                rtol=1e-8,
                atol=1e-10,
            )
        if not solution.success:
            raise RuntimeError(f"Gradient flow failed for d={d}: {solution.message}")

        return solution

    data = {"ds": DS}
    for d in DS:
        data[f"d_{d}_solution"] = simulate(d)
    return data


def load_data():
    """Generate the simulation data used by the figure."""
    return make_data()


def postprocess(data):
    """Preserve the raw solutions for plotting."""
    return dict(data)


def settings(plt):  # pylint: disable=redefined-outer-name
    """Apply the shared plotting style."""
    update_style(plt, nrows=1, ncols=2, rel_width=1.0, height_to_width_ratio=0.75)


def make_figure(fig, data):  # pylint: disable=too-many-locals
    """Compare weights and predicted probabilities for d=10000."""
    d = 10000
    n_curves = 9
    axes = pf.make_subplots_on_figure(fig, nrows=1, ncols=2)
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, n_curves))
    # The initial condition is retained in the data; log time starts at t>0.
    solution = data[f"d_{d}_solution"]
    times = solution.t[1:]
    tau = np.log(times) / np.log(d)
    rank_exponents = np.linspace(0, np.log10(d), n_curves)
    selected_ranks = np.rint(10**rank_exponents).astype(int)
    logits = solution.y[:, 1:]
    probabilities = softmax(logits, axis=0)
    panels = [
        (logits, "Weights", "linear", -np.log(d)),
        (probabilities, "Probabilities", "log", 1.0 / d),
    ]

    for ax, (values, label, scale, initial_value) in zip(axes, panels):
        for rank, exponent, color in zip(selected_ranks, rank_exponents, colors):
            ax.plot(
                tau,
                values[rank - 1],
                color=color,
                linewidth=1.5,
                label=rf"$10^{{{exponent:g}}}$",
            )
        reference_line = ax.axhline(initial_value, linestyle="--", color="black")
        ax.set_yscale(scale)
        ax.margins(y=0.1)
        ax.set_title(rf"{label}")
        ax.set_ylabel(label)
        ax.set_xlabel(r"$\tau = \log(t) / \log(d)$")
        ax.set_xlim([-0.5, 1.5])

        rank_legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 1.04),
            ncol=1,
            frameon=False,
            fontsize=6,
            labelspacing=0.2,
            borderpad=0,
        )
        ax.add_artist(rank_legend)
        ax.legend(
            handles=[reference_line],
            labels=[r"$-\log(d)$" if scale == "linear" else r"$1/d$"],
            loc="lower left",
            frameon=False,
            fontsize=6,
        )

    fig.tight_layout(pad=0.5)
    return fig


def main():
    """Generate, save, and optionally display the figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noshow", action="store_true", help="Skip displaying the figure."
    )
    args = parser.parse_args()

    settings(plt)
    fig = plt.figure()
    data = postprocess(load_data())
    make_figure(fig, data)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    fig.savefig(OUT_FILE, bbox_inches="tight")
    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()
