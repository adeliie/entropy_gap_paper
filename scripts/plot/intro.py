import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from package.plot_functions import update_style

from package import DATA_FILE_INTRO
from package import plot_functions as pf

OUT_FILE = "plot/optim_quad_vs_ce.pdf"

COLORS = {"gd": "royalblue", "sd": "crimson", "adam": "darkorange"}
LABELS = {"gd": "GD", "sd": "SD", "adam": "Adam"}


def load_data():
    with np.load(DATA_FILE_INTRO) as loaded:
        return {key: loaded[key] for key in loaded.files}


def postprocess(data):
    mask = data["eval_steps"] <= 100
    processed = dict(data)
    processed["eval_steps"] = data["eval_steps"][mask]

    for algo_key in COLORS:
        processed[f"err_q_{algo_key}"] = data[f"err_q_{algo_key}"][mask]
        processed[f"err_ce_{algo_key}"] = data[f"err_ce_{algo_key}"][mask]

    return processed


def settings(plt):
    update_style(plt, nrows=1, ncols=2, rel_width=1.0)


def make_figure(fig, data):
    axes = pf.make_subplots_on_figure(fig, nrows=1, ncols=2, sharey=True)
    ax1, ax2 = axes

    for algo_key, color in COLORS.items():
        x = data["eval_steps"]
        ax1.plot(
            x,
            data[f"err_q_{algo_key}"],
            color=color,
            lw=2,
            label=LABELS[algo_key],
        )
        ax2.plot(x, data[f"err_ce_{algo_key}"], color=color, lw=2)

    ax1.set_title("Quadratic (MSE)")
    ax2.set_title("Cross-Entropy")

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    ax1.set_ylabel("relative error")
    ax2.tick_params(labelleft=False)

    legend_handles = [
        mlines.Line2D([0], [0], color=color, lw=2, label=LABELS[algo_key])
        for algo_key, color in COLORS.items()
    ]
    ax1.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=7)

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
