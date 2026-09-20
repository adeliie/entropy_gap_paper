import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from package import DATA_FILE_GD_MATRIX
from package import plot_functions as pf
from package.plot_functions import update_style

OUT_FILE = "plot/sequential_wave_gd.pdf"
DS = [100, 1000, 10000]
U_TARGETS = [0.0, 0.25, 0.5, 0.75, 1]


def load_data():
    with np.load(DATA_FILE_GD_MATRIX) as loaded:
        return {key: loaded[key] for key in loaded.files}


def postprocess(data):
    processed = dict(data)
    panels = []

    for d in DS:
        panel = {"d": d, "curves": []}
        tau_key = f"d_{d}_tau"
        err_key = f"d_{d}_err"

        if tau_key in data and err_key in data:
            tau = data[tau_key]
            err = data[err_key]

            for u in U_TARGETS:
                column = max(1, int(np.round(d**u))) - 1
                panel["curves"].append(
                    {
                        "u": u,
                        "tau": tau,
                        "empirical": err[:, column],
                        "theory": pf.theory_row_specific(tau, u),
                    }
                )

        panels.append(panel)

    processed["panels"] = panels
    return processed


def settings(plt):
    update_style(plt, nrows=1, ncols=len(DS), rel_width=1.0, height_to_width_ratio=0.8)


def make_figure(fig, data):
    axes = pf.make_subplots_on_figure(fig, nrows=1, ncols=len(DS), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(U_TARGETS)))

    for ax_idx, (panel, ax) in enumerate(zip(data["panels"], axes)):
        for curve, color in zip(panel["curves"], colors):
            ax.plot(
                curve["tau"],
                curve["theory"],
                "--",
                color=color,
                linewidth=1.5,
                alpha=0.6,
            )
            ax.plot(
                curve["tau"],
                curve["empirical"],
                "-",
                color=color,
                linewidth=1.5,
            )

        if panel["curves"]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel(r"$\tau : t = d^{\tau} $")
            ax.set_title(rf"$d = {panel['d']:,}$")
            # ax.set_xticklabels([])
            if ax_idx == 0:
                ax.set_ylabel("Relative error")
            ax.set_ylim([0, 1.05])

    color_handles = [
        Line2D([0], [0], color=color, lw=1.5, label=rf"$u = {u}$")
        for u, color in zip(U_TARGETS, colors)
    ]
    style_handles = [
        Line2D([0], [0], color="gray", lw=1.5, linestyle="-", label="$\ell_{d^u}$, GD"),
        Line2D(
            [0],
            [0],
            color="gray",
            lw=1.5,
            linestyle="--",
            alpha=0.8,
            label="$\ell_{d^u}$, Theory",
        ),
    ]
    spacer_handle = Line2D([0], [0], color="none", lw=0, label="")

    fig.legend(
        handles=color_handles + [spacer_handle] + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.15),
        ncol=8,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.5,
        handlelength=1.5,
    )
    fig.tight_layout(pad=0.5, rect=(0, 0.1, 1, 1))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noshow", action="store_true", help="Skip displaying the figure."
    )
    args = parser.parse_args()

    settings(plt)
    fig = plt.figure()
    try:
        data = postprocess(load_data())
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{DATA_FILE_GD_MATRIX}' est introuvable.")
        raise SystemExit(1)
    make_figure(fig, data)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    fig.savefig(OUT_FILE, bbox_inches="tight")
    if not args.noshow:
        plt.show()
