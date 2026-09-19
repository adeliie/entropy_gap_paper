import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from package import DATA_FILE_OPTETA
from package import plot_functions as pf
from package.plot_functions import update_style

OUT_FILE = "plot/optimal_eta_gd.pdf"


def load_data():
    with np.load(DATA_FILE_OPTETA) as loaded:
        return {key: loaded[key] for key in loaded.files}


def postprocess(data):
    processed = dict(data)
    curves = []

    for d in data["all_ds"]:
        try:
            etas = data[f"d_{d}_etas"]
            errors = data[f"d_{d}_errors_gf"]
        except KeyError:
            print(f"Données manquantes pour d={d}")
            continue

        exponent = int(np.log10(d)) if d in [10**i for i in range(1, 10)] else None
        label = rf" $d=10^{{{exponent}}}$" if exponent else rf"$d={d:,}$"
        opt_eta = np.log(d) ** 2
        curves.append(
            {
                "d": d,
                "normalized_etas": etas / opt_eta,
                "errors": errors,
                "label": label,
            }
        )

    processed["curves"] = curves
    return processed


def settings(plt):
    update_style(
        plt,
        nrows=1,
        ncols=1,
        rel_width=0.5,
    )


def make_figure(fig, data):
    ax = pf.make_subplots_on_figure(fig, nrows=1, ncols=1)
    curves = data["curves"]
    colors = cm.viridis(np.linspace(0, 0.9, len(curves)))

    for curve, color in zip(curves, colors):
        ax.plot(
            curve["normalized_etas"],
            curve["errors"],
            linestyle="-",
            linewidth=2,
            alpha=0.8,
            color=color,
            label=curve["label"],
        )
    ax.set_title("Grid search")

    ax.set_ylim([1e-2, 1e0])
    ax.set_xlim([10**-1, 1e1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best", frameon=False)
    ax.set_xlabel(r" $\eta / \log d^2$")
    ax.set_ylabel("Relative error")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.2)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noshow", action="store_true", help="Skip displaying the figure.")
    args = parser.parse_args()

    settings(plt)
    fig = plt.figure()
    try:
        data = postprocess(load_data())
    except FileNotFoundError:
        print(f"Fichier '{DATA_FILE_OPTETA}' introuvable.")
        raise SystemExit(1)
    make_figure(fig, data)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    fig.savefig(OUT_FILE, bbox_inches="tight")
    if not args.noshow:
        plt.show()
