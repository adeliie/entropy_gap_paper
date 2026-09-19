import argparse
import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from package import DATA_FILE_SIGN
from package import plot_functions as pf
from package.plot_functions import update_style

OUT_FILE = "plot/sgd_eta_osc.pdf"


def load_data():
    with np.load(DATA_FILE_SIGN) as loaded:
        return {key: loaded[key] for key in loaded.files}


def postprocess(data):
    processed = dict(data)
    ds = data["ds"]
    Ts = data["Ts"]
    T_target = Ts[np.argmin(np.abs(Ts - 1000))]
    etas_T_target = data[f"etas_T{T_target}"]

    def cross_entropy_at_init(d):
        # Target: pi_i ∝ 1/i
        i = np.arange(1, d + 1)
        pi = 1.0 / i
        pi /= pi.sum()
        q = np.ones(d) / d

        cross_entropy = -np.sum(pi * np.log(q))
        return cross_entropy

    curves = []
    for d in ds:
        eta_theory_target = np.log(d) / (2 * T_target)

        error_at_init = cross_entropy_at_init(d)

        emp_opt_etas = []
        theo_opt_etas = []
        for T in Ts:
            errs = data[f"err_d{d}_T{T}"]
            etas = data[f"etas_T{T}"]
            emp_opt_etas.append(etas[np.argmin(errs)])
            theo_opt_etas.append(np.log(d) / (2 * T))

        print(error_at_init)
        exponent = int(np.log10(d))
        curves.append(
            {
                "d": d,
                "label": rf"$d = 10^{{{exponent}}}$",
                "normalized_etas": etas_T_target / eta_theory_target,
                "target_errors": data[f"err_d{d}_T{T_target}"] / error_at_init,
                "emp_opt_etas": np.asarray(emp_opt_etas),
                "theo_opt_etas": np.asarray(theo_opt_etas),
            }
        )

    processed["curves"] = curves
    return processed


def settings(plt):
    update_style(plt, nrows=1, ncols=2, rel_width=1.0)


def make_figure(fig, data):
    ax1, ax2 = pf.make_subplots_on_figure(fig, nrows=1, ncols=2)

    Ts = data["Ts"]
    curves = data["curves"]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(curves)))

    lines_d = []
    labels_d = []

    for curve, col in zip(curves, colors):
        (line,) = ax1.plot(
            curve["normalized_etas"],
            curve["target_errors"],
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
            color=col,
            label=curve["label"],
        )
        lines_d.append(line)
        labels_d.append(curve["label"])

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim([1e-7, 1e0])
    ax1.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
    ax1.set_xlabel(r"$\eta / \eta^*$", labelpad=0.0)
    ax1.set_ylabel("Relative error")

    leg_d = ax1.legend(
        lines_d,
        labels_d,
        loc="upper right",
        # bbox_to_anchor=(0.5, 1.3),
        frameon=False,
        handlelength=1.5,
        labelspacing=0.3,
    )
    ax1.add_artist(leg_d)

    for curve, col in zip(curves, colors):
        ax2.plot(
            Ts,
            curve["theo_opt_etas"],
            linestyle="--",
            linewidth=1.5,
            color=col,
            alpha=0.6,
        )
        ax2.plot(
            Ts,
            curve["emp_opt_etas"],
            linestyle="-",
            linewidth=1.5,
            color=col,
            markeredgecolor="black",
            markeredgewidth=0.5,
            alpha=0.9,
        )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$T$")
    ax2.set_ylabel(r"Best step-size")

    legend_elements_ax2 = [
        mlines.Line2D(
            [0],
            [0],
            color="gray",
            linestyle="--",
            lw=1.5,
            label=r"Theoretical $\eta^*$",
        ),
        mlines.Line2D([0], [0], color="gray", linestyle="-", lw=1.5, label="Empirical"),
    ]
    ax2.legend(
        handles=legend_elements_ax2, loc="upper right", frameon=False, handlelength=1.5
    )
    ax1.set_title("Grid search vs. Theory")
    ax2.set_title("Best empirical vs. Theory")

    for ax in [ax1, ax2]:
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
        print(f"Erreur : Le fichier '{DATA_FILE_SIGN}' est introuvable.")
        raise SystemExit(1)
    make_figure(fig, data)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    fig.savefig(OUT_FILE, bbox_inches="tight")
    if not args.noshow:
        plt.show()
