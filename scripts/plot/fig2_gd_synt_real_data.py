import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D

from package import DATA_FILE_GD_MATRIX, DATA_FILE_GF, DATA_FILE_REAL
from package import plot_functions as pf
from package.plot_functions import update_style

OUT_FILE = "plot/limit_synthetic_vs_real.pdf"
ETA_NAME = "1_max_joint"
SYNTHETIC_DS = [100, 1000, 10000]


def load_data():
    with np.load(DATA_FILE_GD_MATRIX, allow_pickle=True) as loaded:
        gd = {key: loaded[key] for key in loaded.files}
    with np.load(DATA_FILE_GF, allow_pickle=True) as loaded:
        gf = {key: loaded[key] for key in loaded.files}
    with np.load(DATA_FILE_REAL, allow_pickle=True) as loaded:
        real = {key: loaded[key] for key in loaded.files}

    real_marginals = {}
    for d in real["ds"]:
        freqs = np.load(os.path.join("freqs", f"token_freq_total_{d}.npy"))
        real_marginals[int(d)] = freqs / np.sum(freqs)

    return {
        "gd": gd,
        "gf": gf,
        "real": real,
        "real_marginals": real_marginals,
    }


def dimension_label(d):
    exponent = int(np.log10(d)) if d in [10**i for i in range(1, 10)] else None
    exponent_str = str(exponent) if exponent == np.log10(d) else f"{np.log10(d):.1f}"
    return rf"$10^{{{exponent_str}}}$"


def postprocess(data):
    processed = dict(data)
    gd = data["gd"]
    gf = data["gf"]
    real = data["real"]

    all_ds = sorted(set(gf["ds"]).union(real["ds"]))

    synthetic_curves = []
    for d in gd["ds"]:
        if d not in SYNTHETIC_DS:
            continue

        curve = {
            "d": int(d),
            "label": dimension_label(d),
            "tau": None,
            "errors": None,
        }
        if f"d_{d}_err" in gd:
            ranks = np.arange(1, d + 1, dtype=np.float64)
            pi = (1.0 / ranks) / np.sum(1.0 / ranks)
            curve["tau"] = gd[f"d_{d}_tau"]
            curve["errors"] = np.sum(gd[f"d_{d}_err"] * pi, axis=1)

        synthetic_curves.append(curve)

    real_curves = []
    for d in real["ds"]:
        tau_key = f"d_{d}_{ETA_NAME}_tau"
        error_key = f"d_{d}_{ETA_NAME}_err"
        if tau_key not in real or error_key not in real:
            continue

        pi = data["real_marginals"][int(d)]
        real_curves.append(
            {
                "d": int(d),
                "label": dimension_label(d),
                "tau": real[tau_key],
                "errors": np.nansum(real[error_key] * pi, axis=1),
            }
        )

    theory_tau = np.linspace(0, 2.05, 500)
    processed.update(
        {
            "all_ds": all_ds,
            "synthetic_curves": synthetic_curves,
            "real_curves": real_curves,
            "theory_tau": theory_tau,
            "theory_errors": pf.theory_global(theory_tau),
        }
    )
    return processed


def settings(plt):
    update_style(plt, nrows=1, ncols=2, rel_width=1.0)


def make_figure(fig, data):
    ax1, ax2 = pf.make_subplots_on_figure(fig, nrows=1, ncols=2, sharey=True)

    colors = cm.viridis(np.linspace(0, 0.9, len(data["all_ds"])))
    color_by_d = {int(d): color for d, color in zip(data["all_ds"], colors)}

    (theory_line_1,) = ax1.plot(
        data["theory_tau"],
        data["theory_errors"],
        "--",
        color="red",
        linewidth=2.5,
        zorder=10,
    )
    (theory_line_2,) = ax2.plot(
        data["theory_tau"],
        data["theory_errors"],
        "--",
        color="red",
        linewidth=2.5,
        zorder=10,
    )

    synthetic_handles = [theory_line_1]
    synthetic_labels = ["Theory"]
    for curve in data["synthetic_curves"]:
        color = color_by_d[curve["d"]]
        if curve["tau"] is not None:
            ax1.plot(
                curve["tau"],
                curve["errors"],
                color=color,
                linestyle="-",
                linewidth=2.5,
                alpha=0.7,
            )

        synthetic_handles.append(
            Line2D([0], [0], color=color, linestyle="-", linewidth=2)
        )
        synthetic_labels.append(curve["label"])

    ax1.legend(
        synthetic_handles,
        synthetic_labels,
        loc="upper right",
        borderaxespad=0.0,
        frameon=False,
        handletextpad=0.8,
    )

    real_handles = [theory_line_2]
    real_labels = ["Theory"]
    for curve in data["real_curves"]:
        (line,) = ax2.plot(
            curve["tau"],
            curve["errors"],
            color=color_by_d[curve["d"]],
            linestyle="-",
            linewidth=2.5,
        )
        real_handles.append(line)
        real_labels.append(curve["label"])

    ax2.legend(
        real_handles,
        real_labels,
        loc="upper right",
        borderaxespad=0.0,
        frameon=False,
        handletextpad=0.8,
    )
    for ax in [ax1, ax2]:
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 1.05])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticks([0, 0.5, 1, 1.5, 2])

    ax1.set_title("Zipf's Law Frequencies")
    ax2.set_title("Frequencies from Real Data")

    for ax in [ax1, ax2]:
        ax.set_xlabel(r"$\tau: \quad  t = d^{\tau}$")
        ax.axhline(0, color="black", linewidth=0.8, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.set_ylabel("Relative error")
    # fig.subplots_adjust(bottom=0.15, right=0.95, left=0.08, wspace=0.25)
    fig.tight_layout(pad=0.5)
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
    except FileNotFoundError as error:
        print(f"Fichier introuvable : {error.filename}")
        raise SystemExit(1)
    make_figure(fig, data)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    fig.savefig(OUT_FILE, bbox_inches="tight")
    if not args.noshow:
        plt.show()
