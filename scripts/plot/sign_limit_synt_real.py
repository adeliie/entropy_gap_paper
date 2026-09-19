import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from package import DATA_FILE_REAL_SIGN, DATA_FILE_SIGN, DATA_FILE_SIGN_BIG
from package import plot_functions as pf
from package.plot_functions import update_style

OUT_FILE = "plot/sgd_limit.pdf"


def load_data():
    with np.load(DATA_FILE_SIGN_BIG, allow_pickle=True) as loaded:
        relative = {key: loaded[key] for key in loaded.files}
    with np.load(DATA_FILE_SIGN) as loaded:
        sgd = {key: loaded[key] for key in loaded.files}
    with np.load(DATA_FILE_REAL_SIGN, allow_pickle=True) as loaded:
        real = {key: loaded[key] for key in loaded.files}

    return {"relative": relative, "sgd": sgd, "real": real}


def postprocess(data):
    processed = dict(data)
    relative = data["relative"]
    sgd = data["sgd"]
    real = data["real"]

    relative_Ts = relative["Ts"]
    relative_curves = [
        {
            "d": d,
            "label": rf"$10^{{{int(np.log10(d))}}}$",
            "Ts": relative_Ts,
            "errors": relative[f"rel_err_d{d}"],
        }
        for d in relative["ds"]
    ]

    sgd_Ts = sgd["Ts"]
    sgd_curves = []
    for d in sgd["ds"]:
        errors = [np.min(sgd[f"err_d{d}_T{T}"]) * 2 / np.log(d) for T in sgd_Ts]
        sgd_curves.append(
            {
                "d": d,
                "label": rf"$10^{{{int(np.log10(d))}}}$",
                "Ts": sgd_Ts,
                "errors": np.asarray(errors),
            }
        )

    real_Ts = real["Ts"]
    real_curves = []
    for d in real["ds"]:
        errors = []
        available_Ts = []
        for T in real_Ts:
            error_key = f"err_d{d}_T{T}"
            if error_key in real:
                errors.append(np.min(real[error_key]) * 2 / np.log(d))
                available_Ts.append(T)

        real_curves.append(
            {
                "d": d,
                "Ts": np.asarray(available_Ts),
                "errors": np.asarray(errors),
            }
        )

    relative_theory_Ts = relative_Ts
    sgd_theory_Ts = np.geomspace(sgd_Ts.min(), sgd_Ts.max(), 200)
    real_theory_Ts = np.geomspace(real_Ts.min(), real_Ts.max(), 200)

    processed.update(
        {
            "relative_curves": relative_curves,
            "relative_theory_Ts": relative_theory_Ts,
            "relative_theory": 1.0 / relative_theory_Ts,
            "sgd_curves": sgd_curves,
            "sgd_theory_Ts": sgd_theory_Ts,
            "sgd_theory": np.log(sgd["ds"][0]) / (12.0 * sgd_theory_Ts**2),
            "real_curves": real_curves,
            "real_theory_Ts": real_theory_Ts,
            "real_theory": np.log(real["ds"][0]) / (12.0 * real_theory_Ts**2),
        }
    )
    return processed


def settings(plt):
    update_style(plt, nrows=1, ncols=3, rel_width=1.0)


def make_figure(fig, data):
    ax1, ax2, ax3 = pf.make_subplots_on_figure(fig, nrows=1, ncols=3)

    relative_curves = data["relative_curves"]
    relative_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(relative_curves)))
    relative_handles = []
    relative_labels = []

    for curve, color in zip(relative_curves, relative_colors):
        (line,) = ax1.plot(curve["Ts"], curve["errors"], alpha=0.8, color=color)
        relative_handles.append(line)
        relative_labels.append(curve["label"])

    (relative_theory_line,) = ax1.plot(
        data["relative_theory_Ts"],
        data["relative_theory"],
        linestyle="--",
        color="crimson",
        linewidth=2.0,
    )
    dimension_legend = ax1.legend(
        relative_handles,
        relative_labels,
        loc="upper right",
        bbox_to_anchor=(1.2, 1.2),
        frameon=False,
        fontsize=8,
        handlelength=1.5,
        labelspacing=0.15,
    )
    ax1.add_artist(dimension_legend)
    ax1.legend(
        [relative_theory_line],
        [r"1/T"],
        loc="lower left",
        bbox_to_anchor=(0.05, 0.0),
        frameon=False,
        handlelength=1.5,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("T")
    ax1.set_ylabel("Relative Error")
    ax1.set_title("Synthetic Data")

    sgd_curves = data["sgd_curves"]
    sgd_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sgd_curves)))
    sgd_handles = []
    sgd_labels = []

    for curve, color in zip(sgd_curves, sgd_colors):
        (line,) = ax2.plot(curve["Ts"], curve["errors"], color=color)
        sgd_handles.append(line)
        sgd_labels.append(curve["label"])

    ax2.plot(
        data["sgd_theory_Ts"],
        data["sgd_theory"],
        linestyle="--",
        color="crimson",
        linewidth=2.0,
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("T")
    ax2.set_title("Synthetic Data")
    ax2.tick_params(axis="y", labelleft=False)

    real_curves = data["real_curves"]
    real_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(real_curves)))
    for curve, color in zip(real_curves, real_colors):
        if curve["Ts"].size:
            ax3.plot(curve["Ts"], curve["errors"], linewidth=1.5, color=color)

    ax3.plot(
        data["real_theory_Ts"],
        data["real_theory"],
        linestyle="--",
        color="crimson",
        linewidth=2.0,
    )
    ax3.legend(
        sgd_handles,
        sgd_labels,
        loc="upper right",
        bbox_to_anchor=(1.15, 1.15),
        frameon=False,
        handlelength=1,
        labelspacing=0.2,
    )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("T")
    ax3.set_title("Real Data")
    ax3.tick_params(axis="y", labelleft=False)

    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

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
