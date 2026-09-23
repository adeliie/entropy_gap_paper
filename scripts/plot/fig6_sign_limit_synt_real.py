import argparse
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

    def initial_error(d):
        ranks = np.arange(1, d + 1, dtype=np.float64)
        pi = 1.0 / ranks
        pi /= pi.sum()
        return np.sum(pi * np.log(pi * d))

    relative_Ts = relative["Ts"]
    relative_curves = [
        {
            "d": d,
            "label": rf"$10^{{{int(np.log10(d))}}}$",
            "Ts": relative_Ts,
            "errors": relative[f"rel_err_d{d}"] / initial_error(d),
        }
        for d in relative["ds"]
    ]

    relative_theory_d = relative["ds"][-1]
    relative_theory_inverse_Ts = np.geomspace(1, 10, 200)
    relative_theory_quadratic_Ts = np.geomspace(2, relative_Ts.max(), 200)

    sgd_Ts = sgd["Ts"]
    sgd_curves = []
    for d in sgd["ds"]:
        errors = [np.min(sgd[f"err_d{d}_T{T}"]) / (initial_error(d)) for T in sgd_Ts]
        sgd_curves.append(
            {
                "d": d,
                "label": rf"$10^{{{int(np.log10(d))}}}$",
                "Ts": sgd_Ts,
                "tau": sgd_Ts**2 / np.log(d),
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
                errors.append(np.min(real[error_key]))
                available_Ts.append(T)

        exponent = int(np.log10(d))
        print(d, np.floor(10**exponent))
        label = rf"$10^{{{exponent}}}$" if d == np.floor(10**exponent) else rf"${d:,}$"
        exponent_str = (
            str(exponent) if exponent == np.log10(d) else f"{np.log10(d):.1f}"
        )
        label = rf"$10^{{{exponent_str}}}$"
        real_curves.append(
            {
                "d": d,
                "label": label,
                "Ts": np.asarray(available_Ts),
                "tau": np.asarray(available_Ts) ** 2 / np.log(d),
                "errors": np.asarray(errors),
            }
        )

    sgd_theory_tau = np.geomspace(
        min(curve["tau"].min() for curve in sgd_curves),
        max(curve["tau"].max() for curve in sgd_curves),
        200,
    )
    nonempty_real_curves = [curve for curve in real_curves if curve["tau"].size]
    real_theory_tau = np.geomspace(
        min(curve["tau"].min() for curve in nonempty_real_curves),
        max(curve["tau"].max() for curve in nonempty_real_curves),
        200,
    )

    processed.update(
        {
            "relative_curves": relative_curves,
            "relative_theory_inverse_Ts": relative_theory_inverse_Ts,
            "relative_theory_inverse": 1.0 / relative_theory_inverse_Ts,
            "relative_theory_quadratic_Ts": relative_theory_quadratic_Ts,
            "relative_theory_quadratic": np.log(relative_theory_d)
            / (12.0 * relative_theory_quadratic_Ts**2),
            "sgd_curves": sgd_curves,
            "sgd_theory_tau": sgd_theory_tau,
            "sgd_theory": 1.0 / (12 * sgd_theory_tau),
            "real_curves": real_curves,
            "real_theory_tau": real_theory_tau,
            "real_theory": 1.0 / (12 * real_theory_tau),
        }
    )
    return processed


def settings(plt):
    update_style(plt, nrows=1, ncols=3, rel_width=1.0, height_to_width_ratio=0.8)


def make_figure(fig, data):
    ax1, ax2, ax3 = pf.make_subplots_on_figure(fig, nrows=1, ncols=3, sharey=False)

    relative_curves = data["relative_curves"]
    relative_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(relative_curves)))
    relative_handles = []
    relative_labels = []

    large_d_synth = [1e5, 1e6, 1e7, 1e8]

    for curve, color in zip(relative_curves, relative_colors):
        if curve["d"] not in large_d_synth:
            continue
        (line,) = ax1.plot(curve["Ts"], curve["errors"], alpha=0.8, color=color)
        relative_handles.append(line)
        relative_labels.append(curve["label"])

    (theory_line_inverse,) = ax1.plot(
        data["relative_theory_inverse_Ts"],
        data["relative_theory_inverse"],
        linestyle="dotted",
        color="crimson",
    )
    (theory_line_quadratic,) = ax1.plot(
        data["relative_theory_quadratic_Ts"],
        data["relative_theory_quadratic"],
        linestyle="--",
        color="crimson",
    )
    dimension_legend = ax1.legend(
        relative_handles,
        relative_labels,
        loc="lower left",
        # bbox_to_anchor=(1.2, 1.2),
        borderaxespad=0.0,
        labelspacing=0.2,
        frameon=False,
    )
    ax1.add_artist(dimension_legend)
    ax1.legend(
        [theory_line_inverse, theory_line_quadratic],
        [r"$\frac{1}{T}$", r"$\frac{\log(d)}{12T^2}$"],
        # bbox_to_anchor=(-0.05, -0.05),
        loc="upper right",
        borderaxespad=0.0,
        labelspacing=0.2,
        frameon=False,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("T")
    ax1.set_ylabel("Relative Error")
    ax1.set_title("Transition")
    ax1.set_ylim([10**-4.5, 1.5 * 1e0])
    ax1.set_yticks([1e-4, 1e-2, 1e0])

    sgd_curves = data["sgd_curves"]
    sgd_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sgd_curves)))
    sgd_handles = []
    sgd_labels = []

    for curve, color in zip(sgd_curves, sgd_colors):
        (line,) = ax2.plot(curve["tau"], curve["errors"], color=color)
        sgd_handles.append(line)
        sgd_labels.append(curve["label"])

    (sgd_theory_line,) = ax2.plot(
        data["sgd_theory_tau"],
        data["sgd_theory"],
        linestyle="--",
        color="crimson",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\tau = T^2 / \log(d)$")
    ax2.set_title(r"$T \gg \log(d)$")
    sgd_dimension_legend = ax2.legend(
        sgd_handles,
        sgd_labels,
        loc="lower left",
        # bbox_to_anchor=(1.15, 1.15),
        borderaxespad=0.0,
        labelspacing=0.2,
        frameon=False,
    )
    ax2.add_artist(sgd_dimension_legend)
    ax2.legend(
        [sgd_theory_line],
        [r"$\frac{1}{12\tau}$"],
        loc="upper right",
        borderaxespad=0.0,
        labelspacing=0.2,
        frameon=False,
    )
    ax2.set_ylim([1e-7, 1.5 * 1e0])
    ax2.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
    # ax2.tick_params(axis="y", labelleft=False)

    real_curves = data["real_curves"]
    real_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(real_curves)))
    real_handles = []
    real_labels = []
    for curve, color in zip(real_curves, real_colors):
        if curve["tau"].size:
            (line,) = ax3.plot(
                curve["tau"], curve["errors"], linewidth=1.5, color=color
            )
            real_handles.append(line)
            real_labels.append(curve["label"])

    (real_theory_line,) = ax3.plot(
        data["real_theory_tau"],
        data["real_theory"],
        linestyle="--",
        color="crimson",
    )
    real_dimension_legend = ax3.legend(
        real_handles,
        real_labels,
        loc="lower left",
        # bbox_to_anchor=(1.15, 1.15),
        borderaxespad=0.0,
        labelspacing=0.2,
        frameon=False,
    )
    ax3.add_artist(real_dimension_legend)
    ax3.legend(
        [real_theory_line],
        [r"$\frac{1}{12\tau}$"],
        loc="upper right",
        borderaxespad=0.0,
        labelspacing=0.2,
        frameon=False,
    )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel(r"$\tau = T^2 / \log(d)$")
    ax3.set_title("Real Data")
    # ax3.tick_params(axis="y", labelleft=False)
    ax3.set_ylim([1e-7, 1.5 * 1e0])
    ax3.set_yticks([1e-6, 1e-4, 1e-2, 1e0])

    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

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
