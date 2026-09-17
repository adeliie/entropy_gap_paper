import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D

from package import (
    DATA_FILE_GD_MATRIX,
    DATA_FILE_GF,
    DATA_FILE_REAL,
)
from package import algo_functions as f
from package import (
    plot_functions as pf,
)

pf.set_font_sizes(plt)


data_gd = np.load(DATA_FILE_GD_MATRIX, allow_pickle=True)
ds_gd = data_gd["ds"]
data_gf = np.load(DATA_FILE_GF, allow_pickle=True)
ds_gf = data_gf["ds"]


file_path_real = DATA_FILE_REAL
data_real = np.load(file_path_real, allow_pickle=True)
ds_real = data_real["ds"]
eta_name = "1_max_joint"


all_ds = sorted(list(set(ds_gf).union(set(ds_real))))
shared_colors = cm.viridis(np.linspace(0, 0.8, len(all_ds)))
color_dict = {d: col for d, col in zip(all_ds, shared_colors)}

fig, (ax1, ax2) = pf.make_subplots(1, 2, ratio=1.61)

handles_ax1 = []
labels_ax1 = []


tau_theory = np.linspace(0, 2.05, 500)
if hasattr(pf, "theory_global"):
    R_th_glob = pf.theory_global(tau_theory)
    (line_th1,) = ax1.plot(
        tau_theory, R_th_glob, "--", color="red", linewidth=2.5, zorder=10
    )
    (line_th2,) = ax2.plot(
        tau_theory, R_th_glob, "--", color="red", linewidth=2.5, zorder=10
    )
    handles_ax1.append(line_th1)
    labels_ax1.append("Theory")


for d in ds_gf:
    if d in [100, 1000, 10000]:
        col = color_dict[d]

        tau_gf = data_gf[f"d_{d}_tau"]
        exponent = int(np.log10(d)) if d in [10**i for i in range(1, 10)] else None
        label_d = rf"$10^{{{exponent}}}$" if exponent else rf"${d:,}$"

        pi_synth = f.generate_data(d).cpu().numpy()
        has_gd = f"d_{d}_err" in data_gd

        if has_gd:
            tau_gd = data_gd[f"d_{d}_tau"]
            err_gd = np.sum(data_gd[f"d_{d}_err"] * pi_synth, axis=1)
            ax1.plot(tau_gd, err_gd, color=col, linestyle="-", linewidth=2.5, alpha=0.7)

        dummy_line = Line2D([0], [0], color=col, linestyle="-", linewidth=2)
        handles_ax1.append(dummy_line)
        labels_ax1.append(label_d)

ax1.legend(
    handles_ax1,
    labels_ax1,
    loc="upper right",
    borderaxespad=0.0,
    frameon=False,
    handletextpad=0.8,
)


handles_ax2 = [line_th2]
labels_ax2 = ["Theory"]

for d in ds_real:
    col = color_dict[d]

    exponent = int(np.log10(d)) if d in [10**i for i in range(1, 10)] else None
    label_d = rf"$10^{{{exponent}}}$" if exponent else rf"${d:,}$"

    tau_key = f"d_{d}_{eta_name}_tau"
    err_key = f"d_{d}_{eta_name}_err"

    if tau_key in data_real and err_key in data_real:
        tau_real = data_real[tau_key]
        err_matrix = data_real[err_key]

        freqs, _ = f.load_freqs(d)
        pi_real = freqs / np.sum(freqs)

        err_global_real = np.nansum(err_matrix * pi_real, axis=1)

        (line_real,) = ax2.plot(
            tau_real, err_global_real, color=col, linestyle="-", linewidth=2.5
        )
        handles_ax2.append(line_real)
        labels_ax2.append(label_d)

ax1.set_title("Synthetic Data")
ax2.set_title("Real Data")

for ax in [ax1, ax2]:
    ax.set_xlabel(r"$\tau: \quad  t = d^{\tau}$")
    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax1.set_ylabel("relative error")

plt.subplots_adjust(bottom=0.15, right=0.95, left=0.08, wspace=0.25)

os.makedirs("plot", exist_ok=True)
plt.savefig("plot/limit_synthetic_vs_real.pdf", bbox_inches="tight")

plt.show()
