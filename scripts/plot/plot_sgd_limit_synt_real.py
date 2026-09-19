import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from package import plot_functions as pf

pf.set_font_sizes(plt)

data_rel = np.load("data/data_sgd_big.npz", allow_pickle=True)
data_sgd = np.load("data/data_sgd.npz")
data_real = np.load("data/sign_real_data.npz", allow_pickle=True)

ds_p1 = data_rel["ds"]
Ts_p1 = data_rel["Ts"]

ds_sgd, Ts_sgd = data_sgd["ds"], data_sgd["Ts"]
ds_real, Ts_real = data_real["ds"], data_real["Ts"]

fig, (ax1, ax2, ax3) = pf.make_subplots(nrows=1, ncols=3, ratio=1.61)
colors_p1 = plt.cm.viridis(np.linspace(0.1, 0.9, len(ds_p1)))
h_p1_dims, l_p1_dims = [], []

for d, col in zip(ds_p1, colors_p1):

    rel_errors = data_rel[f"rel_err_d{d}"] * 2 / np.log(d)
    (line,) = ax1.plot(Ts_p1, rel_errors, alpha=0.8, color=col)
    h_p1_dims.append(line)
    l_p1_dims.append(rf"$10^{{{int(np.log10(d))}}}$")

d_theory_p1 = ds_p1[-1]
T_transition_p1 = np.log(d_theory_p1) / 4
T_before_transition_p1 = np.geomspace(Ts_p1.min(), T_transition_p1, 200)
T_after_transition_p1 = np.geomspace(T_transition_p1, Ts_p1.max(), 200)
(line_th1,) = ax1.plot(
    T_before_transition_p1,
    1.0 / T_before_transition_p1,
    linestyle="--",
    color="crimson",
    linewidth=2.5,
)
(line_th2,) = ax1.plot(
    T_after_transition_p1,
    np.log(d_theory_p1) / (12.0 * T_after_transition_p1**2),
    linestyle="--",
    color="crimson",
    linewidth=2.5,
)
leg_dims = ax1.legend(
    h_p1_dims,
    l_p1_dims,
    loc="upper right",
    bbox_to_anchor=(1.2, 1.2),
    frameon=False,
    fontsize=8,
    handlelength=1.5,
    labelspacing=0.15,
)
ax1.add_artist(leg_dims)
ax1.legend(
    [line_th1, line_th2],
    [r"1/T", r"$\log(d)/(12T^2)$"],
    loc="lower left",
    bbox_to_anchor=(-0.05, -0.05),
    frameon=False,
    handlelength=1.5,
)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("T")
ax1.set_ylabel("Relative Error")
ax1.set_title("Synthetic Data")


colors_p2 = plt.cm.viridis(np.linspace(0.1, 0.9, len(ds_sgd)))
h_dim, l_dim = [], []

for d, col in zip(ds_sgd, colors_p2):
    min_errs = [np.min(data_sgd[f"err_d{d}_T{T}"]) * 2 / np.log(d) for T in Ts_sgd]
    (line,) = ax2.plot(Ts_sgd, np.array(min_errs), color=col)
    h_dim.append(line)
    l_dim.append(rf"$10^{{{int(np.log10(d))}}}$")

d_theory = ds_sgd[0]
T_theory = np.geomspace(Ts_sgd.min(), Ts_sgd.max(), 200)
theory = np.log(d_theory) / (12.0 * T_theory**2)
ax2.plot(T_theory, theory, linestyle="--", color="crimson", linewidth=2.0)

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("T")
ax2.set_title("Synthetic Data")
ax2.legend(
    h_dim,
    l_dim,
    loc="upper right",
    bbox_to_anchor=(1.15, 1.15),
    frameon=False,
    handlelength=1,
    labelspacing=0.2,
)
ax2.tick_params(axis="y", labelleft=False)

colors_p3 = plt.cm.viridis(np.linspace(0.1, 0.9, len(ds_real)))
for d, col in zip(ds_real, colors_p3):
    errs, Ts_plot = [], []
    for T in Ts_real:
        if f"err_d{d}_T{T}" in data_real:
            errs.append(np.min(data_real[f"err_d{d}_T{T}"]) * 2 / np.log(d))
            Ts_plot.append(T)
    if len(Ts_plot) > 0:
        Ts_plot = np.array(Ts_plot)
        errs = np.array(errs)
        ax3.plot(Ts_plot, errs, linewidth=1.5, color=col)

d_theory_real = ds_real[0]
T_theory_real = np.geomspace(Ts_real.min(), Ts_real.max(), 200)
theory_real = np.log(d_theory_real) / (12.0 * T_theory_real**2)
ax3.plot(T_theory_real, theory_real, linestyle="--", color="crimson", linewidth=2.0)

ax3.legend(
    h_dim,
    l_dim,
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

os.makedirs("plot", exist_ok=True)
plt.savefig("plot/sgd_limit.pdf", bbox_inches="tight")
plt.show()
