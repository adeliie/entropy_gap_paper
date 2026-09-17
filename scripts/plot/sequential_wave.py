import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from package import plot_functions as pf

pf.set_font_sizes(plt)

file_path = DATA_FILE_GD_MATRIX
data = np.load(file_path)

ds = [100, 1000, 10000]

u_targets = [0.0, 0.25, 0.5, 1]

fig, axes = pf.make_subplots(1, len(ds), ratio=2)

colors = plt.cm.viridis(np.linspace(0, 0.9, len(u_targets)))

for ax_idx, (d, ax) in enumerate(zip(ds, axes)):
    if f"d_{d}_tau" in data and f"d_{d}_err" in data:
        tau = data[f"d_{d}_tau"]
        err = data[f"d_{d}_err"]

        for u_req, col in zip(u_targets, colors):
            j = max(1, int(np.round(d**u_req)))
            idx = j - 1

            err_u = err[:, idx]
            theory_curve = pf.theory_row_specific(tau, u_req)

            ax.plot(tau, theory_curve, "--", color=col, linewidth=1.5, alpha=0.6)
            ax.plot(tau, err_u, "-", color=col, linewidth=1.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel(r"$\tau : t = d^{\tau} $", labelpad=10)
        ax.set_title(rf"$d = {d:,}$")
        ax.set_xticklabels([])
        if ax_idx == 0:
            ax.set_ylabel("relative error")


color_handles = [
    Line2D([0], [0], color=col, lw=1.5, label=rf"$u = {u}$")
    for u, col in zip(u_targets, colors)
]

style_handles = [
    Line2D([0], [0], color="gray", lw=1.5, linestyle="-", label="GD"),
    Line2D([0], [0], color="gray", lw=1.5, linestyle="--", alpha=0.8, label="Theory"),
]

spacer_handle = Line2D([0], [0], color="none", lw=0, label="")

all_handles = color_handles + [spacer_handle] + style_handles


fig.legend(
    handles=all_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.15),
    ncol=7,
    frameon=False,
    columnspacing=0.3,
    handletextpad=0.3,
    handlelength=1.5,
)

plt.tight_layout(rect=[0, 0.25, 0.98, 1])

os.makedirs("plot", exist_ok=True)
out_file = "plot/sequential_wave_gd.pdf"
plt.savefig(out_file, bbox_inches="tight")
print(f"Graphique sauvegardé sous '{out_file}'")

plt.show()
