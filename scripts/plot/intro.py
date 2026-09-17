import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from package import DATA_FILE_INTRO
from package import plot_functions as pf

pf.set_font_sizes(plt)

OUT_FILE = "plot/optim_quad_vs_ce.pdf"

COLORS = {"gd": "royalblue", "sd": "crimson", "adam": "darkorange"}
LABELS = {"gd": "GD", "sd": "SD", "adam": "Adam"}

fig, axes = pf.make_subplots(nrows=1, ncols=2, ratio=1.6)
ax1, ax2 = axes
ax2.sharey(ax1)


data = np.load(DATA_FILE_INTRO)
mask = data["eval_steps"] <= 100

for algo_key, color in COLORS.items():
    x = data["eval_steps"][mask]
    y_q = data[f"err_q_{algo_key}"][mask]
    y_ce = data[f"err_ce_{algo_key}"][mask]
    ax1.plot(x, y_q, color=color, lw=2, label=LABELS[algo_key])
    ax2.plot(x, y_ce, color=color, lw=2)

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
    mlines.Line2D([0], [0], color=c, lw=2, label=LABELS[k]) for k, c in COLORS.items()
]
ax1.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=7)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
plt.savefig(OUT_FILE, bbox_inches="tight")
plt.show()
