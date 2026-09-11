
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from package import plot_functions as pf
from package import algo_functions as f

pf.set_font_sizes(plt)

fichier = 'data/data_gd_opteta.npz'

try:
    data = np.load(fichier)
except FileNotFoundError:
    print(f"Fichier '{fichier}' introuvable.")
    exit()

all_ds = data['all_ds']

fig = pf.make_figure(rel_width=0.9, ratio = 1.9)
ax = fig.add_subplot(111)

colors = cm.viridis(np.linspace(0, 0.9, len(all_ds)))

for idx, d in enumerate(all_ds):
    try:
        etas_d = data[f'd_{d}_etas']
        exponent = int(np.log10(d)) if d in [10**i for i in range(1, 10)] else None
        label_d = rf" $d=10^{{{exponent}}}$" if exponent else rf"$d={d:,}$"
        erreurs_d_gf = data[f'd_{d}_errors_gf']
        
        x_universel =  etas_d/(np.log(d)**2)
        
        ax.plot(x_universel, erreurs_d_gf, linestyle='-', 
                linewidth=2, alpha=0.8, color=colors[idx], label= label_d)
        
    except KeyError:
        print(f"Données manquantes pour d={d}")

ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='best', frameon=False)
ax.set_xlabel(r" $\eta / \log d^2$")
ax.set_ylabel("relative error")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("plot/optimal_eta_gd.pdf", bbox_inches='tight')
plt.show()