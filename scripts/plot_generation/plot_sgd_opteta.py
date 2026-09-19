import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from package import plot_functions as pf  

if __name__ == "__main__":
    
    pf.set_font_sizes(plt)
    
    file_path = "data/data_sgd.npz"
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_path}' est introuvable.")
        exit()
   
    ds = data['ds']
    Ts = data['Ts']

    
    fig, (ax1, ax2) = pf.make_subplots(nrows=1, ncols=2, ratio = 1.61)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ds)))
    
    T_target = Ts[np.argmin(np.abs(Ts - 1000))]
    
    
    etas_T_target = data[f'etas_T{T_target}']
    
    lines_d = []
    labels_d = []
    
    for d, col in zip(ds, colors):
        errs = data[f'err_d{d}_T{T_target}']
        
       
        eta_theory = np.log(d) / (2 * T_target)
        x_vals = etas_T_target / eta_theory
        
        exponent = int(np.log10(d))
        label_d = rf"$d = 10^{{{exponent}}}$"
        
        line, = ax1.plot(x_vals, errs, linestyle='-', 
                 linewidth=1.5, alpha=0.8, color=col, label=label_d)
        lines_d.append(line)
        labels_d.append(label_d)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r"$\eta / \eta^*$")
    ax1.set_ylabel("excess loss")
    
   
    leg_d = ax1.legend(lines_d, labels_d, loc='upper left', bbox_to_anchor=(0.6, 1.1), 
                       frameon=False, handlelength=1.5, labelspacing=0.3)
    ax1.add_artist(leg_d) 


   
    for d, col in zip(ds, colors):
        emp_opt_etas = []
        theo_opt_etas = []
        
        for T in Ts:
            errs = data[f'err_d{d}_T{T}']
            etas = data[f'etas_T{T}']
            
          
            best_idx = np.argmin(errs)
            eta_empiric = etas[best_idx]
            
            eta_theory = np.log(d) / (2 * T)
            
            emp_opt_etas.append(eta_empiric)
            theo_opt_etas.append(eta_theory)

        ax2.plot(Ts, theo_opt_etas, linestyle='--', linewidth=1.5, color=col, alpha=0.6)
        ax2.plot(Ts, emp_opt_etas, linestyle='-', linewidth=1.5,
                 color=col, markeredgecolor='black', markeredgewidth=0.5, alpha=0.9)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r"$T$")
    ax2.set_ylabel(r" $\eta^*$")
    

    
    legend_elements_ax2 = [
        mlines.Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label=r"theory"),
        mlines.Line2D([0], [0], color='gray', linestyle='-', lw=1.5, label="empirical")
    ]
    ax2.legend(handles=legend_elements_ax2, loc='upper right', frameon=False, handlelength=1.5)

    
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    
    plt.tight_layout(w_pad=1.0) 
    
    out_file = "plot/sgd_eta_osc.pdf"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, bbox_inches='tight')
    
    plt.show()