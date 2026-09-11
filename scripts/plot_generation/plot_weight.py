import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from package import plot_functions as pf  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pf.set_font_sizes(plt)

def simulate_sign_descent_3panels(d=10000, T=3000, eta=0.05):
    ranks = torch.arange(1, d + 1, dtype=torch.float64, device=device)
    pi = 1.0 / ranks
    pi /= torch.sum(pi)
    log_pi = torch.log(pi)

    W = torch.full((d,), -np.log(d), dtype=torch.float64, device=device)
    
    track_indices = [49, 99, 1999] 
    
    W_traj = {k: np.zeros(T) for k in track_indices}
    target_traj = {k: np.zeros(T) for k in track_indices}
    
    delta_history_window = [] 
    
    with torch.no_grad():
        for t in range(T):
           
            P = F.softmax(W, dim=0)
            Z = torch.sum(torch.exp(W))

            c = torch.mean(W - log_pi)
            
            delta = W - (log_pi + c)
            
            for k in track_indices:
                W_traj[k][t] = W[k].item()
                target_traj[k][t] = (log_pi[k] + torch.log(Z)).item()
            
            if t >= T - 30:
                delta_history_window.append(delta.cpu().numpy())
            
            W += eta * torch.sign(pi - P)
        
        delta_history = np.array(delta_history_window)
        
        deltas = delta_history[-1:].flatten()
        
    return W_traj, target_traj, deltas, track_indices

if __name__ == "__main__":
    d, T, eta = 10000, 3000, 0.05
    W_traj, target_traj, deltas, track_indices = simulate_sign_descent_3panels(d, T, eta)
    
   
    fig, (ax1, ax2, ax3) = pf.make_subplots(nrows=1, ncols=3, ratio=1.61)
    pf.set_font_sizes(plt)
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(track_indices)))
    

    lines, labels = [], []
    for idx, (k, col) in enumerate(zip(track_indices, colors)):
        line, = ax1.plot(W_traj[k], color=col, alpha=0.8, lw=1.5, label=rf"$w_{{{k+1}}}$")
        ax1.plot(target_traj[k], color=col, linestyle='--', alpha=0.5, lw=2.5)
        lines.append(line)
        labels.append(rf"$w_{{{k+1}}}$")
        
    ax1.set_xlim(0, 200)
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$w_k(t)$ and target")
    ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.05, 1.35), frameon=False)

    for idx, (k, col) in enumerate(zip(track_indices, colors)):
        delta_k = W_traj[k] - target_traj[k]
        ax2.plot(delta_k, color=col, alpha=0.5, lw=1)
    
    
    ax2.axhline(2 * eta, color='black', linestyle='--', alpha=0.5, lw=1, zorder=1)
    ax2.axhline(-2 * eta, color='black', linestyle='--', alpha=0.5, lw=1, zorder=1)
    
    ax2.fill_between([0, T], -2 * eta, 2 * eta, color='red', alpha=0.1, zorder=0)

    ax2.set_title(r"Residuals $res_k(t)$")
    ax2.set_xlabel("t")
    ax2.set_xlim(0, 200) 
    ax2.set_ylim(-4 * eta, 4 * eta) 

    ax3.hist(deltas, bins=60, density=True, alpha=0.8, edgecolor='white') 
   
    rect_uniform = patches.Rectangle((-eta, 0), 2*eta, 1/(2*eta), linewidth=1.5, edgecolor='black', 
                                     facecolor='none', linestyle='--', zorder=10)
    ax3.add_patch(rect_uniform)
    
   
    theory_handle = mlines.Line2D([], [], color='black', linestyle='--', lw=1.5, label=r"Uniform $[-\eta, \eta]$")
    ax3.legend(handles=[theory_handle], loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False)

    ax3.set_title(r"Distribution of $\delta_k$")
    ax3.set_xlabel(r"$\delta_k$")
    
    ax3.set_xlim(-eta - 0.03, eta + 0.03)

    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major')

  
    os.makedirs("plot", exist_ok=True)
    plt.savefig("plot/sgd_weight.pdf", bbox_inches='tight')
    
    plt.show()