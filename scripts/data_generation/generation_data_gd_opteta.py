import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from package import algo_functions as f

ds = [100, 1000, 10000]
results_dict = {'all_ds': []}

for d in ds:
    T = d
    start_time = time.time()
    print(f"Calcul pour d={d:5d}...", end=" ", flush=True)
        
    pi = f.generate_data(d)
    pi = pi.cpu().numpy()

    eta_min = max(0.01, (np.log(d) ** 2) / 100.0)
    eta_max = min(1000.0, (np.log(d) ** 2) * 10.0)
        
    etas = np.logspace(np.log10(eta_min), np.log10(eta_max), 50)

    errors_gd = np.empty(len(etas))

    for idx, eta in enumerate(etas):
        
        err_matrix = f.gd_matrix_per_col(T, d, eta)[1]
    
        err_final = err_matrix[-1]
        
        errors_gd[idx] = np.sum(err_final * pi)
        
    results_dict['all_ds'].append(d)
    results_dict[f'd_{d}_etas'] = etas
    results_dict[f'd_{d}_errors_gf'] = errors_gd
    elapsed = time.time() - start_time

    print(f"terminé en {elapsed:.2f}s")

os.makedirs("data", exist_ok=True)
np.savez("data/data_gd_opteta.npz", **results_dict)