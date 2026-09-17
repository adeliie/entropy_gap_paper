import os
import time

import numpy as np
from tqdm import tqdm

from package import DATA_FILE_OPTETA
from package import algo_functions as f

ds = [100, 1000, 10000]
results_dict = {"all_ds": []}

for d in tqdm(ds, total=len(ds), leave=True):
    T = d
    start_time = time.time()
    tqdm.write(f"Calcul pour d={d:5d}...", end=" ")

    pi = f.generate_data(d)
    pi = pi.cpu().numpy()

    eta_min = max(0.01, (np.log(d) ** 2) / 100.0)
    eta_max = min(1000.0, (np.log(d) ** 2) * 10.0)

    etas = np.logspace(np.log10(eta_min), np.log10(eta_max), 50)

    errors_gd = np.empty(len(etas))

    for idx, eta in tqdm(enumerate(etas), total=len(etas)):

        err_matrix = f.gd_matrix_per_col(T, d, eta)[1]

        err_final = err_matrix[-1]

        errors_gd[idx] = np.sum(err_final * pi)

    results_dict["all_ds"].append(d)
    results_dict[f"d_{d}_etas"] = etas
    results_dict[f"d_{d}_errors_gf"] = errors_gd
    elapsed = time.time() - start_time

    tqdm.write(f"terminé en {elapsed:.2f}s")

os.makedirs("data", exist_ok=True)
np.savez(DATA_FILE_OPTETA, **results_dict)
