import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from package import DATA_FILE_REAL
from package import algo_functions as f

ds = [1000, 3162, 10000]
file_path = DATA_FILE_REAL


eta_names = ["1_max_joint"]

results_dict = {"ds": ds, "eta_names": eta_names}

if os.path.exists(file_path):
    print(" -> Fichier existant trouvé, chargement des résultats actuels...")
    old_data = np.load(file_path, allow_pickle=True)
    for key in old_data.files:
        if key not in ["ds", "eta_names"]:
            results_dict[key] = old_data[key]
else:
    print(" -> Aucun fichier existant, démarrage d'une nouvelle simulation.")

start_time = time.time()

for d in tqdm(ds, total=len(ds), desc="Dimensions"):
    t0 = time.time()
    T = int(d ** (1.6))

    tqdm.write(f" -> Génération pour d = {d} (T = {T:.1e} itérations)...")

    eta_name = "1_max_joint"

    if f"d_{d}_{eta_name}_err" in results_dict:
        tqdm.write(f"    -> eta = {eta_name} déjà calculé. Passage au suivant.")
    else:
        tqdm.write(f"    -> Calcul pour eta = {eta_name} ")

        saved_t, err_matrix = f.gradient_real_data(
            vocab_size=d, T=T, eta=0, N_checkpoints=100
        )

        tau = np.zeros_like(saved_t, dtype=np.float64)
        tau[1:] = np.log(saved_t[1:]) / np.log(d)

        results_dict[f"d_{d}_{eta_name}_tau"] = tau
        results_dict[f"d_{d}_{eta_name}_err"] = err_matrix

        os.makedirs("data", exist_ok=True)
        np.savez(file_path, **results_dict)

        tqdm.write(f"       Terminé pour {eta_name}. Données sauvegardées.")

    tqdm.write(f" -> Fin pour d={d} en {time.time() - t0:.2f}s.\n")
