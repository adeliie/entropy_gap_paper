import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from package import algo_functions as f

ds = [100, 500, 1000, 10000]
file_path = "data/data_gd_matrix.npz"


results_dict = {'ds': ds}
if os.path.exists(file_path):
    print(" -> Fichier existant trouvé, chargement des résultats actuels...")
    old_data = np.load(file_path)
    for key in old_data.files:
        results_dict[key] = old_data[key]
else:
    print(" -> Aucun fichier existant, démarrage d'une nouvelle simulation.")

start_time = time.time()

for d in ds:
   
    if f'd_{d}_err' in results_dict:
        print(f" -> d = {d} déjà calculé. Passage au suivant.")
        continue
        
    t0 = time.time()
    T = int(d**(1.6))

    print(f" -> Génération pour d = {d} (T = {T:.1e} itérations)...")

    t_save, err = f.gd_matrix_per_col(T, d, 0)  
    tau = np.zeros_like(t_save)
    tau[1:] = np.log(t_save[1:]) / np.log(d)
    results_dict[f'd_{d}_tau'] = tau
    results_dict[f'd_{d}_err'] = err   
        
    os.makedirs("data", exist_ok=True)
    np.savez(file_path, **results_dict)
    
    print(f"    Terminé en {time.time() - t0:.2f}s. Données sauvegardées.")

