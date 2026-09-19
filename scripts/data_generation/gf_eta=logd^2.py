import numpy as np
import time

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from package import algo_functions as f


ds = [100, 1000]
results_dict = {'ds': ds}

start_time = time.time()

for d in ds:
    t0 = time.time()
    T = d**2
    print(f" -> Génération pour d = {d} (T = {T:.1e} itérations)...")
    
    t_save = np.unique(np.geomspace(1, T, 100).astype(np.int64))
    t_save = np.insert(t_save, 0, 0)
    
    err = f.gradient_flow(T, t_save, d)
        
    tau = np.zeros_like(t_save, dtype=np.float64)
    tau[1:] = np.log(t_save[1:]) / np.log(d)
    
    results_dict[f'd_{d}_tau'] = tau
    results_dict[f'd_{d}_err'] = err
    
    np.savez("data/data_gf_eta=logd^2.npz", **results_dict)
    

