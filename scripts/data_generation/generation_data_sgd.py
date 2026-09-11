import numpy as np
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from package import algo_functions as f

ds = [100, 1000, 10000]       
Ts = np.unique(np.geomspace(1, 3000, 15).astype(int))
    
t_global = time.time()


save_dict = {
    'ds': np.array(ds),
    'Ts': Ts
}

d_min, d_max = min(ds), max(ds)
for T in Ts:
    eta_min = 0.1 * np.log(d_min) / (2 * T)
    eta_max = 4.0 * np.log(d_max) / (2 * T)
    save_dict[f'etas_T{T}'] = np.geomspace(eta_min, eta_max, 50)

os.makedirs("data", exist_ok=True)
file_path = "data/data_sgd.npz"

for d in ds:
    print(f" -> Calcul pour d = {d:<7} ")
    
    for T in Ts:            
        emp_errors = [] 
        
        etas = save_dict[f'etas_T{T}']
        
        for eta in etas:
            
            err = f.sign_descent(T, d, eta, normalized=False) 
            emp_errors.append(err) 

        save_dict[f'err_d{d}_T{T}'] = np.array(emp_errors)
        
       
        np.savez(file_path, **save_dict)
