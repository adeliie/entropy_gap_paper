import numpy as np
import time
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from package import algo_functions as f

if __name__ == "__main__":
    ds = [1000, 3162, 10000]       
    Ts = np.unique(np.geomspace(1, 3000, 15).astype(int))
        
    t_global = time.time()

    save_dict = {
        'ds': np.array(ds),
        'Ts': Ts
    }

    os.makedirs("data", exist_ok=True)
    file_path = "data/sign_real_data.npz"

    for d in ds:
        print(f" -> Calcul pour d = {d:<7} ")
        t_d = time.time()
        
        for T in Ts:            
           
            eta_opt = np.log(d) / (2 * T)
            
            multipliers = np.geomspace(0.1, 10.0, 15)
            etas_to_test = eta_opt * multipliers
            
            emp_errors = []
            
            for eta in etas_to_test:
                err = f.sign_descent_real_data(vocab_size=d, T=T, eta=eta) 
                emp_errors.append(err)

            save_dict[f'err_d{d}_T{T}'] = np.array(emp_errors)
            save_dict[f'etas_d{d}_T{T}'] = etas_to_test
            
            np.savez(file_path, **save_dict)
            
        print(f"    [d={d}] terminé en {time.time() - t_d:.1f}s")
        