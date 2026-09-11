import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from package import algo_functions as f


ds_p1 = [100, 100000, 1000000]
Ts_p1 = np.unique(np.geomspace(1, 50, 20).astype(int))

save_dict = {'ds': np.array(ds_p1), 'Ts': Ts_p1}
os.makedirs('data', exist_ok=True)

for d in ds_p1:
    rel_errors = []
    for T in Ts_p1:
        T_val = int(T)
        eta = np.log(d) / (2 * T_val)
        rel_err = f.sign_descent(T_val, int(d), eta, normalized=True)
        rel_errors.append(rel_err)
    save_dict[f'rel_err_d{d}'] = np.array(rel_errors)

np.savez('data/data_sgd_big.npz', **save_dict)
