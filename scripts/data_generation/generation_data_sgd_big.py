import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from package import algo_functions as f


ds_p1 = [1000000, 10000000, 100000000]
Ts_p1 = np.unique(np.geomspace(1, 50, 20).astype(int))

os.makedirs('data', exist_ok=True)
output_file = 'data/data_sgd_big.npz'

if os.path.exists(output_file):
    existing_data = np.load(output_file)
    existing_ds = existing_data['ds'].tolist()
    save_dict = {key: existing_data[key] for key in existing_data.files}
else:
    existing_ds = []
    save_dict = {'ds': np.array([], dtype=int), 'Ts': Ts_p1}

missing_ds = [d for d in ds_p1 if d not in existing_ds]

for d in missing_ds:
    print(d)
    rel_errors = []
    for T in Ts_p1:
        T_val = int(T)
        eta = np.log(d) / (2 * T_val)
        rel_err = f.sign_descent(T_val, int(d), eta, normalized=False)
        rel_errors.append(rel_err)
    save_dict[f'rel_err_d{d}'] = np.array(rel_errors)

save_dict['ds'] = np.array(existing_ds + missing_ds, dtype=int)
save_dict['Ts'] = Ts_p1
np.savez(output_file, **save_dict)
