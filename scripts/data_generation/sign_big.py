import os

import numpy as np
from tqdm import tqdm

from package import DATA_FILE_SIGN_BIG
from package import algo_functions as f

ds_p1 = [100, 100000, 1000000]
Ts_p1 = np.unique(np.geomspace(1, 50, 20).astype(int))

save_dict = {"ds": np.array(ds_p1), "Ts": Ts_p1}
os.makedirs("data", exist_ok=True)

for d in tqdm(ds_p1, total=len(ds_p1), leave=True, desc="Dimension"):
    rel_errors = []
    for T in tqdm(Ts_p1, total=len(Ts_p1), leave=True, desc="T"):
        T_val = int(T)
        eta = np.log(d) / (2 * T_val)
        rel_err = f.sign_descent(T_val, int(d), eta, normalized=True)
        rel_errors.append(rel_err)
    save_dict[f"rel_err_d{d}"] = np.array(rel_errors)

np.savez(DATA_FILE_SIGN_BIG, **save_dict)
