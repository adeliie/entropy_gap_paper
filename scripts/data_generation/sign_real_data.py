import os
import time

import numpy as np
from tqdm import tqdm

from package import DATA_FILE_REAL_SIGN
from package import algo_functions as f

if __name__ == "__main__":
    ds = [1000, 3162, 10000]
    Ts = np.unique(np.geomspace(1, 3000, 15).astype(int))

    t_global = time.time()

    save_dict = {"ds": np.array(ds), "Ts": Ts}

    os.makedirs("data", exist_ok=True)
    file_path = DATA_FILE_REAL_SIGN

    for d in tqdm(ds, total=len(ds), desc="Dimension"):
        tqdm.write(f" -> Calcul pour d = {d:<7} ")
        t_d = time.time()

        for T in Ts:

            eta_opt = np.log(d) / (2 * T)

            multipliers = np.geomspace(0.1, 10.0, 15)
            etas_to_test = eta_opt * multipliers

            emp_errors = []

            for eta in etas_to_test:
                err = f.sign_descent_real_data(vocab_size=d, T=T, eta=eta)
                emp_errors.append(err)

            save_dict[f"err_d{d}_T{T}"] = np.array(emp_errors)
            save_dict[f"etas_d{d}_T{T}"] = etas_to_test

            np.savez(file_path, **save_dict)

        tqdm.write(f"    [d={d}] terminé en {time.time() - t_d:.1f}s")
