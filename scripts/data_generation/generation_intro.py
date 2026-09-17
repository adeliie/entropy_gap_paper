import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from package import DATA_FILE_INTRO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

d = 500
T_steps = 100
eval_interval = 10
num_etas = 50
filename = DATA_FILE_INTRO

pi = torch.arange(1, d + 1, dtype=DTYPE, device=device)
pi = (1.0 / pi) / (1.0 / pi).sum()
pi_j = pi.unsqueeze(0)
Pi_mat = pi.unsqueeze(1).expand(-1, pi.numel())
entropy = -torch.sum(pi * torch.log(pi + 1e-30)).item()

W0_ce = torch.full((d, d), -np.log(d), dtype=DTYPE, device=device)
W0_q = torch.zeros((d, d), dtype=DTYPE, device=device)

init_ce_err = (
    -torch.sum(pi_j * Pi_mat * torch.log(F.softmax(W0_ce, dim=0) + 1e-30)).item()
) - entropy
init_q_err = 0.5 * torch.sum(pi_j * (W0_q - Pi_mat) ** 2).item()

eta_grid_ce_gd = np.geomspace(1e-1, 1e4, num_etas)
eta_grid_q_gd = np.geomspace(1e-2, 1e3, num_etas)
W_ce_gd_list = [W0_ce.clone() for _ in range(num_etas)]
W_q_gd_list = [W0_q.clone() for _ in range(num_etas)]

eta_grid_ce_sd = np.geomspace(1e-4, 0.5, num_etas)
eta_grid_q_sd = np.geomspace(1e-6, 0.1, num_etas)
W_ce_sd_list = [W0_ce.clone() for _ in range(num_etas)]
W_q_sd_list = [W0_q.clone() for _ in range(num_etas)]

eta_grid_ce_adam = np.geomspace(1e-4, 1.0, num_etas)
eta_grid_q_adam = np.geomspace(1e-4, 1.0, num_etas)
W_ce_adam_list = [torch.nn.Parameter(W0_ce.clone()) for _ in range(num_etas)]
W_q_adam_list = [torch.nn.Parameter(W0_q.clone()) for _ in range(num_etas)]

opt_ce_adam_list = [
    torch.optim.Adam([p], lr=eta, betas=(0.0, 0.999), eps=1e-15)
    for p, eta in zip(W_ce_adam_list, eta_grid_ce_adam)
]
opt_q_adam_list = [
    torch.optim.Adam([p], lr=eta, betas=(0.0, 0.999), eps=1e-15)
    for p, eta in zip(W_q_adam_list, eta_grid_q_adam)
]

hist_steps = []
num_evals = len(set(range(0, T_steps + 1, eval_interval)))
hist_all = {
    "ce_gd": np.zeros((num_etas, num_evals)),
    "q_gd": np.zeros((num_etas, num_evals)),
    "ce_sd": np.zeros((num_etas, num_evals)),
    "q_sd": np.zeros((num_etas, num_evals)),
    "ce_adam": np.zeros((num_etas, num_evals)),
    "q_adam": np.zeros((num_etas, num_evals)),
}

with torch.no_grad():
    eval_idx = 0
    for k in tqdm(range(T_steps + 1), total=T_steps + 1, desc="Time"):
        if k in set(range(0, T_steps + 1, eval_interval)):
            hist_steps.append(k)
            for i, W in enumerate(W_ce_gd_list):
                P = F.softmax(W, dim=0)
                hist_all["ce_gd"][i, eval_idx] = (
                    -torch.sum(pi_j * Pi_mat * torch.log(P + 1e-30)).item() - entropy
                ) / init_ce_err
            for i, W in enumerate(W_q_gd_list):
                hist_all["q_gd"][i, eval_idx] = (
                    0.5 * torch.sum(pi_j * (W - Pi_mat) ** 2).item()
                ) / init_q_err
            for i, W in enumerate(W_ce_sd_list):
                P = F.softmax(W, dim=0)
                hist_all["ce_sd"][i, eval_idx] = (
                    -torch.sum(pi_j * Pi_mat * torch.log(P + 1e-30)).item() - entropy
                ) / init_ce_err
            for i, W in enumerate(W_q_sd_list):
                hist_all["q_sd"][i, eval_idx] = (
                    0.5 * torch.sum(pi_j * (W - Pi_mat) ** 2).item()
                ) / init_q_err
            for i, p in enumerate(W_ce_adam_list):
                P = F.softmax(p, dim=0)
                hist_all["ce_adam"][i, eval_idx] = (
                    -torch.sum(pi_j * Pi_mat * torch.log(P + 1e-30)).item() - entropy
                ) / init_ce_err
            for i, p in enumerate(W_q_adam_list):
                hist_all["q_adam"][i, eval_idx] = (
                    0.5 * torch.sum(pi_j * (p - Pi_mat) ** 2).item()
                ) / init_q_err
            eval_idx += 1

        if k == T_steps:
            break

        for W, eta in zip(W_ce_gd_list, eta_grid_ce_gd):
            P = F.softmax(W, dim=0)
            grad = pi_j * (P - Pi_mat)
            W.sub_(eta * grad)
        for W, eta in zip(W_q_gd_list, eta_grid_q_gd):
            grad = pi_j * (W - Pi_mat)
            W.sub_(eta * grad)

        for W, eta in zip(W_ce_sd_list, eta_grid_ce_sd):
            P = F.softmax(W, dim=0)
            grad = pi_j * (P - Pi_mat)
            W.sub_(eta * grad.sign())
        for W, eta in zip(W_q_sd_list, eta_grid_q_sd):
            grad = pi_j * (W - Pi_mat)
            W.sub_(eta * grad.sign())

        for p, opt in zip(W_ce_adam_list, opt_ce_adam_list):
            P = F.softmax(p, dim=0)
            grad = pi_j * (P - Pi_mat)
            p.grad = grad.clone()
            opt.step()
        for p, opt in zip(W_q_adam_list, opt_q_adam_list):
            grad = pi_j * (p - Pi_mat)
            p.grad = grad.clone()
            opt.step()

best_curves = {}
for key in hist_all:
    if "sd" in key or "adam" in key:
        best_curves[key] = np.min(hist_all[key], axis=0)
    else:
        best_eta_idx = np.argmin(hist_all[key][:, -1])
        best_curves[key] = hist_all[key][best_eta_idx, :]

np.savez_compressed(
    filename,
    d=d,
    T_steps=T_steps,
    eval_steps=np.array(hist_steps),
    err_q_gd=best_curves["q_gd"],
    err_q_sd=best_curves["q_sd"],
    err_q_adam=best_curves["q_adam"],
    err_ce_gd=best_curves["ce_gd"],
    err_ce_sd=best_curves["ce_sd"],
    err_ce_adam=best_curves["ce_adam"],
)

print(f"Simulation terminée -> {filename}")
