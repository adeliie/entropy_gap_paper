import os

import numpy as np
import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_data(d):
    """Generate Zipf data"""
    ranks = torch.arange(1, d + 1, dtype=torch.float64, device=device)
    pi = torch.reciprocal(ranks)
    pi /= torch.sum(pi)
    return pi


def loss_gap(pi, W):
    """Return the KL divergence between the pi distribution and W."""
    log_P = F.log_softmax(W, dim=0)
    return torch.sum(pi * (torch.log(pi + 1e-30) - log_P)).item()


def loss_gap_per_col(pi, W):
    """Compute the per-column KL gap between pi and W."""
    log_P = F.log_softmax(W, dim=0)
    return torch.sum(
        pi.unsqueeze(1) * (torch.log(pi.unsqueeze(1) + 1e-30) - log_P),
        dim=0,
    )


def gd_matrix_per_col(T, d, eta=0, N_checkpoints=100):
    """Run the matrix-form gradient descent and return the recorded relative errors over time."""
    compute_dtype = torch.float32

    pi = generate_data(d).to(dtype=compute_dtype)
    if eta == 0:
        eta = (1.0 / (pi[0] ** 2)).item()

    pi_col = pi.unsqueeze(1)
    pi_row = pi.unsqueeze(0)
    update_const_matrix = eta * (pi_col @ pi_row)
    eta_pi_row = eta * pi_row

    W_T = torch.full((d, d), -np.log(d), dtype=compute_dtype, device=device)
    U_T = update_const_matrix.t().contiguous()
    eta_pi_col = eta_pi_row.t().contiguous()
    pi_row_loss = pi.unsqueeze(0).contiguous()
    log_pi_row_loss = torch.log(pi_row_loss + 1e-30)

    def loss_gap_per_col_T(W_T_curr):
        log_P_T = F.log_softmax(W_T_curr, dim=-1)
        return torch.sum(pi_row_loss * (log_pi_row_loss - log_P_T), dim=-1)

    init_err_per_col = loss_gap_per_col_T(torch.zeros_like(W_T))

    raw = np.unique(np.round(np.geomspace(1, T, N_checkpoints)).astype(int))
    checkpoint_set = set(raw.tolist())

    saved_t = [0]
    errors = [np.ones(d)]

    with torch.no_grad():
        for t in range(1, T + 1):
            P_T = F.softmax(W_T, dim=-1)
            W_T.add_(U_T)
            W_T.addcmul_(eta_pi_col, P_T, value=-1.0)

            if t in checkpoint_set:
                curr_err_per_col = loss_gap_per_col_T(W_T)
                errors.append((curr_err_per_col / init_err_per_col).cpu().numpy())
                saved_t.append(t)

    return np.array(saved_t, dtype=np.float64), np.array(errors)


def gradient_descent(T, t_save, d, eta=0):
    """Run the scalar gradient descent and interpolate the relative error at requested times."""
    pi = generate_data(d)

    if eta == 0:
        eta = 1.0 / (pi[0] ** 2)

    pi_cpu = pi.cpu().numpy()
    lr_fast = (eta * pi[0]).item()

    W_zeros = torch.zeros(d, dtype=torch.float64, device=device)
    init_err = loss_gap(pi, W_zeros)

    W = torch.full((d,), -np.log(d), dtype=torch.float64, device=device)

    N_checkpoints = 500
    raw = np.unique(np.round(np.geomspace(1, T, N_checkpoints)).astype(int))
    checkpoint_set = set(raw.tolist())

    saved_t = [0]
    saved_errors = [1.0]

    with torch.no_grad():
        for t in range(1, T + 1):
            P = F.softmax(W, dim=0)
            W += lr_fast * (pi - P)

            if t in checkpoint_set:
                saved_errors.append(loss_gap(pi, W) / init_err)
                saved_t.append(t)

    saved_t = np.array(saved_t, dtype=np.float64)
    saved_errors = np.array(saved_errors, dtype=np.float64)

    total_errors = []
    for t in t_save:
        t_equiv = t * pi_cpu / pi_cpu[0]
        errs_at_t = np.interp(t_equiv, saved_t, saved_errors)
        total_errors.append(errs_at_t)

    return np.array(total_errors)


def gradient_flow(T, t_save, d, eta=0):
    """Run the continuous-time gradient flow by integrating the exact exponential dynamics."""
    pi = generate_data(d).to(device, dtype=torch.float64)

    if eta == 0:
        eta = 1.0 / (pi[0] ** 2).item()

    lr_eff = eta * pi
    max_tau = (lr_eff[0] * T).item()

    W_zeros = torch.zeros(d, dtype=torch.float64, device=device)
    init_err = loss_gap(pi, W_zeros)

    N_steps = 3000
    s_grid = np.geomspace(1e-6, max_tau, N_steps)
    s_grid = np.insert(s_grid, 0, 0.0)

    Y = torch.full((d,), d, dtype=torch.float64, device=device)
    errors_traj = np.zeros(len(s_grid))
    errors_traj[0] = 1.0

    pi_inv = 1.0 / pi

    with torch.no_grad():
        for i in range(1, len(s_grid)):
            ds = s_grid[i] - s_grid[i - 1]
            Z = torch.sum(1.0 / Y)
            exp_term = torch.exp(-pi * ds)
            Y = Y * exp_term + (1.0 - exp_term) * pi_inv / Z
            loss = loss_gap(pi, -torch.log(Y + 1e-30))
            errors_traj[i] = loss / init_err

    lr_eff_cpu = lr_eff.cpu().numpy()
    total_errors = []

    for t in t_save:
        tau_j = lr_eff_cpu * t
        errs_at_t = np.interp(tau_j, s_grid, errors_traj)
        total_errors.append(errs_at_t)

    return np.array(total_errors)


def load_freqs(vocab_size):
    """Load token and conditional frequency arrays for the real-data experiments."""
    output_dir = "freqs"
    freqs = np.load(os.path.join(output_dir, f"token_freq_total_{vocab_size}.npy"))
    cond_freqs = np.load(os.path.join(output_dir, f"bigram_freq_total_{vocab_size}.npy"))
    return freqs, cond_freqs


def loss_gap_real_global(pi_cond, pi_marginal, W):
    """Compute the weighted global excess loss for the real-data conditional model."""
    log_P = F.log_softmax(W, dim=0)
    kl_per_col = torch.sum(pi_cond * (torch.log(pi_cond + 1e-30) - log_P), dim=0)
    global_kl = torch.sum(pi_marginal * kl_per_col).item()
    return global_kl


def gradient_real_data(vocab_size, T, eta=0, N_checkpoints=100):
    freqs, cond_freqs = load_freqs(vocab_size)
    d = vocab_size
    
    compute_dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pi = torch.tensor(freqs, dtype=compute_dtype, device=device)
    pi = pi / torch.sum(pi) 

    pi_cond = torch.tensor(cond_freqs, dtype=compute_dtype, device=device).T
    col_sums = torch.sum(pi_cond, dim=0, keepdim=True)
    pi_cond = pi_cond / torch.clamp(col_sums, min=1e-30) 
    
    pi_joint = pi * pi_cond
    if eta == 0:
        eta = 1.0 / torch.max(pi_joint).item()

    update_const_matrix = eta * pi * pi_cond
    eta_pi_row = eta * pi.unsqueeze(0) 

    W_T = torch.full((d, d), -np.log(d), dtype=compute_dtype, device=device)
    U_T = update_const_matrix.t().contiguous()
    eta_pi_col = eta_pi_row.t().contiguous()
    
    pi_cond_T = pi_cond.t().contiguous()
    log_pi_cond_T = torch.log(pi_cond_T + 1e-30)
    
    def loss_gap_per_col_T(W_T_curr):
        log_P_T = F.log_softmax(W_T_curr, dim=-1)
        kl_terms = pi_cond_T * (log_pi_cond_T - log_P_T)
        kl_terms = torch.nan_to_num(kl_terms, nan=0.0)
        
        return torch.sum(kl_terms, dim=-1)

    init_err_per_col = loss_gap_per_col_T(torch.zeros_like(W_T))
    safe_init_err = torch.clamp(init_err_per_col, min=1e-30)

    raw = np.unique(np.round(np.geomspace(1, T, N_checkpoints)).astype(int))

    saved_t = [0]
    errors = [np.ones(d)]

    def step(W, U, eta_pi_c):
        P = F.softmax(W, dim=-1)
        return W + U - (eta_pi_c * P)

    prev_t = 0
    with torch.no_grad():
        for t in raw:
            steps = t - prev_t
            for _ in range(steps):
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                W_T = step(W_T, U_T, eta_pi_col)

            curr_err_per_col = loss_gap_per_col_T(W_T)
            errors.append((curr_err_per_col / safe_init_err).cpu().numpy())
            
            saved_t.append(t)
            prev_t = t

    return np.array(saved_t, dtype=np.float64), np.array(errors)


def sign_descent_real_data(vocab_size, T, eta):
    """Run sign descent on the real data distribution and return the normalized final error."""
    freqs, cond_freqs = load_freqs(vocab_size)
    d = vocab_size

    compute_dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pi = torch.tensor(freqs, dtype=compute_dtype, device=device)
    pi = pi / torch.sum(pi)

    pi_cond = torch.tensor(cond_freqs, dtype=compute_dtype, device=device).T
    col_sums = torch.sum(pi_cond, dim=0, keepdim=True)
    pi_cond = pi_cond / torch.clamp(col_sums, min=1e-30)

    pi_cond_T = pi_cond.t().contiguous()
    log_pi_cond_T = torch.log(pi_cond_T + 1e-30)
    W_T = torch.full((d, d), -np.log(d), dtype=compute_dtype, device=device)

    def global_loss_gap(W_T_curr):
        log_P_T = F.log_softmax(W_T_curr, dim=-1)
        kl_terms = pi_cond_T * (log_pi_cond_T - log_P_T)
        kl_terms = torch.nan_to_num(kl_terms, nan=0.0)
        kl_per_col = torch.sum(kl_terms, dim=-1)
        return torch.sum(pi * kl_per_col).item()

    init_err = global_loss_gap(torch.zeros_like(W_T))
    safe_init_err = max(init_err, 1e-30)

    def step_fused_sign(W, pi_c_T, lr):
        P = F.softmax(W, dim=-1)
        return W + lr * torch.sign(pi_c_T - P)

    with torch.no_grad():
        for _ in range(T):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            W_T = step_fused_sign(W_T, pi_cond_T, eta)

    final_err = global_loss_gap(W_T)
    return final_err / safe_init_err


def sign_descent(T, d, eta, normalized=False):
    """Run sign descent on the synthetic distribution and optionally return the normalized error."""
    compute_dtype = torch.float64
    pi = generate_data(d).to(dtype=compute_dtype, device=device)

    if eta == 0:
        eta = np.log(d) / (2 * T)

    W = torch.full((d,), -np.log(d), dtype=compute_dtype, device=device)
    init_err = loss_gap(pi, W)
    safe_init_err = max(init_err, 1e-30)

    with torch.no_grad():
        for _ in range(T):
            P = F.softmax(W, dim=0)
            W += eta * torch.sign(pi - P)

    final_err = loss_gap(pi, W)

    if normalized:
        return final_err / safe_init_err
    return final_err
