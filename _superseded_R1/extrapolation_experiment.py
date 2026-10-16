"""
Extrapolation Experiment — Response to Reviewer #1
===================================================
Train both PINN and NODE on t ∈ [0, 200] ms,
evaluate on the unseen interval t ∈ [200, 300] ms.

This directly addresses Reviewer #1's concern that the original
experiment was in-distribution (train and test on same interval).

Usage:
    cd Code_final
    python extrapolation_experiment.py

Outputs:
    extrapolation_results/  — plots and CSV metrics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torchdiffeq import odeint
import os, time, random

# --- Reproducibility ---
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

os.makedirs("extrapolation_results", exist_ok=True)

# --- Scenarios ---
scenarios = {
    'hopf': dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
                 V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                 phi=0.04, V1=-1.2, V2=18.0, V3=2.0, V4=30.0, I=90.0),
    'snlc': dict(C=20.0, g_Ca=4.0, g_K=8.0, g_L=2.0,
                 V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                 phi=0.067, V1=-1.2, V2=18.0, V3=12.0, V4=17.4, I=42.0),
    'homoclinic': dict(C=20.0, g_Ca=4.0, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.23, V1=-1.2, V2=18.0, V3=12.0, V4=17.4, I=50.0),
}

TRAIN_END = 200.0   # ms — training horizon
EPOCHS    = 10000   # epochs (use best epoch from original study)

# --- Models ---
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 2)
        )
    def forward(self, t):
        return self.net(t)

class NODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 2)
        )
    def forward(self, t, y):
        return self.net(y)

# --- PINN residual loss ---
def residual_loss(t, pred, p):
    V, N = pred[:, 0:1], pred[:, 1:2]
    dVdt = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dNdt = torch.autograd.grad(N, t, grad_outputs=torch.ones_like(N), create_graph=True)[0]
    M_inf = 0.5 * (1 + torch.tanh((V - p['V1']) / p['V2']))
    N_inf = 0.5 * (1 + torch.tanh((V - p['V3']) / p['V4']))
    tau_N = 1.0 / torch.cosh((V - p['V3']) / (2 * p['V4']))
    f_V = (-p['g_Ca']*M_inf*(V-p['V_Ca']) - p['g_K']*N*(V-p['V_K']) - p['g_L']*(V-p['V_L']) + p['I']) / p['C']
    f_N = p['phi'] * (N_inf - N) / tau_N
    return torch.mean((dVdt - f_V)**2 + (dNdt - f_N)**2)

# --- Metrics helper ---
def _safe_mape(y_true, y_pred):
    """MAPE with zero-guard: skip points where |y_true| < 1e-8."""
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return np.nan
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def _rmspe(y_true, y_pred):
    """RMSPE with zero-guard."""
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask])**2)) * 100.0

def compute_metrics(y_true, y_pred):
    mse_v = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_n = mean_squared_error(y_true[:, 1], y_pred[:, 1])
    return dict(
        MSE_V=mse_v, MSE_N=mse_n,
        RMSE_V=np.sqrt(mse_v), RMSE_N=np.sqrt(mse_n),
        MAE_V=mean_absolute_error(y_true[:, 0], y_pred[:, 0]),
        MAE_N=mean_absolute_error(y_true[:, 1], y_pred[:, 1]),
        R2_V=r2_score(y_true[:, 0], y_pred[:, 0]),
        R2_N=r2_score(y_true[:, 1], y_pred[:, 1]),
        MAPE_V=_safe_mape(y_true[:, 0], y_pred[:, 0]),
        MAPE_N=_safe_mape(y_true[:, 1], y_pred[:, 1]),
        RMSPE_V=_rmspe(y_true[:, 0], y_pred[:, 0]),
        RMSPE_N=_rmspe(y_true[:, 1], y_pred[:, 1]),
        MaxErr_V=float(np.max(np.abs(y_true[:, 0] - y_pred[:, 0]))),
        MaxErr_N=float(np.max(np.abs(y_true[:, 1] - y_pred[:, 1]))),
    )

# --- Main experiment ---
all_results = []

for scenario_name, params in scenarios.items():
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {scenario_name.upper()}")
    print(f"{'='*60}")

    # Load full ground truth
    df = pd.read_csv(f"scenario_data/{scenario_name}_ground_truth.csv")
    t_np   = df['t'].values.astype(np.float32)
    y_np   = df[['V', 'N']].values.astype(np.float32)

    # Split train / test
    train_mask = t_np <= TRAIN_END
    test_mask  = t_np >  TRAIN_END

    t_train = t_np[train_mask];  y_train = y_np[train_mask]
    t_test  = t_np[test_mask];   y_test  = y_np[test_mask]

    print(f"  Train points: {train_mask.sum()}  |  Test points: {test_mask.sum()}")

    # --- ① PINN ---
    print(f"\n  [PINN] Training on t ∈ [0, {TRAIN_END}] ms ...")
    pinn = PINN().to(device)
    opt  = torch.optim.Adam(pinn.parameters(), lr=1e-3)

    t_tr = torch.tensor(t_train.reshape(-1,1), dtype=torch.float32, requires_grad=True).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)

    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        opt.zero_grad()
        pred = pinn(t_tr)
        loss = torch.mean((pred - y_tr)**2) + residual_loss(t_tr, pred, params)
        loss.backward(); opt.step()
        if (ep+1) % 2000 == 0:
            print(f"    Epoch {ep+1}/{EPOCHS}  loss={loss.item():.6f}")
    pinn_train_time = time.perf_counter() - t0

    # Evaluate on TEST interval
    with torch.no_grad():
        t_te_tensor = torch.tensor(t_test.reshape(-1,1), dtype=torch.float32)
        pinn_pred_test = pinn(t_te_tensor).cpu().numpy()

    # Also evaluate on TRAIN interval (for reference)
    with torch.no_grad():
        t_tr_eval = torch.tensor(t_train.reshape(-1,1), dtype=torch.float32)
        pinn_pred_train = pinn(t_tr_eval).cpu().numpy()

    m_pinn_train = compute_metrics(y_train, pinn_pred_train)
    m_pinn_test  = compute_metrics(y_test,  pinn_pred_test)

    print(f"    PINN  | Train RMSE_V={m_pinn_train['RMSE_V']:.4f}  | Test RMSE_V={m_pinn_test['RMSE_V']:.4f}  R²_V={m_pinn_test['R2_V']:.4f}")

    # --- ② NODE ---
    print(f"\n  [NODE] Training on t ∈ [0, {TRAIN_END}] ms ...")

    # Normalise on TRAIN data only
    scaler_V = MinMaxScaler(); scaler_N = MinMaxScaler()
    t_min_tr, t_max_tr = t_train.min(), t_train.max()

    V_sc = scaler_V.fit_transform(y_train[:, 0].reshape(-1,1)).flatten()
    N_sc = scaler_N.fit_transform(y_train[:, 1].reshape(-1,1)).flatten()
    t_sc = (t_train - t_min_tr) / (t_max_tr - t_min_tr)

    y_sc = np.stack([V_sc, N_sc], axis=1).astype(np.float32)
    t_tensor_tr = torch.tensor(t_sc, dtype=torch.float32).to(device)
    y_tensor_tr = torch.tensor(y_sc, dtype=torch.float32).to(device)

    func = NODEFunc().to(device)
    opt2 = torch.optim.Adam(func.parameters(), lr=1e-3)

    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        opt2.zero_grad()
        y0 = torch.tensor(y_sc[0], dtype=torch.float32).to(device)
        pred_y = odeint(func, y0, t_tensor_tr, method='dopri5', atol=1e-5, rtol=1e-4)
        loss = torch.mean((pred_y - y_tensor_tr)**2)
        loss.backward(); opt2.step()
        if (ep+1) % 2000 == 0:
            print(f"    Epoch {ep+1}/{EPOCHS}  loss={loss.item():.6f}")
    node_train_time = time.perf_counter() - t0

    # Evaluate on TEST interval — extend normalised time beyond [0,1]
    t_ext = (t_test - t_min_tr) / (t_max_tr - t_min_tr)   # values > 1.0
    t_all_sc = np.concatenate([t_sc, t_ext]).astype(np.float32)
    t_all_tensor = torch.tensor(t_all_sc, dtype=torch.float32).to(device)

    with torch.no_grad():
        y0 = torch.tensor(y_sc[0], dtype=torch.float32).to(device)
        pred_all = odeint(func, y0, t_all_tensor, method='dopri5').cpu().numpy()

    # Split and inverse-transform
    n_tr = len(t_sc)
    pred_train_sc = pred_all[:n_tr]
    pred_test_sc  = pred_all[n_tr:]

    V_pred_train = scaler_V.inverse_transform(pred_train_sc[:, 0].reshape(-1,1)).flatten()
    N_pred_train = scaler_N.inverse_transform(pred_train_sc[:, 1].reshape(-1,1)).flatten()
    V_pred_test  = scaler_V.inverse_transform(pred_test_sc[:,  0].reshape(-1,1)).flatten()
    N_pred_test  = scaler_N.inverse_transform(pred_test_sc[:,  1].reshape(-1,1)).flatten()

    node_pred_train = np.stack([V_pred_train, N_pred_train], axis=1)
    node_pred_test  = np.stack([V_pred_test,  N_pred_test],  axis=1)

    m_node_train = compute_metrics(y_train, node_pred_train)
    m_node_test  = compute_metrics(y_test,  node_pred_test)

    print(f"    NODE  | Train RMSE_V={m_node_train['RMSE_V']:.4f}  | Test RMSE_V={m_node_test['RMSE_V']:.4f}  R²_V={m_node_test['R2_V']:.4f}")

    # --- Save results ---
    for split, m_pinn, m_node in [('train', m_pinn_train, m_node_train),
                                   ('test',  m_pinn_test,  m_node_test)]:
        for metric, val in m_pinn.items():
            all_results.append(dict(scenario=scenario_name, split=split,
                                    model='PINN', metric=metric, value=val))
        for metric, val in m_node.items():
            all_results.append(dict(scenario=scenario_name, split=split,
                                    model='NODE', metric=metric, value=val))

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Extrapolation — {scenario_name.upper()} regime\n"
                 f"Train: t∈[0,{TRAIN_END}]ms  |  Test: t∈({TRAIN_END},300]ms", fontsize=12)

    for ax_row, var_idx, var_name in zip(axes, [0, 1], ['V (mV)', 'N (dimensionless)']):
        # PINN
        ax = ax_row[0]
        ax.plot(t_train, y_train[:, var_idx], 'k-',  lw=1.5, label='Ground truth (train)')
        ax.plot(t_test,  y_test[:,  var_idx], 'k--', lw=1.5, label='Ground truth (test)')
        ax.plot(t_train, pinn_pred_train[:, var_idx], 'b-',  lw=1.2, label='PINN train pred')
        ax.plot(t_test,  pinn_pred_test[:,  var_idx], 'r-',  lw=1.2, label='PINN extrap.')
        ax.axvline(TRAIN_END, color='gray', linestyle=':', lw=1)
        ax.set_title(f'PINN — {var_name}')
        ax.set_xlabel('t (ms)'); ax.legend(fontsize=7)

        # NODE
        ax = ax_row[1]
        ax.plot(t_train, y_train[:, var_idx], 'k-',  lw=1.5, label='Ground truth (train)')
        ax.plot(t_test,  y_test[:,  var_idx], 'k--', lw=1.5, label='Ground truth (test)')
        ax.plot(t_train, node_pred_train[:, var_idx], 'b-',  lw=1.2, label='NODE train pred')
        ax.plot(t_test,  node_pred_test[:,  var_idx], 'r-',  lw=1.2, label='NODE extrap.')
        ax.axvline(TRAIN_END, color='gray', linestyle=':', lw=1)
        ax.set_title(f'NODE — {var_name}')
        ax.set_xlabel('t (ms)'); ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"extrapolation_results/{scenario_name}_extrapolation.eps", format='eps', dpi=150)
    plt.savefig(f"extrapolation_results/{scenario_name}_extrapolation.png", dpi=150)
    plt.close()
    print(f"  → Saved plot: extrapolation_results/{scenario_name}_extrapolation.png")

# --- Save all metrics to CSV ---
df_res = pd.DataFrame(all_results)
df_res.to_csv("extrapolation_results/extrapolation_metrics.csv", index=False)

# --- Print summary table ---
print("\n" + "="*70)
print("  EXTRAPOLATION SUMMARY (Test interval t ∈ (200, 300] ms)")
print("="*70)
pivot = df_res[df_res['split'] == 'test'].pivot_table(
    index=['scenario', 'model'], columns='metric', values='value')
print(pivot[['RMSE_V', 'RMSE_N', 'R2_V', 'R2_N', 'MAPE_V', 'MAPE_N', 'MaxErr_V']].to_string())
print("\nAll metrics saved to: extrapolation_results/extrapolation_metrics.csv")
