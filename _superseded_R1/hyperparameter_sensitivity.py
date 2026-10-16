"""
Hyperparameter Sensitivity Analysis — Response to Reviewer #2
==============================================================
Tests three architectures for both PINN and NODE on the Hopf regime:
  - Small:   2 hidden layers × 64 neurons
  - Medium:  3 hidden layers × 128 neurons  (original, from paper)
  - Large:   4 hidden layers × 256 neurons

The experiment uses a fixed epoch count (10000) and the same
optimizer / seed as the original study.

Usage:
    cd Code_final
    python hyperparameter_sensitivity.py

Outputs:
    hyperparameter_results/  — plots and CSV metrics table
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

os.makedirs("hyperparameter_results", exist_ok=True)

# --- Fixed scenario: Hopf ---
SCENARIO = 'hopf'
PARAMS = dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
              V_Ca=120.0, V_K=-84.0, V_L=-60.0,
              phi=0.04, V1=-1.2, V2=18.0, V3=2.0, V4=30.0, I=90.0)
EPOCHS = 10000

# --- Architecture definitions ---
ARCHITECTURES = {
    'Small  (2×64)':   {'layers': 2, 'hidden': 64},
    'Medium (3×128)':  {'layers': 3, 'hidden': 128},   # original
    'Large  (4×256)':  {'layers': 4, 'hidden': 256},
}

# --- Model builders ---
def build_pinn(n_layers, n_hidden):
    layers = [nn.Linear(1, n_hidden), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
    layers.append(nn.Linear(n_hidden, 2))
    return nn.Sequential(*layers)

def build_node_func(n_layers, n_hidden):
    layers = [nn.Linear(2, n_hidden), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
    layers.append(nn.Linear(n_hidden, 2))
    return nn.Sequential(*layers)

class NODEWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
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

# --- Metrics ---
def compute_metrics(y_true, y_pred):
    mse_v = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_n = mean_squared_error(y_true[:, 1], y_pred[:, 1])
    return dict(
        MSE_V=mse_v,
        RMSE_V=np.sqrt(mse_v),
        MAE_V=mean_absolute_error(y_true[:, 0], y_pred[:, 0]),
        R2_V=r2_score(y_true[:, 0], y_pred[:, 0]),
        MaxErr_V=float(np.max(np.abs(y_true[:, 0] - y_pred[:, 0]))),
        MSE_N=mse_n,
        RMSE_N=np.sqrt(mse_n),
        R2_N=r2_score(y_true[:, 1], y_pred[:, 1]),
        MaxErr_N=float(np.max(np.abs(y_true[:, 1] - y_pred[:, 1]))),
    )

# --- Load data ---
df = pd.read_csv(f"scenario_data/{SCENARIO}_ground_truth.csv")
t_np = df['t'].values.astype(np.float32)
y_np = df[['V', 'N']].values.astype(np.float32)

# PINN tensors
t_pinn = torch.tensor(t_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
y_pinn = torch.tensor(y_np, dtype=torch.float32).to(device)

# NODE normalised tensors
scaler_V = MinMaxScaler(); scaler_N = MinMaxScaler()
t_min, t_max = t_np.min(), t_np.max()
V_sc = scaler_V.fit_transform(y_np[:, 0].reshape(-1,1)).flatten()
N_sc = scaler_N.fit_transform(y_np[:, 1].reshape(-1,1)).flatten()
t_sc = (t_np - t_min) / (t_max - t_min)
y_sc = np.stack([V_sc, N_sc], axis=1).astype(np.float32)
t_node = torch.tensor(t_sc, dtype=torch.float32).to(device)
y_node = torch.tensor(y_sc, dtype=torch.float32).to(device)

# --- Run experiments ---
all_results = []

for arch_name, arch_cfg in ARCHITECTURES.items():
    # Reset seed per architecture for fair, reproducible comparison
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    n_layers = arch_cfg['layers']
    n_hidden = arch_cfg['hidden']
    print(f"\n{'='*60}")
    print(f"  ARCHITECTURE: {arch_name}")
    print(f"{'='*60}")

    # Count params
    pinn_net = build_pinn(n_layers, n_hidden)
    n_params_pinn = sum(p.numel() for p in pinn_net.parameters())

    node_net = build_node_func(n_layers, n_hidden)
    n_params_node = sum(p.numel() for p in node_net.parameters())

    print(f"  PINN params: {n_params_pinn:,}  |  NODE params: {n_params_node:,}")

    # --- PINN ---
    print(f"  Training PINN ...")
    model = pinn_net.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        opt.zero_grad()
        pred = model(t_pinn)
        loss = torch.mean((pred - y_pinn)**2) + residual_loss(t_pinn, pred, PARAMS)
        loss.backward(); opt.step()
        if (ep+1) % 2000 == 0:
            print(f"    Epoch {ep+1}/{EPOCHS}  loss={loss.item():.6f}")
    pinn_time = time.perf_counter() - t0

    with torch.no_grad():
        pinn_pred = model(torch.tensor(t_np.reshape(-1,1), dtype=torch.float32)).cpu().numpy()

    m_pinn = compute_metrics(y_np, pinn_pred)
    m_pinn['TrainTime_s'] = round(pinn_time, 1)
    m_pinn['Params'] = n_params_pinn
    print(f"  PINN  RMSE_V={m_pinn['RMSE_V']:.4f}  R2_V={m_pinn['R2_V']:.4f}  time={pinn_time:.1f}s")

    # --- NODE ---
    print(f"  Training NODE ...")
    func = NODEWrapper(node_net.to(device))
    opt2 = torch.optim.Adam(func.parameters(), lr=1e-3)

    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        opt2.zero_grad()
        y0 = torch.tensor(y_sc[0], dtype=torch.float32).to(device)
        pred_y = odeint(func, y0, t_node, method='dopri5', atol=1e-5, rtol=1e-4)
        loss = torch.mean((pred_y - y_node)**2)
        loss.backward(); opt2.step()
        if (ep+1) % 2000 == 0:
            print(f"    Epoch {ep+1}/{EPOCHS}  loss={loss.item():.6f}")
    node_time = time.perf_counter() - t0

    with torch.no_grad():
        y0 = torch.tensor(y_sc[0], dtype=torch.float32).to(device)
        pred_sc = odeint(func, y0, t_node, method='dopri5').cpu().numpy()

    V_pred = scaler_V.inverse_transform(pred_sc[:, 0].reshape(-1,1)).flatten()
    N_pred = scaler_N.inverse_transform(pred_sc[:, 1].reshape(-1,1)).flatten()
    node_pred = np.stack([V_pred, N_pred], axis=1)

    m_node = compute_metrics(y_np, node_pred)
    m_node['TrainTime_s'] = round(node_time, 1)
    m_node['Params'] = n_params_node
    print(f"  NODE  RMSE_V={m_node['RMSE_V']:.4f}  R2_V={m_node['R2_V']:.4f}  time={node_time:.1f}s")

    # Store
    for m, model_name in [(m_pinn, 'PINN'), (m_node, 'NODE')]:
        row = {'Architecture': arch_name, 'Model': model_name}
        row.update(m)
        all_results.append(row)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Hopf regime — {arch_name}", fontsize=11)
    for ax, pred, title in zip(axes, [pinn_pred, node_pred], ['PINN', 'NODE']):
        ax.plot(t_np, y_np[:, 0], 'k-', lw=1.5, label='Ground truth')
        ax.plot(t_np, pred[:, 0], 'r--', lw=1.2, label='Prediction')
        ax.set_title(f'{title} — V(t)')
        ax.set_xlabel('t (ms)'); ax.set_ylabel('V (mV)')
        ax.legend(fontsize=8)
    plt.tight_layout()
    tag = arch_name.split('(')[1].rstrip(')').replace('×', 'x')
    plt.savefig(f"hyperparameter_results/hopf_{tag}.eps", format='eps', dpi=150)
    plt.savefig(f"hyperparameter_results/hopf_{tag}.png", dpi=150)
    plt.close()

# --- Save & print summary ---
df_res = pd.DataFrame(all_results)
df_res.to_csv("hyperparameter_results/hyperparameter_sensitivity.csv", index=False)

print("\n" + "="*70)
print("  HYPERPARAMETER SENSITIVITY SUMMARY (Hopf regime, 10000 epochs)")
print("="*70)
print(df_res[['Architecture', 'Model', 'Params', 'RMSE_V', 'R2_V', 'MaxErr_V', 'TrainTime_s']].to_string(index=False))
print("\nFull results saved to: hyperparameter_results/hyperparameter_sensitivity.csv")
