import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
from tqdm import tqdm  # Progress bar library

# ==== GPU CONFIGURATION ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")

os.makedirs("plots_pinn", exist_ok=True)

# === Morris–Lecar scenarios ===
scenarios = {
    'hopf_50': {
        'params': dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.04, V1=-1.2, V2=18.0, V3=2.0, V4=30.0, I=50.0),
        'y0': [0.0, 0.0],
        'reference': ''
    }
}

# === PINN model ===
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )
    def forward(self, t):
        return self.net(t)

# === Residual physics-based loss ===
def residual_loss(t, pred, p):
    V, N = pred[:, 0:1], pred[:, 1:2]
    dVdt = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dNdt = torch.autograd.grad(N, t, grad_outputs=torch.ones_like(N), create_graph=True)[0]
    M_inf = 0.5 * (1 + torch.tanh((V - p['V1']) / p['V2']))
    N_inf = 0.5 * (1 + torch.tanh((V - p['V3']) / p['V4']))
    tau_N = 1.0 / torch.cosh((V - p['V3']) / (2 * p['V4']))
    f_V = (-p['g_Ca'] * M_inf * (V - p['V_Ca']) - p['g_K'] * N * (V - p['V_K']) - p['g_L'] * (V - p['V_L']) + p['I']) / p['C']
    f_N = p['phi'] * (N_inf - N) / tau_N
    return torch.mean((dVdt - f_V)**2 + (dNdt - f_N)**2)

# === Load dataset ===
def load_ground_truth_from_csv(scenario_name):
    path = f"scenario_data/{scenario_name}_ground_truth.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    t = df["t"].values
    V = df["V"].values
    N = df["N"].values
    y = np.stack([V, N], axis=1)
    return t, y

# === Train and evaluate function ===
def train_and_evaluate(scenario_name, params, epoch_list):
    t_np, y_true_np = load_ground_truth_from_csv(scenario_name)
    t_tensor = torch.tensor(t_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
    y_true = torch.tensor(y_true_np, dtype=torch.float32).to(device)
    results = []

    best_loss = float("inf")
    best_model_state = None
    best_epoch = None

    for epochs in epoch_list:
        print(f"\nTraining {scenario_name.upper()} for {epochs} epochs")
        model = PINN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500)
        mse = nn.MSELoss()

        loss_list = []

        for epoch in tqdm(range(epochs), desc=f"{scenario_name.upper()} {epochs} Epochs", ncols=90):
            optimizer.zero_grad()
            pred = model(t_tensor)
            loss_data = mse(pred, y_true)
            loss_phys = residual_loss(t_tensor, pred, params)
            total_loss = loss_data + loss_phys
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            loss_list.append(total_loss.item())

        # Inference
        with torch.no_grad():
            pred = model(t_tensor).cpu().numpy()

        # Metrics
        mse_v = mean_squared_error(y_true_np[:, 0], pred[:, 0])
        mse_n = mean_squared_error(y_true_np[:, 1], pred[:, 1])
        rmse_v = np.sqrt(mse_v)
        rmse_n = np.sqrt(mse_n)
        total = mse_v + mse_n

        if total < best_loss:
            best_loss = total
            best_model_state = model.state_dict()
            best_epoch = epochs

        results.append(dict(Scenario=scenario_name, Epochs=epochs,
                            MSE_V=mse_v, MSE_N=mse_n, RMSE_V=rmse_v, RMSE_N=rmse_n))

        # === ΜΟΝΟ το V(t) ===
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle(f"{scenario_name.upper()}/PINN/{epochs} Epochs/I=50", fontsize=14)

        ax.plot(t_np, y_true_np[:, 0], label='True V')
        ax.plot(t_np, pred[:, 0], '--', label='PINN V')
        ax.set_title('V(t)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Voltage V')
        ax.legend()
        ax.grid()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"plots_pinn/{scenario_name}_{epochs}_PINN.eps", format='eps', dpi=600)
        plt.close()

    return results

# === Main execution ===
all_results = []
epoch_list = [10000]

for name, cfg in scenarios.items():
    res = train_and_evaluate(name, cfg['params'], epoch_list)
    all_results.extend(res)

print("All PINN training complete. Metrics saved.")
