import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, max_error, r2_score, mean_absolute_error
import os
import time
from tqdm import tqdm  # Progress bar
from thop import profile
import gc
import random

############################################### ==== DETERMINISTIC SETUP ==== ##############################################
USE_DETERMINISTIC = True
if USE_DETERMINISTIC:
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("🧪 Deterministic mode active with seed", SEED)

############################################### ==== DEVICE CONFIGURATION ==== ##############################################
device = torch.device("cpu")
print(f" Device in use: {device}")

############################################### ==== USE FLOAT32 BY DEFAULT ==== ##############################################
torch.set_default_dtype(torch.float32)

############################################### ==== DIRECTORY SETUP ==== ##############################################
os.makedirs("plots_pinn", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("models_pinn", exist_ok=True)

############################################### ==== DEFINITION OF SCENARIOS ==== ##############################################
scenarios = {
    'hopf': {
        'params': dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.04, V1=-1.2, V2=18.0, V3=2.0, V4=30.0, I=90.0),
        'y0': [0.0, 0.0]
    },
    'snlc': {
        'params': dict(C=20.0, g_Ca=4.0, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.067, V1=-1.2, V2=18.0, V3=12.0, V4=17.4, I=42),
        'y0': [0.0, 0.0]
    },
    'homoclinic': {
        'params': dict(C=20.0, g_Ca=4.0, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.23, V1=-1.2, V2=18.0, V3=12.0, V4=17.4, I=50),
        'y0': [0.0, 0.0]
    }
}

############################################### ==== PINN MODEL ==== ##############################################
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
            nn.Linear(128, 2)  # Output: [V, N]
        )

    def forward(self, t):
        return self.net(t)

############################################### ==== RESIDUAL LOSS ==== ##############################################
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

# ==== LOAD DATA ====
def load_ground_truth_from_csv(scenario_name):
    path = f"scenario_data/{scenario_name}_ground_truth.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f" File not found: {path}")
    df = pd.read_csv(path)
    t = df["t"].values
    V = df["V"].values
    N = df["N"].values
    y = np.stack([V, N], axis=1)
    return t, y

# ==== TRAINING ====
def train_and_evaluate(scenario_name, params, epoch_list):
    t_np, y_true_np = load_ground_truth_from_csv(scenario_name)
    t_tensor = torch.tensor(t_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
    y_true = torch.tensor(y_true_np, dtype=torch.float32).to(device)

    results = []
    best_loss = float("inf")
    best_model_state = None
    best_epoch = None

    for epochs in epoch_list:
        print(f"\n Training {scenario_name.upper()} for {epochs} epochs")
        model = PINN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        mse = nn.MSELoss()
        loss_list = []

        start_time = time.perf_counter()

        for epoch in tqdm(range(epochs), desc=f"{scenario_name.upper()} {epochs} Epochs", ncols=90):
            optimizer.zero_grad()
            pred = model(t_tensor)
            loss_data = mse(pred, y_true)
            loss_phys = residual_loss(t_tensor, pred, params)
            total_loss = loss_data + loss_phys
            total_loss.backward()
            optimizer.step()
            loss_list.append(total_loss.item())

        elapsed_time = time.perf_counter() - start_time

        with torch.no_grad():
            pred = model(t_tensor).cpu().numpy()

        dummy_input = torch.rand(1, 1, dtype=torch.float32).to(device)
        flops, params_count = profile(model.net, inputs=(dummy_input,), verbose=False)

        mse_v = mean_squared_error(y_true_np[:, 0], pred[:, 0])
        mse_n = mean_squared_error(y_true_np[:, 1], pred[:, 1])
        rmse_v = np.sqrt(mse_v)
        rmse_n = np.sqrt(mse_n)
        mae_v = mean_absolute_error(y_true_np[:, 0], pred[:, 0])
        mae_n = mean_absolute_error(y_true_np[:, 1], pred[:, 1])
        r2_v = r2_score(y_true_np[:, 0], pred[:, 0])
        r2_n = r2_score(y_true_np[:, 1], pred[:, 1])
        maxerr_v = max_error(y_true_np[:, 0], pred[:, 0])
        maxerr_n = max_error(y_true_np[:, 1], pred[:, 1])

        eps = 1e-8
        rmspe_v = np.sqrt(np.mean(((y_true_np[:, 0] - pred[:, 0]) / (y_true_np[:, 0] + eps))**2))
        rmspe_n = np.sqrt(np.mean(((y_true_np[:, 1] - pred[:, 1]) / (y_true_np[:, 1] + eps))**2))
        mape_v = np.mean(np.abs((y_true_np[:, 0] - pred[:, 0]) / (y_true_np[:, 0] + eps))) * 100
        mape_n = np.mean(np.abs((y_true_np[:, 1] - pred[:, 1]) / (y_true_np[:, 1] + eps))) * 100

        total = mse_v + mse_n
        if total < best_loss:
            best_loss = total
            best_model_state = model.state_dict()
            best_epoch = epochs

        results.append(dict(
            Scenario=scenario_name,
            Epochs=epochs,
            MSE_V=mse_v, MSE_N=mse_n,
            Total_MSE=total,
            RMSE_V=rmse_v, RMSE_N=rmse_n,
            RMSPE_V=rmspe_v, RMSPE_N=rmspe_n,
            MAPE_V=mape_v, MAPE_N=mape_n,
            MAE_V=mae_v, MAE_N=mae_n,
            R2_V=r2_v, R2_N=r2_n,
            MAXERR_V=maxerr_v, MAXERR_N=maxerr_n,
            Training_Time_Sec=elapsed_time,
            FLOPs=flops, Params=params_count
        ))

        fig, axs = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle(f"{scenario_name.upper()} - PINN - {epochs} Epochs", fontsize=14)

        axs[0, 0].plot(t_np, y_true_np[:, 0], label='True V')
        axs[0, 0].plot(t_np, pred[:, 0], '--', label='PINN V')
        axs[0, 0].set_title('V(t)'); axs[0, 0].legend(); axs[0, 0].grid()

        axs[0, 1].plot(t_np, y_true_np[:, 1], label='True N')
        axs[0, 1].plot(t_np, pred[:, 1], '--', label='PINN N')
        axs[0, 1].set_title('N(t)'); axs[0, 1].legend(); axs[0, 1].grid()

        axs[1, 0].plot(y_true_np[:, 0], y_true_np[:, 1], label='True')
        axs[1, 0].plot(pred[:, 0], pred[:, 1], '--', label='PINN')
        axs[1, 0].set_title('Phase Portrait'); axs[1, 0].legend(); axs[1, 0].grid()

        axs[1, 1].plot(loss_list)
        axs[1, 1].set_title('Training Loss'); axs[1, 1].set_xlabel("Epoch"); axs[1, 1].set_ylabel("Loss"); axs[1, 1].grid()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"plots_pinn/{scenario_name}_{epochs}_PINN.eps", format='eps', dpi=600)
        plt.savefig(f"plots_pinn/{scenario_name}_{epochs}_PINN.png", dpi=600)
        plt.close()

        torch.save(model.state_dict(), f"models_pinn/{scenario_name}_{epochs}_model.pth")
        gc.collect()

    if best_model_state is not None:
        torch.save(best_model_state, f"models_pinn/{scenario_name}_best_model_{best_epoch}epochs.pth")
        print(f"💾 Best model saved: models_pinn/{scenario_name}_best_model_{best_epoch}epochs.pth (Loss = {best_loss:.5f})")

    return results

############################################### ==== MAIN ==== ##############################################
all_results = []
epoch_list = [1000, 2000, 5000, 10000,20000]

for name, cfg in scenarios.items():
    res = train_and_evaluate(name, cfg['params'], epoch_list)
    all_results.extend(res)

df = pd.DataFrame(all_results)
df.to_csv("metrics/metrics_pinn_summary.csv", index=False)
print(" All PINN training complete. Metrics saved.")

print(" Best PINN Models Per Scenario:")
grouped = df.groupby("Scenario")

for scenario, group in grouped:
    best_row = group.loc[(group["MSE_V"] + group["MSE_N"]).idxmin()]
    epochs = int(best_row["Epochs"])
    total_mse = best_row["MSE_V"] + best_row["MSE_N"]
    rmse_v = best_row["RMSE_V"]
    rmse_n = best_row["RMSE_N"]
    time_sec = best_row["Training_Time_Sec"]
    flops = best_row["FLOPs"]
    params = best_row["Params"]

    print(f"   {scenario.upper()}:")
    print(f"     → Best at {epochs} epochs")
    print(f"     → Total MSE = {total_mse:.4e}")
    print(f"     → RMSE_V = {rmse_v:.4f}, RMSE_N = {rmse_n:.4f}")
    print(f"     → Training Time = {time_sec:.2f} sec")
    print(f"     → FLOPs = {flops:,}, Params = {params:,}")
