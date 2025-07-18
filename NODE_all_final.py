import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, max_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torchdiffeq import odeint
from tqdm import tqdm
import os
import gc
import time
from thop import profile
import platform
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
    print(" Deterministic mode active with seed", SEED)

############################################### ==== DEVICE CONFIGURATION ==== ##############################################
device = torch.device("cpu")
print(f"Device in use: {device}")

# ==== USE FLOAT32 BY DEFAULT ====
torch.set_default_dtype(torch.float32)
# ==== DIRECTORY SETUP ====
os.makedirs("plots_node", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("models_node", exist_ok=True)

# ==== SCENARIO DEFINITIONS ====
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

############################################### ==== NODE MODEL ==== ##############################################
class NODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

    def forward(self, t, y):
        return self.net(y)

############################################### ==== TRAINING AND EVALUATION ====##############################################
def train_and_evaluate(scenario_name, params, epoch_list):
    # === Load raw data ===
    df = pd.read_csv(f"scenario_data/{scenario_name}_ground_truth.csv")
    t_np = df['t'].values
    y_np = df[['V', 'N']].values

    # === Normalize (scale) V, N, and t ===
    scaler_V = MinMaxScaler()
    scaler_N = MinMaxScaler()
    t_min, t_max = t_np.min(), t_np.max()

    V_scaled = scaler_V.fit_transform(y_np[:, 0].reshape(-1, 1)).flatten()
    N_scaled = scaler_N.fit_transform(y_np[:, 1].reshape(-1, 1)).flatten()
    t_scaled = (t_np - t_min) / (t_max - t_min)


    y_scaled = np.stack([V_scaled, N_scaled], axis=1)

    # === Convert to PyTorch tensors ===
    t_tensor = torch.tensor(t_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(device)

    results = []
    best_loss = float('inf')
    best_model_state = None
    best_epoch = None

    for epochs in epoch_list:
        print(f"\n▶ Training {scenario_name.upper()} for {epochs} epochs")
        func = NODEFunc().to(device)
        optimizer = torch.optim.Adam(func.parameters(), lr=1e-3)
        loss_list = []

        start_time = time.perf_counter()

        for epoch in tqdm(range(epochs), desc=f"{scenario_name.upper()} {epochs} Epochs", ncols=90):
            optimizer.zero_grad()
            pred_y = odeint(func, torch.tensor(y_scaled[0], dtype=torch.float32).to(device), t_tensor, method='dopri5', atol=1e-5, rtol=1e-4)
            loss = torch.mean((pred_y - y_tensor) ** 2)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        elapsed_time = time.perf_counter() - start_time

        with torch.no_grad():
            pred_scaled = odeint(func, torch.tensor(y_scaled[0], dtype=torch.float32), t_tensor, method='dopri5').cpu().numpy()

        # === Inverse transform to original physical units ===
        t_unscaled = t_tensor.cpu().numpy().flatten() * (t_max - t_min) + t_min
        V_pred = scaler_V.inverse_transform(pred_scaled[:, 0].reshape(-1, 1)).flatten()
        N_pred = scaler_N.inverse_transform(pred_scaled[:, 1].reshape(-1, 1)).flatten()
        pred = np.stack([V_pred, N_pred], axis=1)

        # === Compute FLOPs and parameter count ===
        dummy_input = torch.rand(1, 2, dtype=torch.float32).to(device)
        flops, params_count = profile(func.net, inputs=(dummy_input,), verbose=False)

        # === Compute metrics ===
        mse_v = mean_squared_error(y_np[:, 0], pred[:, 0])
        mse_n = mean_squared_error(y_np[:, 1], pred[:, 1])
        rmse_v = np.sqrt(mse_v)
        rmse_n = np.sqrt(mse_n)
        mae_v = mean_absolute_error(y_np[:, 0], pred[:, 0])
        mae_n = mean_absolute_error(y_np[:, 1], pred[:, 1])
        r2_v = r2_score(y_np[:, 0], pred[:, 0])
        r2_n = r2_score(y_np[:, 1], pred[:, 1])
        maxerr_v = max_error(y_np[:, 0], pred[:, 0])
        maxerr_n = max_error(y_np[:, 1], pred[:, 1])

        eps = 1e-8
        rmspe_v = np.sqrt(np.mean(((y_np[:, 0] - pred[:, 0]) / (y_np[:, 0] + eps))**2))
        rmspe_n = np.sqrt(np.mean(((y_np[:, 1] - pred[:, 1]) / (y_np[:, 1] + eps))**2))
        mape_v = np.mean(np.abs((y_np[:, 0] - pred[:, 0]) / (y_np[:, 0] + eps))) * 100
        mape_n = np.mean(np.abs((y_np[:, 1] - pred[:, 1]) / (y_np[:, 1] + eps))) * 100

        total = mse_v + mse_n
        if total < best_loss:
            best_loss = total
            best_model_state = func.state_dict()
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

        # === Plotting ===
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle(f"{scenario_name.upper()} - NODE - {epochs} Epochs", fontsize=14)

        axs[0, 0].plot(t_unscaled, y_np[:, 0], label='True V')
        axs[0, 0].plot(t_unscaled, pred[:, 0], '--', label='NODE V')
        axs[0, 0].set_title('V(t)'); axs[0, 0].legend(); axs[0, 0].grid()

        axs[0, 1].plot(t_unscaled, y_np[:, 1], label='True N')
        axs[0, 1].plot(t_unscaled, pred[:, 1], '--', label='NODE N')
        axs[0, 1].set_title('N(t)'); axs[0, 1].legend(); axs[0, 1].grid()

        axs[1, 0].plot(y_np[:, 0], y_np[:, 1], label='True')
        axs[1, 0].plot(pred[:, 0], pred[:, 1], '--', label='NODE')
        axs[1, 0].set_title('Phase Portrait'); axs[1, 0].legend(); axs[1, 0].grid()

        axs[1, 1].plot(loss_list)
        axs[1, 1].set_title('Training Loss'); axs[1, 1].set_xlabel("Epoch"); axs[1, 1].set_ylabel("Loss"); axs[1, 1].grid()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"plots_node/{scenario_name}_{epochs}_NODE.eps", format='eps', dpi=600)
        plt.savefig(f"plots_node/{scenario_name}_{epochs}_NODE.png", dpi=600)
        plt.close()

        torch.save(func.state_dict(), f"models_node/{scenario_name}_{epochs}_model.pth")
        gc.collect()

    # === Save best model ===
    if best_model_state is not None:
        torch.save(best_model_state, f"models_node/{scenario_name}_best_model_{best_epoch}epochs.pth")
        print(f" Best model saved: models_node/{scenario_name}_best_model_{best_epoch}epochs.pth (Loss = {best_loss:.5f})")

    return results

############################################### ==== MAIN EXECUTION ==== ##############################################
epoch_list = [1000, 2000, 5000, 10000,20000]
all_results = []

for name, cfg in scenarios.items():
    res = train_and_evaluate(name, cfg['params'], epoch_list)
    all_results.extend(res)

############################################### === Save results to CSV === ##############################################
df = pd.DataFrame(all_results)
df.to_csv("metrics/metrics_node_summary.csv", index=False)
print(" All NODE training complete. Metrics saved.")

############################################### === Display best models summary === ##############################################
print(" Best NODE Models Per Scenario:")
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
