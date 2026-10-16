import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, max_error
from sklearn.preprocessing import MinMaxScaler
from torchdiffeq import odeint
from tqdm import trange
import os
import gc

# ==== DEVICE CONFIG ====
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"✅ Device in use: {device}")

# ==== FOLDERS ====
os.makedirs("plots_node", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("models_node", exist_ok=True)

# ==== SCENARIOS ====
scenarios = {
    'hopf': {
        'params': dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.04, V1=-1.2, V2=18.0, V3=2.0, V4=30.0, I=50.0),
        'y0': [0.0, 0.0],
        'reference': ''
    }
}

class ODEFunc(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim)
        )

    def forward(self, t, y):
        return self.net(y)

def load_ground_truth_from_csv(scenario_name):
    path = f"scenario_data/{scenario_name}_ground_truth.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    df = pd.read_csv(path)
    t = df["t"].values
    V = df["V"].values.reshape(-1, 1)
    N = df["N"].values.reshape(-1, 1)
    scaler_V = MinMaxScaler()
    scaler_N = MinMaxScaler()
    V_scaled = scaler_V.fit_transform(V)
    N_scaled = scaler_N.fit_transform(N)
    t = (t - t.min()) / (t.max() - t.min())
    y_scaled = np.hstack([V_scaled, N_scaled])
    return t, y_scaled, scaler_V, scaler_N

def clean_memory(*args):
    for obj in args:
        del obj
    torch.cuda.empty_cache()
    gc.collect()

def train_node(scenario_name, t_np, y_np, epochs, lr=1e-4, patience=2000, delta=1e-6, verbose_step=500):
    if len(t_np) > 200:
        t_np = t_np[::5]
        y_np = y_np[::5]

    t_tensor = torch.tensor(t_np, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_np, dtype=torch.float32).to(device)

    model = ODEFunc().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    counter = 0
    loss_list = []

    progress = trange(epochs, desc=f"🧠 Training {scenario_name} ({epochs} ep)", leave=False)

    for epoch in progress:
        optimizer.zero_grad()
        y0 = y_tensor[0]
        try:
            pred_y = odeint(model, y0, t_tensor, method='dopri5')
        except Exception as e:
            print(f"❌ Integration failed at epoch {epoch}: {e}")
            break

        loss = loss_fn(pred_y, y_tensor)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        loss_list.append(loss_value)
        progress.set_postfix(loss=loss_value)

        if epoch % verbose_step == 0:
            print(f"📉 Epoch {epoch:5d} | Loss = {loss_value:.4f}")

        if loss_value + delta < best_loss:
            best_loss = loss_value
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch} (no improvement in {patience} steps)")
                break

    return model, loss_list

# ==== EXECUTION ====
all_results = []
epoch_list = [10000]

for scenario_name in scenarios:
    print(f"🔁 Scenario: {scenario_name.upper()}")
    t_np, y_np, scaler_V, scaler_N = load_ground_truth_from_csv(scenario_name)
    t_tensor = torch.tensor(t_np, dtype=torch.float32).to(device)

    best_loss = float("inf")
    best_model_state = None
    best_epoch = None

    for epochs in epoch_list:
        print(f"▶ Training NODE - {epochs} epochs")
        model, loss_list = train_node(scenario_name, t_np, y_np, epochs)

        with torch.no_grad():
            y0 = torch.tensor(y_np[0], dtype=torch.float32).to(device)
            pred = odeint(model, y0, t_tensor, method='dopri5').cpu().numpy()
            pred_unscaled = np.hstack([
                scaler_V.inverse_transform(pred[:, 0].reshape(-1, 1)),
                scaler_N.inverse_transform(pred[:, 1].reshape(-1, 1))
            ])
            y_unscaled = np.hstack([
                scaler_V.inverse_transform(y_np[:, 0].reshape(-1, 1)),
                scaler_N.inverse_transform(y_np[:, 1].reshape(-1, 1))
            ])

        # === ΜΟΝΟ V(t) ===
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle(f"{scenario_name.upper()}/NODE/{epochs} Epochs/I=50", fontsize=14)

        ax.plot(t_np, y_unscaled[:, 0], label='True V')
        ax.plot(t_np, pred_unscaled[:, 0], '--', label='NODE V')
        ax.set_title('V(t)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Voltage V')
        ax.legend()
        ax.grid()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"plots_node/{scenario_name}_{epochs}_NODE.eps", format='eps', dpi=600)
        plt.close()

        clean_memory(model, loss_list, pred, y0)

print("\n✅ All NODE training complete.")
