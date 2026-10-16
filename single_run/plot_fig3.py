"""
plot_fig3.py — Figure 3 of the paper (Hopf parameter set at I = 50, below the onset of
oscillations). Re-plots the stored PINN and NODE models trained for 10,000 epochs by
Pinns_50_final.py and NODE_50_final.py (no retraining). Run from inside single_run/.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from torchdiffeq import odeint

def mlp(nin):
    return nn.Sequential(nn.Linear(nin, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(),
                         nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))

def load(path, nin):
    net = mlp(nin)
    sd = torch.load(path, map_location="cpu")
    net.load_state_dict({k.replace("net.", "", 1): v for k, v in sd.items() if "total_" not in k})
    return net.eval()

df = pd.read_csv("scenario_data/hopf_50_ground_truth.csv")
t, y = df["t"].values, df[["V", "N"]].values

def pinn_pred(epochs):
    net = load(f"models_pinn/hopf_50_{epochs}_model.pth", 1)
    with torch.no_grad():
        return net(torch.tensor(t.reshape(-1, 1), dtype=torch.float32)).numpy()

def node_pred(epochs):
    net = load(f"models_node/hopf_50_{epochs}_model.pth", 2)
    lo, hi = y.min(0), y.max(0)                       # Min-Max scaling as in NODE_50_final.py
    ys = (y - lo) / (hi - lo); ts = (t - t.min()) / (t.max() - t.min())
    with torch.no_grad():
        p = odeint(lambda s, z: net(z), torch.tensor(ys[0], dtype=torch.float32),
                   torch.tensor(ts, dtype=torch.float32), method="dopri5").numpy()
    return p * (hi - lo) + lo

if __name__ == "__main__":
    for e in [1000, 2000, 5000, 10000]:
        for name, f in [("PINN", pinn_pred), ("NODE", node_pred)]:
            try:
                p = f(e)
                print(f"{name} {e:>5} epochs: MSE_V = {mean_squared_error(y[:, 0], p[:, 0]):.4g}, "
                      f"R2_V = {r2_score(y[:, 0], p[:, 0]):.5f}")
            except FileNotFoundError:
                pass
    for name, f in [("PINN", pinn_pred), ("NODE", node_pred)]:
        p = f(10000)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, y[:, 0], lw=2, label="Ground truth")
        ax.plot(t, p[:, 0], "--", lw=2, label=name)
        ax.set_title(f"{name}, 10,000 epochs", fontsize=14)
        ax.set_xlabel("$t$ (ms)", fontsize=12); ax.set_ylabel("$V$ (mV)", fontsize=12)
        ax.legend(fontsize=11); ax.grid(color="0.85")
        fig.tight_layout()
        fig.savefig(f"hopf_50_10000_{name}.eps", format="eps"); fig.savefig(f"hopf_50_10000_{name}.png", dpi=150)
        plt.close(fig)
