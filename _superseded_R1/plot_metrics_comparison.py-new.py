import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Αρχεία CSV ===
pinn_csv = "metrics/metrics_pinn_summary.csv"
node_csv = "metrics/metrics_node_summary.csv"
save_dir = "plots_comparison/"
os.makedirs(save_dir, exist_ok=True)

# === Διαβάζουμε δεδομένα ===
pinn = pd.read_csv(pinn_csv)
node = pd.read_csv(node_csv)

pinn["Model"] = "PINN"
node["Model"] = "NODE"

# === Συνένωση δεδομένων ===
data = pd.concat([pinn, node], ignore_index=True)

# === Πλοκή RMSE_V vs Training Time για κάθε Scenario ===
for scenario in data["Scenario"].unique():
    df = data[data["Scenario"] == scenario]

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="Training_Time_Sec",
        y="RMSE_V",
        hue="Model",
        marker="o"
    )
    plt.title(f"RMSE$_V$ vs Training Time – {scenario.capitalize()} Scenario")
    plt.xlabel("Training Time (sec)")
    plt.ylabel("RMSE$_V$")
    plt.grid(True)
    plt.tight_layout()
    fname = scenario.lower() + "_rmse_vs_time"
    plt.savefig(os.path.join(save_dir, f"{fname}.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"{fname}.eps"))
    plt.close()

print("Τα plots αποθηκεύτηκαν στον φάκελο:", save_dir)
