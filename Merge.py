import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === STEP 1: FILE CHECK ===
NODE_METRICS = "metrics/metrics_node_summary.csv"
PINN_METRICS = "metrics/metrics_pinn_summary.csv"

if not os.path.exists(NODE_METRICS) or not os.path.exists(PINN_METRICS):
    raise FileNotFoundError("❌ Missing metrics. Please run NODE_all_azizi.py and Pinns_all_azizi.py first.")

# === STEP 2: LOAD DATA ===
node_df = pd.read_csv(NODE_METRICS)
pinn_df = pd.read_csv(PINN_METRICS)

node_df["Model"] = "NODE"
pinn_df["Model"] = "PINN"
combined_df = pd.concat([node_df, pinn_df], ignore_index=True)

# === STEP 3: TOTAL MSE ===
combined_df["Total_MSE"] = combined_df["MSE_V"] + combined_df["MSE_N"]
combined_df.to_csv("metrics/combined_summary.csv", index=False)

# === STEP 4: PLOTS ===
sns.set_theme(style="whitegrid", font_scale=1.2)
os.makedirs("plots_comparison", exist_ok=True)

# -- Barplots for RMSE_V and RMSE_N
for metric in ["RMSE_V", "RMSE_N"]:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=combined_df, x="Scenario", y=metric, hue="Model", palette="tab10")
    plt.title(f"{metric} per Scenario")
    plt.xlabel("Scenario")
    plt.ylabel(metric)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"plots_comparison/barplot_{metric}.eps", format="eps", dpi=600)
    plt.savefig(f"plots_comparison/barplot_{metric}.png", format="png", dpi=600)
    plt.close()

# -- Lineplot of Total MSE vs Epochs
for scenario in combined_df["Scenario"].unique():
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=combined_df[combined_df["Scenario"] == scenario],
        x="Epochs", y="Total_MSE", hue="Model", marker="o", style="Model", palette="tab10"
    )
    plt.title(f"Total MSE vs Epochs — {scenario.capitalize()}")
    plt.xlabel("Epochs")
    plt.ylabel("Total MSE")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"plots_comparison/lineplot_total_mse_{scenario.lower()}.eps", format='eps', dpi=600)
    plt.savefig(f"plots_comparison/lineplot_total_mse_{scenario.lower()}.png", format='png', dpi=600)
    plt.close()

print("✅ Comparative analysis completed.")
