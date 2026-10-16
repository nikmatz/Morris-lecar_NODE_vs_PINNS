"""
make_r2_outputs.py: builds every new figure and table of Revision 2 from results_R2/.
Run from Code_final/:  python make_r2_outputs.py
Outputs: r2_figures/*.eps|png, r2_figures/tables.tex, r2_figures/numbers.json
"""
import os, json, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results_R2")
OUT = os.path.join(ROOT, "r2_figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9})
SC = ["hopf", "snlc", "homoclinic"]
SCN = {"hopf": "Hopf", "snlc": "SNLC", "homoclinic": "Homoclinic"}
NUM = {}


def load(exp):
    R = []
    for f in sorted(glob.glob(os.path.join(RES, exp, "*.json"))):
        r = json.load(open(f)); r["_npz"] = f[:-5] + ".npz"; R.append(r)
    return R


def ms(x, d=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f} \\pm {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".eps"), format="eps", bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


seeds, noise, pcol, grid, main, act, tol, gridh = (load(e) for e in ["seeds", "noise", "pinncol", "grid", "main", "act", "tol", "gridhopf"])
TEX = []

# --- Table: robustness to initialisation (5 seeds), train & test windows ---
rows = []
for sc in SC:
    for lab, runs in [("PINN", [r for r in seeds if r["scenario"] == sc and r["model"] == "pinn"]),
                      ("PINN$^{\\dagger}$", [r for r in pcol if r["scenario"] == sc]),
                      ("NODE", [r for r in seeds if r["scenario"] == sc and r["model"] == "node"])]:
        trR = [r["train"]["R2_V"] for r in runs]; teR = [r["test"]["R2_V"] for r in runs]
        trE = [r["train"]["RMSE_V"] for r in runs]; teE = [r["test"]["RMSE_V"] for r in runs]
        rows.append((SCN[sc], lab, len(runs), ms(trE, 2), ms(trR), ms(teE, 2), ms(teR)))
        key = f"seeds_{sc}_{lab.split('$')[0].lower()}{'col' if 'dagger' in lab else ''}"
        NUM[key] = dict(n=len(runs), trR2=[np.mean(trR), np.std(trR, ddof=1)], teR2=[np.mean(teR), np.std(teR, ddof=1)],
                        teR2_all=sorted(teR), trR2_all=sorted(trR), trE=[np.mean(trE), np.std(trE, ddof=1)], teE=[np.mean(teE), np.std(teE, ddof=1)])
t = ["\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
     "& & \\multicolumn{2}{c}{\\textbf{Training window} $[0,200]$\\,ms} & \\multicolumn{2}{c}{\\textbf{Extrapolation window} $(200,300]$\\,ms} \\\\",
     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
     "\\textbf{Regime} & \\textbf{Model} & RMSE$_V$ (mV) & $R^2_V$ & RMSE$_V$ (mV) & $R^2_V$ \\\\", "\\midrule"]
for i, (a, b, n, c, d, e, f) in enumerate(rows):
    t.append(f"{a if i % 3 == 0 else ''} & {b} & ${c}$ & ${d}$ & ${e}$ & ${f}$ \\\\")
    if i % 3 == 2 and i < len(rows) - 1: t.append("\\midrule")
t += ["\\bottomrule", "\\end{tabular}"]
TEX.append("%% TABLE seeds\n" + "\n".join(t))

# --- Table: robustness to noise ---
t = ["\\begin{tabular}{@{}llcccccc@{}}", "\\toprule",
     "& & \\multicolumn{3}{c}{\\textbf{Training window}, $R^2_V$} & \\multicolumn{3}{c}{\\textbf{Extrapolation window}, $R^2_V$} \\\\",
     "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
     "\\textbf{Regime} & \\textbf{Model} & $\\sigma=0$ & $1\\%$ & $5\\%$ & $\\sigma=0$ & $1\\%$ & $5\\%$ \\\\", "\\midrule"]
for sc in SC:
    for k, m in enumerate(["pinn", "node"]):
        cells_tr, cells_te = [], []
        for nz in [0.0, 0.01, 0.05]:
            runs = [r for r in (seeds if nz == 0 else noise) if r["scenario"] == sc and r["model"] == m and abs(r["noise"] - nz) < 1e-9]
            tr = [r["train"]["R2_V"] for r in runs]; te = [r["test"]["R2_V"] for r in runs]
            cells_tr.append(f"${ms(tr, 2)}$"); cells_te.append(f"${ms(te, 2)}$")
            NUM[f"noise_{sc}_{m}_{int(nz*100)}"] = dict(n=len(runs), trR2=[np.mean(tr), np.std(tr, ddof=1)], teR2=[np.mean(te), np.std(te, ddof=1)])
        t.append(f"{SCN[sc] if k == 0 else ''} & {m.upper()} & " + " & ".join(cells_tr + cells_te) + " \\\\")
    if sc != "homoclinic": t.append("\\midrule")
t += ["\\bottomrule", "\\end{tabular}"]
TEX.append("%% TABLE noise\n" + "\n".join(t))

# --- Grid: heatmaps + table (Hopf and Homoclinic) ---
D, W = [2, 3, 4], [16, 32, 64, 128, 256]
fig, axs = plt.subplots(2, 2, figsize=(9, 6.0))
t = ["\\begin{tabular}{@{}lllccccc@{}}", "\\toprule",
     "\\textbf{Regime} & \\textbf{Model} & \\textbf{Depth} & \\multicolumn{5}{c}{\\textbf{Width}} \\\\", "\\cmidrule(lr){4-8}",
     " & & & 16 & 32 & 64 & 128 & 256 \\\\", "\\midrule"]
panel = iter("abcd")
for ri, (sc, G) in enumerate([("hopf", gridh), ("homoclinic", grid)]):
    for ci, m in enumerate(["pinn", "node"]):
        ax = axs[ri, ci]
        M = np.zeros((3, 5)); S = np.zeros((3, 5))
        for i, d in enumerate(D):
            cells = []
            for k, w in enumerate(W):
                v = [r["train"]["R2_V"] for r in G if r["model"] == m and r["depth"] == d and r["width"] == w]
                M[i, k], S[i, k] = np.mean(v), np.std(v, ddof=1)
                cells.append(f"${M[i,k]:.2f} \\pm {S[i,k]:.2f}$")
            lab_sc = SCN[sc] if (ci == 0 and i == 0) else ""
            t.append(f"{lab_sc} & {m.upper() if i == 0 else ''} & {d} & " + " & ".join(cells) + " \\\\")
        if not (ri == 1 and ci == 1): t.append("\\midrule" if ci == 1 else "\\cmidrule(lr){2-8}")
        key = f"grid_{m}" if sc == "homoclinic" else f"gridhopf_{m}"
        NUM[key] = dict(mean=M.round(3).tolist(), std=S.round(3).tolist())
        te = [r["test"]["R2_V"] for r in G if r["model"] == m]
        NUM[key + "_test_range"] = [float(np.min(te)), float(np.max(te))]
        tr = [r["train"]["R2_V"] for r in G if r["model"] == m]
        NUM[key + "_train_range"] = [float(np.min(tr)), float(np.max(tr))]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", origin="lower", aspect="auto")
        for i in range(3):
            for k in range(5):
                ax.text(k, i, f"{M[i,k]:.2f}\n±{S[i,k]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[i, k] < 0.6 else "k")
        ax.set_xticks(range(5)); ax.set_xticklabels(W); ax.set_yticks(range(3)); ax.set_yticklabels(D)
        ax.set_xlabel("width (neurons per layer)"); ax.set_ylabel("depth (hidden layers)")
        ax.set_title(f"({next(panel)}) {SCN[sc]}, {m.upper()}: $R^2_V$ (training window)", fontsize=9)
fig.tight_layout(); fig.colorbar(im, ax=axs, shrink=0.8, label="$R^2_V$ (mean over 3 seeds)")
save(fig, "R2_grid_heatmap")
t += ["\\bottomrule", "\\end{tabular}"]
TEX.append("%% TABLE grid\n" + "\n".join(t))
NUM["grid_params"] = {f"{r['model']}_d{r['depth']}w{r['width']}": r["params"] for r in grid if r["seed"] == 42}

# --- Activation / tolerance checks (NODE, homoclinic, full window, 10k epochs) ---
for lab, runs in [("tanh", [r for r in act if r["act"] == "tanh"]), ("silu", [r for r in act if r["act"] == "silu"]), ("tol", tol)]:
    v = [r["full"]["R2_V"] for r in runs]
    NUM[f"act_{lab}"] = dict(n=len(v), R2=[np.mean(v), np.std(v, ddof=1)], all={r["seed"]: r["full"]["R2_V"] for r in runs},
                             loss=[np.mean([r["final_loss"] for r in runs]), np.std([r["final_loss"] for r in runs], ddof=1)])
from scipy import stats
a = [r["full"]["R2_V"] for r in act if r["act"] == "silu"]; b = [r["full"]["R2_V"] for r in act if r["act"] == "tanh"]
NUM["act_welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
NUM["act_mwu_p"] = float(stats.mannwhitneyu(a, b).pvalue)
tt = {r["seed"]: r["full"]["R2_V"] for r in tol}; ta = {r["seed"]: r["full"]["R2_V"] for r in act if r["act"] == "tanh"}
NUM["tol_paired_p"] = float(stats.ttest_rel([tt[s] for s in sorted(tt)], [ta[s] for s in sorted(tt)]).pvalue)

# --- Homoclinic main (I=35.5) epoch table rows (Appendix A) ---
mrows = {}
for r in main:
    m = r["full"]; key = (r["model"], r["act"], r["epochs"])
    mrows[key] = dict(total=m["MSE_V"] + m["MSE_N"], MAE_V=m["MAE_V"], MAE_N=m["MAE_N"], R2_V=m["R2_V"], R2_N=m["R2_N"],
                      MaxErr_V=m["MaxErr_V"], MaxErr_N=m["MaxErr_N"], RMSE_V=m["RMSE_V"], RMSE_N=m["RMSE_N"], time=r["train_time_s"])
NUM["main"] = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in mrows.items()}
for m in ["pinn", "node"]:
    L = []
    for e in [1000, 2000, 5000, 10000, 20000]:
        v = mrows[(m, "tanh", e)]
        L.append(f"homoclinic$^{{\\ast}}$ & {e} & {v['total']:.3f} & {v['MAE_V']:.3f} & {v['MAE_N']:.3f} & {v['R2_V']:.3f} & {v['R2_N']:.3f} & {v['MaxErr_V']:.3f} & {v['MaxErr_N']:.3f} & {v['time']:.1f}$^{{\\ddagger}}$ \\\\")
    TEX.append(f"%% APPENDIX rows {m}\n" + "\n".join(L))

# --- Total-MSE-vs-epochs figure (Hopf, SNLC from R1 CSVs; Homoclinic from R2) ---
P = pd.read_csv(os.path.join(ROOT, "metrics", "metrics_pinn_summary.csv")); N = pd.read_csv(os.path.join(ROOT, "metrics", "metrics_node_summary.csv"))
E = [1000, 2000, 5000, 10000, 20000]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
for ax, sc in zip(axs, SC):
    if sc == "homoclinic":
        yp = [mrows[("pinn", "tanh", e)]["total"] for e in E]; yn = [mrows[("node", "tanh", e)]["total"] for e in E]
    else:
        yp = [float(P[(P.Scenario == sc) & (P.Epochs == e)].Total_MSE.iloc[0]) for e in E]
        yn = [float(N[(N.Scenario == sc) & (N.Epochs == e)].Total_MSE.iloc[0]) for e in E]
    ax.plot(E, yp, "o-", color="tab:orange", label="PINN"); ax.plot(E, yn, "x-", color="tab:blue", label="NODE")
    ax.set_yscale("log"); ax.set_xlabel("Epochs"); ax.set_title(f"{SCN[sc]}" + (" ($I=35.5$)" if sc == "homoclinic" else "")); ax.grid(alpha=.3)
axs[0].set_ylabel("Total MSE ($\\mathrm{MSE}_V+\\mathrm{MSE}_N$)"); axs[0].legend()
fig.tight_layout(); save(fig, "R2_total_mse_epochs")

# --- Appendix-B style training-progression figures (homoclinic I=35.5) ---
def progression(model, act_, e):
    r = [x for x in main if x["model"] == model and x["act"] == act_ and x["epochs"] == e][0]
    z = np.load(r["_npz"]); t = z["t"]; y = z["y_true"]; p = z["pred"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    lab = model.upper() + ("" if act_ == "tanh" else " (SiLU)")
    fig.suptitle(f"HOMOCLINIC ($I=35.5$) - {lab} - {e} Epochs", fontsize=10)
    ax[0, 0].plot(t, y[:, 0], label="True V"); ax[0, 0].plot(t, p[:, 0], "--", label=f"{model.upper()} V"); ax[0, 0].set_title("V(t)")
    ax[0, 1].plot(t, y[:, 1], label="True N"); ax[0, 1].plot(t, p[:, 1], "--", label=f"{model.upper()} N"); ax[0, 1].set_title("N(t)")
    ax[1, 0].plot(y[:, 0], y[:, 1], label="True"); ax[1, 0].plot(p[:, 0], p[:, 1], "--", label=model.upper()); ax[1, 0].set_title("Phase Portrait")
    ax[1, 1].plot(np.arange(len(z["loss"])) * 10, z["loss"]); ax[1, 1].set_yscale("log"); ax[1, 1].set_title("Training Loss"); ax[1, 1].set_xlabel("Epoch")
    for a_ in ax.ravel(): a_.grid(alpha=.3)
    for a_ in [ax[0, 0], ax[0, 1], ax[1, 0]]: a_.legend(fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, f"R2_homoclinic355_{e}_{model.upper()}{'' if act_ == 'tanh' else '_SiLU'}")
for m in ["pinn", "node"]:
    for e in [1000, 20000]: progression(m, "tanh", e)

# --- Activation figure: Tanh vs SiLU (seed 42 trajectories + R2 over seeds) ---
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw=dict(width_ratios=[2.3, 1]))
z_t = np.load([r for r in act if r["act"] == "tanh" and r["seed"] == 42][0]["_npz"]); z_s = np.load([r for r in act if r["act"] == "silu" and r["seed"] == 42][0]["_npz"])
ax[0].plot(z_t["t"], z_t["y_true"][:, 0], "k", lw=1.2, label="ground truth")
ax[0].plot(z_t["t"], z_t["pred"][:, 0], "--", color="tab:blue", label="NODE, Tanh")
ax[0].plot(z_s["t"], z_s["pred"][:, 0], "--", color="tab:red", label="NODE, SiLU")
ax[0].set_xlabel("$t$ (ms)"); ax[0].set_ylabel("$V$ (mV)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[0].set_title("(a) Homoclinic regime, seed 42, 10 000 epochs")
groups = [("Tanh", [r["full"]["R2_V"] for r in act if r["act"] == "tanh"]), ("SiLU", [r["full"]["R2_V"] for r in act if r["act"] == "silu"]),
          ("Tanh, tight\ntolerance", [r["full"]["R2_V"] for r in tol])]
for k, (g, v) in enumerate(groups):
    ax[1].scatter(np.full(len(v), k) + np.linspace(-.08, .08, len(v)), v, s=18, color=["tab:blue", "tab:red", "tab:gray"][k])
    ax[1].errorbar(k + .25, np.mean(v), yerr=np.std(v, ddof=1), fmt="s", color="k", ms=4, capsize=3)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels([g for g, _ in groups], fontsize=8); ax[1].set_ylim(0, 1); ax[1].set_ylabel("$R^2_V$")
ax[1].set_title("(b) NODE $R^2_V$ across seeds"); ax[1].grid(alpha=.3, axis="y")
fig.tight_layout(); save(fig, "R2_homoclinic_activation")

# --- Extrapolation figure (seed 42): PINN, PINN with full collocation, NODE ---
fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
for ax, sc in zip(axs, SC):
    zp = np.load([r for r in seeds if r["scenario"] == sc and r["model"] == "pinn" and r["seed"] == 42][0]["_npz"])
    zc = np.load([r for r in pcol if r["scenario"] == sc and r["seed"] == 42][0]["_npz"])
    zn = np.load([r for r in seeds if r["scenario"] == sc and r["model"] == "node" and r["seed"] == 42][0]["_npz"])
    ax.axvspan(200, 300, color="0.92")
    ax.plot(zp["t"], zp["y_true"][:, 0], "k", lw=1.3, label="ground truth")
    ax.plot(zp["t"], zp["pred"][:, 0], "--", color="tab:orange", lw=1.1, label="PINN")
    ax.plot(zc["t"], zc["pred"][:, 0], ":", color="tab:purple", lw=1.4, label="PINN$^{\\dagger}$ (physics residual on $[0,300]$ ms)")
    ax.plot(zn["t"], zn["pred"][:, 0], "-.", color="tab:blue", lw=1.1, label="NODE")
    ax.set_ylabel("$V$ (mV)"); ax.set_title({"hopf": "(a) Hopf, $I=90$", "snlc": "(b) SNLC, $I=42$", "homoclinic": "(c) Homoclinic, $I=35.5$"}[sc], fontsize=9)
    ax.grid(alpha=.3)
axs[0].legend(fontsize=7, ncol=4, loc="lower left"); axs[-1].set_xlabel("$t$ (ms)")
fig.tight_layout(); save(fig, "R2_extrapolation_seed42")

open(os.path.join(OUT, "tables.tex"), "w").write("\n\n".join(TEX))
json.dump(NUM, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)
print("done:", sorted(os.listdir(OUT)))
