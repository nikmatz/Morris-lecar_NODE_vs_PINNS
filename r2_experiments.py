"""
r2_experiments.py: Experiments for Revision 2 (AIDE-D-25-00346R1)
====================================================================

One script for all new experiments requested in the second review round.

Experiments (--exp):
  main   Homoclinic regime at the corrected current I = 35.5 uA/cm^2, original
         protocol (train on the full window t in [0,300] ms, epochs 1000-20000).
         PINN (Tanh), NODE (Tanh), NODE (SiLU). Replaces the old I = 50 results.
  seeds  Robustness to initialisation (Reviewer 2, comment 1a).
         3 regimes x {PINN, NODE} x 5 seeds, train on [0,200] ms,
         evaluate in-window [0,200] and extrapolation window (200,300].
  noise  Robustness to noisy data (Reviewer 2, comment 1b).
         Same as `seeds`, with Gaussian noise of std = 1% and 5% of the range
         of each variable added to the training targets. Evaluation is always
         against the clean ground truth.
  act    Activation check (Tanh vs SiLU) for the NODE in the Homoclinic regime:
         5 seeds x {tanh, silu}, original protocol (full window [0,300] ms),
         10000 epochs. Tests whether SiLU helps beyond run-to-run variance.
  tol    Solver-tolerance check (audit): NODE, Homoclinic, Tanh, 3 seeds,
         full window, 10000 epochs, training tolerances rtol=1e-6, atol=1e-7
         (default training tolerances rtol=1e-4, atol=1e-5 are too loose for
         the Homoclinic trajectory). Compare with `act` (Tanh).
  pinncol  Fair PINN extrapolation (audit): data loss only on t <= 200 ms, but
         physics-residual collocation points on the whole window [0,300] ms
         (no test data used). 3 regimes x 5 seeds, 10000 epochs. Compare with
         the PINN rows of `seeds`.
  gridhopf  Same factorial study as `grid`, in the Hopf regime (where the NODE
         trains successfully): 3 depths x 5 widths x 3 seeds x {PINN, NODE}.
  grid   Factorial architecture study (Reviewer 2, comment 2).
         depth {2,3,4} x width {16,32,64,128,256} x 3 seeds x {PINN, NODE},
         Homoclinic regime, fixed budget of 5000 epochs, Adam lr = 1e-3.

Every run is saved as one JSON (config + metrics) and one NPZ (predictions,
loss curve) in results_R2/<exp>/. Finished runs are skipped, so the script can
be stopped and restarted at any time. Several copies can run in parallel:
each copy claims the next free job with a lock file.

Usage (from inside Code_final/):
    python r2_experiments.py --exp main
    python r2_experiments.py --exp seeds
    python r2_experiments.py --exp noise
    python r2_experiments.py --exp grid
    python r2_experiments.py --exp act
    python r2_experiments.py --exp tol
    python r2_experiments.py --exp pinncol
    python r2_experiments.py --exp gridhopf
    python r2_experiments.py --exp seeds --model pinn      # only PINN jobs
    python r2_experiments.py --exp all                     # everything, in order
    python r2_experiments.py --list --exp all              # show progress only
    python r2_experiments.py --reset-claims                # after a crash
    python r2_experiments.py --summarize                   # tables (mean +- std)

Parallel: open 3-4 terminals and run the same command in each.
Recommended: --threads 2 per process (default).
"""

import argparse, json, os, sys, time, random, glob, itertools
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results_R2")

# --- Scenarios ---
BASE = dict(C=20.0, g_K=8.0, g_L=2.0, V_Ca=120.0, V_K=-84.0, V_L=-60.0, V1=-1.2, V2=18.0)
SCENARIOS = {
    "hopf":       dict(params=dict(BASE, g_Ca=4.4, phi=0.04,  V3=2.0,  V4=30.0, I=90.0), y0=[0.0, 0.0]),
    "snlc":       dict(params=dict(BASE, g_Ca=4.0, phi=0.067, V3=12.0, V4=17.4, I=42.0), y0=[0.0, 0.0]),
    # Corrected homoclinic current (R2): stable limit cycle passing near the saddle;
    # homoclinic bifurcation at I ~ 35.0. y0 must lie in the basin of the cycle
    # (y0 = [0,0] converges to the coexisting resting state).
    "homoclinic": dict(params=dict(BASE, g_Ca=4.0, phi=0.23,  V3=12.0, V4=17.4, I=35.5), y0=[10.0, 0.3]),
}
T_END, N_PTS, T_TRAIN = 300.0, 3000, 200.0

# Experiment settings
SEEDS5 = [42, 43, 44, 45, 46]
SEEDS3 = [42, 43, 44]
EPOCHS_ROBUST = 10000          # same budget as the extrapolation study of R1
EPOCHS_GRID = 5000
DEPTHS = [2, 3, 4]
WIDTHS = [16, 32, 64, 128, 256]
NOISE_LEVELS = [0.01, 0.05]
MAIN_EPOCHS = [1000, 2000, 5000, 10000, 20000]
MAIN_EPOCHS_SILU = [1000, 20000]
GRID_SCENARIO = "homoclinic"


# --- Ground truth ---
def ml_rhs(t, y, p):
    V, N = y
    m = 0.5 * (1 + np.tanh((V - p['V1']) / p['V2']))
    n = 0.5 * (1 + np.tanh((V - p['V3']) / p['V4']))
    tau = 1.0 / np.cosh((V - p['V3']) / (2 * p['V4']))
    dV = (-p['g_Ca'] * m * (V - p['V_Ca']) - p['g_K'] * N * (V - p['V_K']) - p['g_L'] * (V - p['V_L']) + p['I']) / p['C']
    dN = p['phi'] * (n - N) / tau
    return [dV, dN]


def ground_truth(name):
    """Same settings as synthetic_M_L_data.py; cached as CSV in results_R2/data."""
    import pandas as pd
    from scipy.integrate import solve_ivp
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    f = os.path.join(OUT, "data", f"{name}_ground_truth.csv")
    if not os.path.exists(f):
        sc = SCENARIOS[name]
        t_eval = np.linspace(0, T_END, N_PTS)
        sol = solve_ivp(ml_rhs, (0, 600), sc['y0'], args=(sc['params'],), t_eval=t_eval, rtol=1e-6, atol=1e-8)
        pd.DataFrame(dict(t=sol.t, V=sol.y[0], N=sol.y[1])).to_csv(f, index=False)
    d = pd.read_csv(f)
    return d['t'].values.astype(np.float32), d[['V', 'N']].values.astype(np.float32)


# --- Job list ---
def jobs_for(exp):
    J = []
    if exp == "main":
        for e in MAIN_EPOCHS:
            J.append(dict(exp="main", scenario="homoclinic", model="pinn", act="tanh", depth=3, width=128, seed=42, noise=0.0, epochs=e, window="full"))
        for e in MAIN_EPOCHS:
            J.append(dict(exp="main", scenario="homoclinic", model="node", act="tanh", depth=3, width=128, seed=42, noise=0.0, epochs=e, window="full"))
        for e in MAIN_EPOCHS_SILU:
            J.append(dict(exp="main", scenario="homoclinic", model="node", act="silu", depth=3, width=128, seed=42, noise=0.0, epochs=e, window="full"))
    elif exp == "act":
        for s, act in itertools.product(SEEDS5, ["tanh", "silu"]):
            J.append(dict(exp="act", scenario="homoclinic", model="node", act=act, depth=3, width=128, seed=s, noise=0.0, epochs=EPOCHS_ROBUST, window="full"))
    elif exp == "tol":
        for s in SEEDS3:
            J.append(dict(exp="tol", scenario="homoclinic", model="node", act="tanh", depth=3, width=128, seed=s, noise=0.0, epochs=EPOCHS_ROBUST, window="full", rtol=1e-6, atol=1e-7))
    elif exp == "pinncol":
        for s, sc in itertools.product(SEEDS5, SCENARIOS):
            J.append(dict(exp="pinncol", scenario=sc, model="pinn", act="tanh", depth=3, width=128, seed=s, noise=0.0, epochs=EPOCHS_ROBUST, window="extrap", colloc="full"))
    elif exp == "seeds":
        for s, sc, m in itertools.product(SEEDS5, SCENARIOS, ["pinn", "node"]):
            J.append(dict(exp="seeds", scenario=sc, model=m, act="tanh", depth=3, width=128, seed=s, noise=0.0, epochs=EPOCHS_ROBUST, window="extrap"))
    elif exp == "noise":
        for nz, s, sc, m in itertools.product(NOISE_LEVELS, SEEDS5, SCENARIOS, ["pinn", "node"]):
            J.append(dict(exp="noise", scenario=sc, model=m, act="tanh", depth=3, width=128, seed=s, noise=nz, epochs=EPOCHS_ROBUST, window="extrap"))
    elif exp == "gridhopf":
        for s, d, w, m in itertools.product(SEEDS3, DEPTHS, WIDTHS, ["pinn", "node"]):
            J.append(dict(exp="gridhopf", scenario="hopf", model=m, act="tanh", depth=d, width=w, seed=s, noise=0.0, epochs=EPOCHS_GRID, window="extrap"))
    elif exp == "grid":
        for s, d, w, m in itertools.product(SEEDS3, DEPTHS, WIDTHS, ["pinn", "node"]):
            J.append(dict(exp="grid", scenario=GRID_SCENARIO, model=m, act="tanh", depth=d, width=w, seed=s, noise=0.0, epochs=EPOCHS_GRID, window="extrap"))
    return J


def job_id(j):
    return (f"{j['exp']}_{j['scenario']}_{j['model']}_{j['act']}_d{j['depth']}w{j['width']}"
            f"_s{j['seed']}_n{int(round(j['noise']*100)):02d}_e{j['epochs']}"
            + (f"_rtol{j['rtol']:g}" if 'rtol' in j else "")
            + (f"_col{j['colloc']}" if 'colloc' in j else ""))


def paths(j):
    d = os.path.join(OUT, j['exp'])
    os.makedirs(d, exist_ok=True)
    b = os.path.join(d, job_id(j))
    return b + ".json", b + ".npz", b + ".claim"


def claim(j):
    res, _, cl = paths(j)
    if os.path.exists(res):
        return False
    try:
        fd = os.open(cl, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.ctime()}".encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


# --- Models ---
def mlp(nin, depth, width, act):
    import torch.nn as nn
    A = {"tanh": nn.Tanh, "silu": nn.SiLU}[act]
    layers = [nn.Linear(nin, width), A()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), A()]
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)


def residual_loss(torch, t, pred, p):
    V, N = pred[:, 0:1], pred[:, 1:2]
    dVdt = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dNdt = torch.autograd.grad(N, t, grad_outputs=torch.ones_like(N), create_graph=True)[0]
    M_inf = 0.5 * (1 + torch.tanh((V - p['V1']) / p['V2']))
    N_inf = 0.5 * (1 + torch.tanh((V - p['V3']) / p['V4']))
    tau_N = 1.0 / torch.cosh((V - p['V3']) / (2 * p['V4']))
    f_V = (-p['g_Ca'] * M_inf * (V - p['V_Ca']) - p['g_K'] * N * (V - p['V_K']) - p['g_L'] * (V - p['V_L']) + p['I']) / p['C']
    f_N = p['phi'] * (N_inf - N) / tau_N
    return torch.mean((dVdt - f_V) ** 2 + (dNdt - f_N) ** 2)


def metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    out = {}
    for k, lab in enumerate("VN"):
        mse = float(mean_squared_error(y_true[:, k], y_pred[:, k]))
        out[f"MSE_{lab}"] = mse
        out[f"RMSE_{lab}"] = float(np.sqrt(mse))
        out[f"MAE_{lab}"] = float(mean_absolute_error(y_true[:, k], y_pred[:, k]))
        out[f"R2_{lab}"] = float(r2_score(y_true[:, k], y_pred[:, k]))
        out[f"MaxErr_{lab}"] = float(np.max(np.abs(y_true[:, k] - y_pred[:, k])))
    out["Total_MSE"] = out["MSE_V"] + out["MSE_N"]
    return out


# --- One run ---
def run(j, threads):
    import torch
    from torchdiffeq import odeint
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    s = j['seed']
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    rng = np.random.default_rng(1000 + s)          # noise stream independent of init

    p = SCENARIOS[j['scenario']]['params']
    t_all, y_all = ground_truth(j['scenario'])
    tr = np.ones_like(t_all, bool) if j['window'] == "full" else (t_all <= T_TRAIN)
    t_tr, y_tr_clean = t_all[tr], y_all[tr]

    y_tr = y_tr_clean.copy()
    if j['noise'] > 0:
        rngs = y_tr_clean.max(0) - y_tr_clean.min(0)
        y_tr = (y_tr_clean + rng.normal(size=y_tr.shape) * (j['noise'] * rngs)).astype(np.float32)

    E = j['epochs']; loss_curve = np.zeros(E, np.float32)
    t0 = time.perf_counter()

    if j['model'] == "pinn":
        net = mlp(1, j['depth'], j['width'], j['act'])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        tt = torch.tensor(t_tr.reshape(-1, 1), requires_grad=True)
        yy = torch.tensor(y_tr)
        # collocation points for the physics residual: training window (default)
        # or the whole window [0,300] ms ("full": physics only, no test data)
        tc = torch.tensor(t_all.reshape(-1, 1), requires_grad=True) if j.get('colloc') == "full" else None
        for ep in range(E):
            opt.zero_grad()
            pred = net(tt)
            if tc is None:
                loss = torch.mean((pred - yy) ** 2) + residual_loss(torch, tt, pred, p)
            else:
                loss = torch.mean((pred - yy) ** 2) + residual_loss(torch, tc, net(tc), p)
            loss.backward(); opt.step()
            loss_curve[ep] = loss.item()
        with torch.no_grad():
            pred_all = net(torch.tensor(t_all.reshape(-1, 1))).numpy()
    else:
        lo, hi = y_tr.min(0), y_tr.max(0)                  # Min-Max on (noisy) training data
        ysc = ((y_tr - lo) / (hi - lo)).astype(np.float32)
        tmin, tmax = t_tr.min(), t_tr.max()
        tsc = torch.tensor((t_tr - tmin) / (tmax - tmin), dtype=torch.float32)
        yy = torch.tensor(ysc)
        net = mlp(2, j['depth'], j['width'], j['act'])
        f = lambda t, y: net(y)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        y0 = torch.tensor(ysc[0])
        for ep in range(E):
            opt.zero_grad()
            pred = odeint(f, y0, tsc, method='dopri5', atol=j.get('atol', 1e-5), rtol=j.get('rtol', 1e-4))
            loss = torch.mean((pred - yy) ** 2)
            loss.backward(); opt.step()
            loss_curve[ep] = loss.item()
        with torch.no_grad():
            t_eval = torch.tensor((t_all - tmin) / (tmax - tmin), dtype=torch.float32)
            p_sc = odeint(f, y0, t_eval, method='dopri5').numpy()
        pred_all = p_sc * (hi - lo) + lo

    elapsed = time.perf_counter() - t0
    n_params = sum(q.numel() for q in net.parameters())
    finite = bool(np.all(np.isfinite(pred_all)))
    pa = np.nan_to_num(pred_all, nan=1e6, posinf=1e6, neginf=-1e6)

    res = dict(job_id=job_id(j), **j, params=n_params, train_time_s=elapsed,
               final_loss=float(loss_curve[-1]), finite=finite)
    if j['window'] == "full":
        res["full"] = metrics(y_all, pa)
    else:
        te = ~tr
        res["train"] = metrics(y_all[tr], pa[tr])
        res["test"] = metrics(y_all[te], pa[te])

    fj, fn, fc = paths(j)
    np.savez_compressed(fn, t=t_all, y_true=y_all, y_train_used=y_tr, train_mask=tr,
                        pred=pred_all.astype(np.float32), loss=loss_curve[::10])
    with open(fj + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(fj + ".tmp", fj)
    if os.path.exists(fc):
        os.remove(fc)
    return res


# --- Summary ---
def summarize():
    import pandas as pd
    rows = []
    for f in glob.glob(os.path.join(OUT, "*", "*.json")):
        r = json.load(open(f))
        base = {k: r[k] for k in ["exp", "scenario", "model", "act", "depth", "width", "seed", "noise", "epochs", "params", "train_time_s", "finite"]}
        for split in ["full", "train", "test"]:
            if split in r:
                rows.append(dict(base, split=split, **r[split]))
    if not rows:
        print("No results yet."); return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
    keys = ["exp", "scenario", "model", "act", "depth", "width", "noise", "epochs", "split"]
    cols = ["RMSE_V", "RMSE_N", "R2_V", "MaxErr_V", "Total_MSE", "train_time_s"]
    agg = df.groupby(keys)[cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(os.path.join(OUT, "summary_mean_std.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for e in agg.exp.unique():
        print(f"\n===== {e} =====")
        a = agg[agg.exp == e]
        show = keys[1:] + ["RMSE_V_mean", "RMSE_V_std", "R2_V_mean", "R2_V_std", "RMSE_V_count"]
        print(a[show].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nSaved: {os.path.join(OUT, 'all_runs.csv')} and summary_mean_std.csv")


# --- CLI ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["main", "act", "tol", "pinncol", "seeds", "noise", "grid", "gridhopf", "all"])
    ap.add_argument("--model", default="both", choices=["pinn", "node", "both"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--reset-claims", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=0, help="stop after N jobs (0 = no limit)")
    ap.add_argument("--epochs-override", type=int, default=0, help="debug only: force small epoch count")
    a = ap.parse_args()

    if a.summarize:
        summarize(); return
    if a.reset_claims:
        n = 0
        for f in glob.glob(os.path.join(OUT, "*", "*.claim")):
            os.remove(f); n += 1
        print(f"Removed {n} claim files."); return

    exps = ["main", "seeds", "noise", "grid", "act"] if a.exp == "all" else [a.exp]
    J = [j for e in exps for j in jobs_for(e) if a.model in ("both", j['model'])]
    if a.epochs_override:
        for j in J: j['epochs'] = a.epochs_override; j['exp'] = j['exp'] + "_debug"

    done = sum(os.path.exists(paths(j)[0]) for j in J)
    print(f"Jobs: {len(J)} total, {done} done, {len(J) - done} remaining.")
    if a.list:
        for e in exps:
            Je = [j for j in J if j['exp'].startswith(e)]
            de = sum(os.path.exists(paths(j)[0]) for j in Je)
            print(f"  {e:6s}: {de}/{len(Je)}")
        return

    n = 0
    for j in J:
        if not claim(j):
            continue
        tag = job_id(j)
        print(f"\n[{time.strftime('%H:%M:%S')}] START {tag}", flush=True)
        try:
            r = run(j, a.threads)
            m = r.get("test", r.get("full"))
            print(f"[{time.strftime('%H:%M:%S')}] DONE  {tag}  ({r['train_time_s']/60:.1f} min)  "
                  f"RMSE_V={m['RMSE_V']:.3f}  R2_V={m['R2_V']:.3f}", flush=True)
        except KeyboardInterrupt:
            os.remove(paths(j)[2]); print("Interrupted; claim released."); sys.exit(1)
        except Exception as ex:
            os.remove(paths(j)[2]); print(f"FAILED {tag}: {ex}", flush=True)
        n += 1
        if a.max_jobs and n >= a.max_jobs:
            break
    print("\nNo more free jobs for this command.")


if __name__ == "__main__":
    main()
