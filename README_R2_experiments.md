# Revision 2 experiments: how to run

All new experiments of Revision 2 are run with `r2_experiments.py` (from inside `Code_final/`). Each run writes one JSON file (configuration and metrics) and one NPZ file (predictions and loss curve) to `results_R2/<exp>/`.

## Experiments

| `--exp` | Description | Runs |
|---|---|---|
| `main` | Homoclinic regime at I = 35.5 (y0 = [10, 0.3]), main protocol (training on [0, 300] ms, 1,000–20,000 epochs): PINN, NODE-Tanh, NODE-SiLU | 12 |
| `seeds` | 3 regimes × {PINN, NODE} × 5 seeds; training on [0, 200] ms, extrapolation to (200, 300] ms; 10,000 epochs | 30 |
| `pinncol` | PINN† (data loss on [0, 200] ms, physics residual on [0, 300] ms), 3 regimes × 5 seeds, 10,000 epochs | 15 |
| `noise` | as `seeds`, with Gaussian noise of 1% and 5% of the range of each variable | 60 |
| `grid` | Homoclinic regime, depth {2, 3, 4} × width {16, 32, 64, 128, 256} × 3 seeds × {PINN, NODE}, 5,000 epochs | 90 |
| `gridhopf` | as `grid`, Hopf regime | 90 |
| `act` | NODE, Homoclinic regime, Tanh vs. SiLU, 5 seeds, training on [0, 300] ms, 10,000 epochs | 10 |
| `tol` | NODE, Homoclinic regime, Tanh, tightened solver tolerances (rtol = 1e-6, atol = 1e-7), 3 seeds, 10,000 epochs | 3 |

Total: 310 runs.

## Usage

```
python r2_experiments.py --exp seeds              # one experiment
python r2_experiments.py --exp seeds --model pinn # only the PINN runs of an experiment
python r2_experiments.py --list --exp all         # progress
python r2_experiments.py --reset-claims           # after an interrupted run
python r2_experiments.py --summarize              # results_R2/all_runs.csv and summary_mean_std.csv
python make_r2_outputs.py                         # all new tables and figures (r2_figures/)
```

Finished runs are skipped, so a command can be stopped and restarted at any time. Several copies of the same command can run in parallel (each copy claims the next free run with a lock file). All runs use fixed seeds and deterministic algorithms; repeating a run with the same seed reproduces identical predictions.

Requirements: torch, torchdiffeq, scikit-learn, scipy, pandas, numpy, matplotlib.
