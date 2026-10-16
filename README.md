# Code and results — Comparing PINNs and Neural ODEs on the Morris–Lecar model

Which script produces which part of the paper (Revision 2).

| Paper item | Script | Output folder |
|---|---|---|
| Synthetic data (Hopf I=90, SNLC I=42) | `synthetic_M_L_data.py` | `scenario_data/` |
| Main study, Hopf & SNLC (Tables 5, A1, A2; Figs 5, B1–B4) | `Pinns_all_final.py`, `NODE_all_final.py` | `metrics/`, `models_*/`, `plots_*/` |
| Hopf parameters at I=50, below the onset of oscillations (Fig 3) | `single_run/Pinns_50_final.py`, `single_run/NODE_50_final.py` (training), `single_run/plot_fig3.py` (figure) | `single_run/` |
| Homoclinic regime at I=35.5 (Tables 5, A1, A2; Figs 4, 5, B5, B6) | `r2_experiments.py --exp main` | `results_R2/main/` |
| Multi-seed extrapolation (Table 6, Fig 6) | `r2_experiments.py --exp seeds` and `--exp pinncol` | `results_R2/seeds/`, `results_R2/pinncol/` |
| Gaussian noise (Table 7) | `r2_experiments.py --exp noise` | `results_R2/noise/` |
| Factorial depth × width (Table 8, Fig 7) | `r2_experiments.py --exp gridhopf` and `--exp grid` | `results_R2/gridhopf/`, `results_R2/grid/` |
| NODE activation and solver-tolerance checks (Fig 4, Section 4.5) | `r2_experiments.py --exp act` and `--exp tol` | `results_R2/act/`, `results_R2/tol/` |
| All new tables and figures | `make_r2_outputs.py` | `r2_figures/` |
| Bifurcation diagrams (Fig 1), phase portraits and equilibria (Fig 2, Table 3) | `bifurcation_R2/` (`cycles.py`, `fine.py`, `cont_hc.py`, `unstable.py`, `unst_gen.py`, `unst_hc.py`, `unst_fold.py`, `plot_bif.py`, `equilibria_R2.py`) | `bifurcation_R2/` |
| Time-rescaling check (Remark 2) | `time_scaling_R2/time_scaling_check.py` | printed output |
| Verification of data, physics residual, NODE pipeline, metrics | `audit_R2/` | printed output |

How to run the experiments: see `README_R2_experiments.md`.

Note: the homoclinic scenario inside `Pinns_all_final.py`, `NODE_all_final.py` and `synthetic_M_L_data.py` still uses the first-version current I = 50. These scripts are kept to reproduce the Hopf and SNLC results. All homoclinic results in the paper come from `r2_experiments.py` (I = 35.5, y0 = [10, 0.3]).

`_superseded_R1/` contains scripts and results from the first revision that are no longer used in the paper: the old bifurcation and equilibrium scripts, which contained errors corrected in Revision 2; the single-seed extrapolation and three-architecture studies; and the old summary plots. They are kept for transparency.
