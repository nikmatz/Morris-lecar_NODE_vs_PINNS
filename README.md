# Morris–Lecar: PINNs vs Neural ODEs

This repository contains the full implementation of the experiments presented in the paper:

**"Comparing Physics-Informed and Neural ODE Approaches for Modeling Nonlinear Biological Systems: A Case Study Based on the Morris–Lecar Model"**  
*Nikolaos Matzakos & Valantis Sfyrakis, 2025*

##  Overview

We systematically compare two modern deep learning frameworks for modeling dynamical systems:

- **Physics-Informed Neural Networks (PINNs)** – which embed known differential equations into the training loss using automatic differentiation.
- **Neural Ordinary Differential Equations (NODEs)** – which learn system dynamics directly from time-series data using ODE solvers.

All experiments are based on the **Morris–Lecar model**, a nonlinear 2D neuronal system known for its rich bifurcation dynamics.

##  Key Features

- Full PyTorch implementations of PINN and NODE models
- Synthetic data generation for 3 canonical bifurcation regimes:
  - Hopf
  - Saddle-Node on Limit Cycle (SNLC)
  - Homoclinic Orbit
- Evaluation with multiple metrics: MSE, RMSE, MAE, R², MAPE, RMSPE, Max Error
- Automatic phase portrait plotting and loss curve tracking
- FLOP and parameter count logging (via THOP)
- Deterministic training setup for full reproducibility

##  Experimental Setup

| Feature                | PINN                              | NODE                             |
|------------------------|------------------------------------|----------------------------------|
| Input                  | Scalar time `t`                    | Initial state `[V₀, N₀]`         |
| Output                 | `[V(t), N(t)]`                     | Solution from `odeint()`         |
| Architecture           | FCNN (3×128 Tanh)                 | FCNN (3×128 Tanh)               |
| Physics Integration    | Residuals via autograd             | None (pure data-driven)         |
| ODE Solver             | N/A                                | `odeint` (RK4)                  |
| Normalization          | None (raw physical units)          | Min–Max with inverse scaling    |
| Training Epochs        | 1000 to 20000                      | 1000 to 20000                   |
| Loss Function          | Data + Physics Residual Loss       | MSE on time-series              |
| Optimizer              | Adam (`lr=1e-3`)                   | Adam (`lr=1e-3`)                |
| Evaluation Metrics     | 7 metrics on `[V, N]`              | 7 metrics on `[V, N]`           |
| Reproducibility        | Deterministic + fixed seed         | Deterministic + fixed seed      |

## Directory Structure

```
.
├── pinn/
│   ├── pinn_model.py
│   ├── train_pinn.py
│   └── results/
├── node/
│   ├── node_model.py
│   ├── train_node.py
│   └── results/
├── data/
│   ├── synthetic_generator.py
│   ├── hopf.csv
│   ├── snlc.csv
│   └── homoclinic.csv
├── plots/
│   └── *.eps
├── metrics/
│   └── *.csv
├── models/
│   └── *.pth
├── README.md
└── requirements.txt
```

##  Installation

```bash
git clone https://github.com/nikmatz/Morris-lecar_NODE_vs_PINNS.git
cd Morris-lecar_NODE_vs_PINNS
pip install -r requirements.txt
```

## Run Experiments

Generate synthetic data:
```bash
python data/synthetic_generator.py
```

Train PINN:
```bash
python pinn/train_pinn.py
```

Train NODE:
```bash
python node/train_node.py
```

##  Results Summary

| Regime      | Best Method | RMSE (V) | RMSE (N) | Time (s) |
|-------------|-------------|----------|----------|----------|
| Hopf        | PINN        | 0.109    | 0.003    | ~15      |
| SNLC        | PINN        | 1.080    | 0.018    | ~15      |
| Homoclinic  | PINN        | 1.339    | 0.019    | ~15      |

See full metrics in `/metrics/*.csv`.

##  Citation

If you use this work in your research, please cite:

```
@article{matzakos2025morris,
  title={Comparing Physics-Informed and Neural ODE Approaches for Modeling Nonlinear Biological Systems: A Case Study Based on the Morris--Lecar Model},
  author={Matzakos, Nikolaos and Sfyrakis, Valantis},
  journal={Submitted},
  year={2025},
  note={Available at: https://github.com/nikmatz/Morris-lecar_NODE_vs_PINNS}
}
```

## Contact

- Nikolaos Matzakos — [nikmatz@aspete.gr](mailto:nikmatz@aspete.gr)


---

This project was developed using open-source tools and is intended for educational and research purposes in the fields of computational neuroscience and scientific machine learning.
