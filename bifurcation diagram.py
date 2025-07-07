import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eig
import os
import pandas as pd

# === SCENARIOS ===
scenarios = {
    "Hopf": dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
                 V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                 phi=0.04, V1=-1.2, V2=18.0, V3=2.0, V4=30.0),
    
    "SNLC": dict(C=20.0, g_Ca=4.0, g_K=8.0, g_L=2.0,
                 V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                 phi=0.067, V1=-1.2, V2=18.0, V3=12.0, V4=17.4),
    
    "Homoclinic": dict(C=20.0, g_Ca=4.0, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.23, V1=-1.2, V2=18.0, V3=12.0, V4=17.4)
}

I_values = np.linspace(30, 100, 200)
os.makedirs("bifurcation_plots", exist_ok=True)
os.makedirs("bifurcation_data", exist_ok=True)

for name, p in scenarios.items():
    λ1, λ2 = [], []
    bifurcation_I = None

    for I in I_values:
        def m_inf(V): return 0.5 * (1 + np.tanh((V - p['V1']) / p['V2']))
        def N_inf(V): return 0.5 * (1 + np.tanh((V - p['V3']) / p['V4']))
        def tau_N(V): return 1 / np.cosh((V - p['V3']) / (2 * p['V4']))
        def dm_dV(V): return 0.5 * (1 - np.tanh((V - p['V1']) / p['V2'])**2) / p['V2']
        def dNinf_dV(V): return 0.5 * (1 - np.tanh((V - p['V3']) / p['V4'])**2) / p['V4']
        def dtau_dV(V): return -np.sinh((V - p['V3']) / (2 * p['V4'])) / (2 * p['V4'] * np.cosh((V - p['V3']) / (2 * p['V4']))**2)

        def steady_state(vars):
            V, N = vars
            eq1 = I - p['g_Ca'] * m_inf(V) * (V - p['V_Ca']) - p['g_K'] * N * (V - p['V_K']) - p['g_L'] * (V - p['V_L'])
            eq2 = N - N_inf(V)
            return [eq1, eq2]

        try:
            V_star, N_star = fsolve(steady_state, [0.0, 0.0])
            a11 = (-p['g_Ca'] * (dm_dV(V_star) * (V_star - p['V_Ca']) + m_inf(V_star)) 
                   - p['g_K'] * N_star - p['g_K'] * (V_star - p['V_K']) - p['g_L']) / p['C']
            a12 = -p['g_K'] * (V_star - p['V_K']) / p['C']
            a21 = p['phi'] * (dNinf_dV(V_star) * tau_N(V_star) - (N_inf(V_star) - N_star) * dtau_dV(V_star))
            a22 = -p['phi'] / tau_N(V_star)
            J = np.array([[a11, a12], [a21, a22]])
            eigvals = eig(J)[0]
            r1, r2 = np.real(eigvals[0]), np.real(eigvals[1])
            λ1.append(r1)
            λ2.append(r2)

            # Check for bifurcation point
            if bifurcation_I is None and (r1 > 0 or r2 > 0):
                bifurcation_I = I
        except:
            λ1.append(np.nan)
            λ2.append(np.nan)

    # === SAVE CSV ===
    df = pd.DataFrame({'I': I_values, 'Re_lambda1': λ1, 'Re_lambda2': λ2})
    df.to_csv(f"bifurcation_data/{name}_eigenvalues.csv", index=False)

    # === PLOT 2D ===
    plt.figure(figsize=(8, 5))
    plt.plot(I_values, λ1, label="Re(λ₁)")
    plt.plot(I_values, λ2, label="Re(λ₂)")
    plt.axhline(0, color='gray', linestyle='--')
    if bifurcation_I:
        plt.axvline(bifurcation_I, color='red', linestyle=':', label=f"Bifurcation at I = {bifurcation_I:.2f}")
    plt.xlabel("Input Current I")
    plt.ylabel("Real Part of Eigenvalues")
    plt.title(f"Bifurcation Diagram – {name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"bifurcation_plots/bifurcation_2D_{name}.png", dpi=300)
    plt.savefig(f"bifurcation_plots/bifurcation_2D_{name}.eps", format='eps')
    plt.close()
