import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eig
import os

def analyze_equilibrium(C, g_Ca, g_K, g_L, V_Ca, V_K, V_L,
                        phi, V1, V2, V3, V4, I, scenario_name):

    # === Create directory for saving plots and log ===
    os.makedirs("phaseplot", exist_ok=True)
    log_file = os.path.join("phaseplot", "equilibrium_summary.txt")

    # === Define ML functions ===
    def m_inf(V): return 0.5 * (1 + np.tanh((V - V1) / V2))
    def N_inf(V): return 0.5 * (1 + np.tanh((V - V3) / V4))
    def tau_N(V): return 1 / np.cosh((V - V3) / (2 * V4))

    def dm_dV(V): return 0.5 * (1 - np.tanh((V - V1) / V2) ** 2) / V2
    def dNinf_dV(V): return 0.5 * (1 - np.tanh((V - V3) / V4) ** 2) / V4
    def dtau_dV(V): return -np.sinh((V - V3)/(2 * V4)) / (2 * V4 * np.cosh((V - V3)/(2 * V4)) ** 2)

    # === Define steady-state system ===
    def steady_state(vars):
        V, N = vars
        eq1 = I - g_Ca * m_inf(V) * (V - V_Ca) - g_K * N * (V - V_K) - g_L * (V - V_L)
        eq2 = N - N_inf(V)
        return [eq1, eq2]

    # === Find equilibrium point ===
    V_star, N_star = fsolve(steady_state, [0.0, 0.0])

    # === Compute Jacobian ===
    a11 = (-g_Ca * (dm_dV(V_star) * (V_star - V_Ca) + m_inf(V_star)) 
           - g_K * N_star - g_K * (V_star - V_K) - g_L) / C
    a12 = -g_K * (V_star - V_K) / C
    a21 = phi * (dNinf_dV(V_star) * tau_N(V_star) - (N_inf(V_star) - N_star) * dtau_dV(V_star))
    a22 = -phi / tau_N(V_star)

    J = np.array([[a11, a12], [a21, a22]])
    eigvals = eig(J)[0]

    # === Determine type of equilibrium ===
    real_parts = eigvals.real
    imag_parts = eigvals.imag

    if np.any(real_parts > 0) and np.any(real_parts < 0):
        eq_type = "Saddle point"
    elif np.all(real_parts < 0):
        eq_type = "Stable node/focus"
    elif np.all(real_parts > 0):
        eq_type = "Unstable node/focus"
    elif np.allclose(real_parts, 0):
        eq_type = "Center (neutral)"
    else:
        eq_type = "Unknown"

    # === Print to screen and save to file ===
    output_text = f"""
    ==============================
    Scenario: {scenario_name}
    Equilibrium point: V* = {V_star:.4f}, N* = {N_star:.4f}
    Jacobian:
    {np.array2string(J, precision=4, floatmode='fixed')}
    Eigenvalues: {eigvals}
    Type: {eq_type}
    =============================="""

    print(output_text)
    with open(log_file, "a") as f:
        f.write(output_text + "\n")

    # === Generate phase portrait ===
    V_range = np.linspace(V_star - 120, V_star + 120, 500)
    N_range = np.linspace(N_star - 0.6, N_star + 0.6, 500)
    V, N = np.meshgrid(V_range, N_range)

    dVdt = (I - g_Ca * m_inf(V) * (V - V_Ca) - g_K * N * (V - V_K) - g_L * (V - V_L)) / C
    dNdt = phi * (N_inf(V) - N) / tau_N(V)

    fig, ax = plt.subplots(figsize=(8, 6))
    strm = ax.streamplot(V, N, dVdt, dNdt, color="black", density=2.0, maxlength=100, arrowsize=1)
    ax.plot(V_star, N_star, 'o', color='green', markersize=10)
    ax.text(V_star + 2, N_star, eq_type, fontsize=12, color='green')
    ax.set_title(f"Phase Portrait – {scenario_name}")
    ax.set_xlabel("Membrane Voltage V")
    ax.set_ylabel("Gating Variable N")
    ax.grid(True)

    # === Save figure ===
    png_path = f"phaseplot/equilibrium_{scenario_name}.png"
    eps_path = f"phaseplot/equilibrium_{scenario_name}.eps"
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.show()

# === Scenarios ===
scenarios = {
    "Homoclinic": dict(C=20.0, g_Ca=4.0, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.23, V1=-1.2, V2=18.0,
                       V3=12.0, V4=17.4, I=50.0),
    
    "SNLC": dict(C=20.0, g_Ca=4.0, g_K=8.0, g_L=2.0,
                 V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                 phi=0.067, V1=-1.2, V2=18.0,
                 V3=12.0, V4=17.4, I=42.0),
    
    "Hopf": dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
                 V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                 phi=0.04, V1=-1.2, V2=18.0,
                 V3=2.0, V4=30.0, I=90.0),
    
    "Hopf_50": dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
                    V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                    phi=0.04, V1=-1.2, V2=18.0,
                    V3=2.0, V4=30.0, I=50.0)
}

# === Run analysis for all ===
for name, params in scenarios.items():
    analyze_equilibrium(**params, scenario_name=name)
