import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import os

# === Setup directories ===
plot_dir = './scenario_plots'
data_dir = './scenario_data'
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

def save_figure(fig, name_base):
    fig.savefig(f'{plot_dir}/{name_base}_no.eps', format='eps', bbox_inches='tight', dpi=600)
    plt.close(fig)

# === Morris–Lecar model ===
def morris_lecar(t, y, params):
    V, N = y
    C, g_Ca, g_K, g_L = params['C'], params['g_Ca'], params['g_K'], params['g_L']
    V_Ca, V_K, V_L = params['V_Ca'], params['V_K'], params['V_L']
    phi, V1, V2, V3, V4 = params['phi'], params['V1'], params['V2'], params['V3'], params['V4']
    I = params['I']
    M_inf = 0.5 * (1 + np.tanh((V - V1) / V2))
    N_inf = 0.5 * (1 + np.tanh((V - V3) / V4))
    tau_N = 1.0 / np.cosh((V - V3) / (2 * V4))
    dVdt = (-g_Ca * M_inf * (V - V_Ca) - g_K * N * (V - V_K) - g_L * (V - V_L) + I) / C
    dNdt = phi * (N_inf - N) / tau_N
    return [dVdt, dNdt]

# === Parameter scenarios ===
scenarios = {
    'hopf_50': {
        'params': dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.04, V1=-1.2, V2=18.0, V3=2.0, V4=30.0, I=50.0),
        'y0': [0.0, 0.0],
        'I_range': np.linspace(0, 100, 100),
        'reference': ''
    },
    'hopf': {
        'params': dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.04, V1=-1.2, V2=18.0, V3=2.0, V4=30.0, I=90.0),
        'y0': [0.0, 0.0],
        'I_range': np.linspace(0, 100, 100),
        'reference': ''
    },
    'snlc': {
        'params': dict(C=20.0, g_Ca=4.0, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.067, V1=-1.2, V2=18.0, V3=12.0, V4=17.4, I=42),
        'y0': [0.0, 0.0],
        'I_range': np.linspace(0, 100, 100),
        'reference': ''
    },
    'homoclinic': {
        'params': dict(C=20.0, g_Ca=4.0, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       phi=0.23, V1=-1.2, V2=18.0, V3=12.0, V4=17.4, I=50),
        'y0': [0.0, 0.0],
        'I_range': np.linspace(0, 100, 100),
        'reference': ''
    }
}

# === Time span ===
t_eval = np.linspace(0, 300, 3000)

# === Main loop ===
for name, config in scenarios.items():
    p = config['params']
    I_used = p['I']
    ref = config['reference']

    # === Numerical solution ===
    sol = solve_ivp(morris_lecar, (0, 600), config['y0'], args=(p,), t_eval=t_eval, rtol=1e-6, atol=1e-8)
    t, V, N = sol.t, sol.y[0], sol.y[1]

    # === Save data to CSV
    df = pd.DataFrame({'t': t, 'V': V, 'N': N})
    csv_path = os.path.join(data_dir, f'{name}_ground_truth.csv')
    df.to_csv(csv_path, index=False)

    # === Bifurcation data
    I_vals = config['I_range']
    amplitudes = []
    for I in I_vals:
        p_temp = p.copy()
        p_temp['I'] = I
        sol_i = solve_ivp(morris_lecar, (0, 600), config['y0'], args=(p_temp,), t_eval=t_eval, rtol=1e-6, atol=1e-8)
        V_i = sol_i.y[0]
        V_ss = V_i[int(len(V_i)*0.66):]
        amplitudes.append(np.max(V_ss) - np.min(V_ss))

    # === Plotting V(t), N(t)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{name.upper()} Summary (I = {I_used})", fontsize=14)

    axs[0].plot(t, V, color='blue')
    axs[0].set_title(f'V(t) with I = {I_used}')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('V')
    axs[0].grid()

    axs[1].plot(t, N, color='orange')
    axs[1].set_title(f'N(t) with I = {I_used}')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('N')
    axs[1].grid()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, f'{name}_summary')

    # === Plotting bifurcation diagram
    fig_bif, ax_bif = plt.subplots(figsize=(7, 5))
    ax_bif.plot(I_vals, amplitudes, 'b-', linewidth=1.5)
    ax_bif.set_title(f'{name.upper()} Bifurcation Diagram')
    ax_bif.set_xlabel('I (External Current)')
    ax_bif.set_ylabel('Amplitude of V')
    ax_bif.axhline(0, color='gray', linestyle='--')
    ax_bif.grid()

    plt.tight_layout()
    save_figure(fig_bif, f'{name}_bifurcation')

print(f"\n✅ Όλα τα plots αποθηκεύτηκαν στο: {plot_dir}")
print(f"✅ Τα datasets αποθηκεύτηκαν στο: {data_dir}")
