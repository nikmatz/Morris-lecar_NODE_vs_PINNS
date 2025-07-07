import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# ==== Morris-Lecar model functions ====
def N_inf(V, V3, V4):
    return 0.5 * (1 + np.tanh((V - V3) / V4))

def tau_N(V, V3, V4):
    arg = (V - V3) / (2 * V4)
    arg = np.clip(arg, -100, 100)  # prevent overflow
    return 1 / np.cosh(arg)

def I_ext_func(I):
    return lambda t: I

def morris_lecar(t, y, p):
    V, N = y
    I_ext = p['I_ext_func'](t)
    m_inf = 0.5 * (1 + np.tanh((V - p['V1']) / p['V2']))
    dVdt = (I_ext - p['g_L'] * (V - p['V_L']) - p['g_Ca'] * m_inf * (V - p['V_Ca']) - p['g_K'] * N * (V - p['V_K'])) / p['C']
    dNdt = p['phi'] * (N_inf(V, p['V3'], p['V4']) - N) / tau_N(V, p['V3'], p['V4'])
    return [dVdt, dNdt]

# ==== Simulation parameters ====
def run_simulation(I_val, params):
    params = params.copy()
    params['I_ext_func'] = I_ext_func(I_val)
    sol = solve_ivp(lambda t, y: morris_lecar(t, y, params), [0, 500], [0, 0], method='RK45', t_eval=np.linspace(0, 500, 5000))
    return sol.t, sol.y[0]

# ==== Compute oscillation amplitude ====
def compute_amplitude(V, t, threshold=0.5):
    V = V[int(len(V) * 0.5):]  # Discard transient
    max_V = np.max(V)
    min_V = np.min(V)
    amplitude = max_V - min_V
    return amplitude if amplitude > threshold else 0

# ==== Main bifurcation computation ====
def compute_bifurcation(name, param_dict, I_range, threshold=0.5):
    print(f"\n▶ Processing {name.upper()}...")
    amplitudes = []
    I_vals = np.linspace(*I_range, 300)

    for I in I_vals:
        _, V = run_simulation(I, param_dict)
        A = compute_amplitude(V, _, threshold)
        amplitudes.append(A)

    amplitudes = np.array(amplitudes)

    # === Detect bifurcation ===
    bif_indices = np.where(amplitudes > threshold)[0]
    if bif_indices.size > 0:
        I_bif_start, I_bif_end = I_vals[bif_indices[0]], I_vals[bif_indices[-1]]
        print(f"\U0001f9e0 Bifurcation interval detected in {name.upper()}: I ∈ [{I_bif_start:.1f}, {I_bif_end:.1f}] µA/cm²")
    else:
        I_bif_start = I_bif_end = None
        print(f"⚠️ No bifurcation detected in {name.upper()}.")

    # === Plot ===
    plt.figure(figsize=(9, 5))
    plt.plot(I_vals, amplitudes, color='blue', label="Amplitude")

    if bif_indices.size > 0:
        plt.axvspan(I_bif_start, I_bif_end, color='khaki', alpha=0.4,
                    label=f"Bifurcation region: {I_bif_start:.1f}–{I_bif_end:.1f}")
        plt.axvline(I_bif_start, color='red', linestyle='--', linewidth=1)
        plt.axvline(I_bif_end, color='red', linestyle='--', linewidth=1)

    plt.xlabel("External Current $I$ ($\mu A/cm^2$)")
    plt.ylabel("Oscillation Amplitude of $V$ (mV)")
    plt.title(f"{name.upper()} Bifurcation Diagram")
    plt.xlim(0, 120)  # Expanded range to include all Kyoto systems
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("bifurcation_plots", exist_ok=True)
    plt.savefig(f"bifurcation_plots/{name}_bifurcation.png", dpi=600)
    plt.savefig(f"bifurcation_plots/{name}_bifurcation.eps", format='eps')
    plt.close()
    print(f"✅ Saved .png and .eps for {name.upper()}")

# ==== Define scenarios ====
scenarios = {
    'hopf': {
        'params': dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       V1=-1.2, V2=18.0, V3=2.0, V4=30.0,
                       phi=0.04),
        'I_range': (60, 100),
        'threshold': 0.05
    },
    'snlc': {
        'params': dict(C=20.0, g_Ca=4.0, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       V1=-1.2, V2=18.0, V3=2.0, V4=30.0,
                       phi=0.04),
        'I_range': (30, 70),
        'threshold': 0.05
    },
    'homoclinic': {
        'params': dict(C=20.0, g_Ca=4.4, g_K=8.0, g_L=2.0,
                       V_Ca=120.0, V_K=-84.0, V_L=-60.0,
                       V1=-1.2, V2=18.0, V3=2.0, V4=30.0,
                       phi=0.04),
        'I_range': (60, 100),
        'threshold': 0.05
    }
}

# ==== Run for all scenarios ====
for name, cfg in scenarios.items():
    compute_bifurcation(name, cfg['params'], cfg['I_range'], cfg['threshold'])

print("\n🏁 All bifurcation diagrams generated.")