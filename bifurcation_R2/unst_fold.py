"""
unst_fold.py — unstable limit cycles of the Hopf regime close to the fold of limit cycles (LPC).
Near the fold the branch is almost vertical in I, so instead of fixing I and solving for the
fixed point of the return map, the section coordinate v0 is fixed and the current I(v0) is
solved for. Fixed section: N = N*, V > V* (equilibrium at I = 88.5), upward crossings.
Output: Hopf_unst_fold.json and Hopf_stab_fold.json ([I, Vmax, Vmin] of the unstable and stable
cycles near the fold), printed LPC estimate.
"""
import numpy as np, json
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from ml import *

p = REG["Hopf"]
V, Ieq, tr, det = eq_branch(p)
I0 = 88.5
Vc = float(V[np.argmin(np.abs(Ieq - I0))]); Nc = float(ninf(Vc, p))

def G(v0, I, dense=False):
    ev = lambda t, y: y[1] - Nc
    ev.terminal = True; ev.direction = 1
    s = solve_ivp(lambda t, y: rhs(t, y, p, I), [0, 3000], [v0, Nc + 1e-9], method='LSODA',
                  rtol=1e-10, atol=1e-12, events=ev, max_step=0.05 if dense else 1.0)
    if len(s.t_events[0]) == 0:
        return np.nan, s
    return s.y_events[0][0][0] - v0, s

# fixed points of the return map at I = Ib (unstable inner cycle, stable outer cycle)
Ia, Ib = 88.0, 89.25
grid = Vc + np.arange(0.2, 15.0, 0.05)   # the two fixed points lie within < 1 mV of each other
g = np.array([G(v, Ib)[0] for v in grid])
roots = [brentq(lambda v: G(v, Ib)[0], grid[k], grid[k + 1], xtol=1e-8)
         for k in range(len(grid) - 1) if np.isfinite(g[k]) and np.isfinite(g[k + 1]) and g[k] * g[k + 1] < 0]
vu, vs = roots[0], roots[1]
print(f"I = {Ib}: unstable fixed point v0 = {vu:.4f}, stable fixed point v0 = {vs:.4f}")
print(f"check G(v0, {Ia}) < 0:", G(0.5 * (vu + vs), Ia)[0])

def solve(v0):
    Is = brentq(lambda I: G(v0, I)[0], Ia, Ib, xtol=1e-12)
    h = 1e-6
    slope = (G(v0 + h, Is)[0] - G(v0 - h, Is)[0]) / (2 * h)   # > 0: unstable, < 0: stable
    _, s = G(v0, Is, dense=True)
    return [float(v0), float(Is), float(s.y[0].max()), float(s.y[0].min()), float(slope)]

pts = [solve(v0) for v0 in np.linspace(vu, vs, 62)[1:-1]]
# adaptive refinement where the branch is steep (canard-like growth of the amplitude near the fold)
for _ in range(12):
    pts.sort(key=lambda r: r[0]); new = []
    for a, b in zip(pts[:-1], pts[1:]):
        if (abs(a[2] - b[2]) > 1.5 or abs(a[3] - b[3]) > 1.5) and b[0] - a[0] > 1e-11:
            new.append(solve(0.5 * (a[0] + b[0])))
    if not new:
        break
    pts += new
pts.sort(key=lambda r: r[0])
for r in pts:
    print(f"v0 = {r[0]:.10f}  I = {r[1]:.5f}  Vmax = {r[2]:7.2f}  Vmin = {r[3]:7.2f}  {'unstable' if r[4] > 0 else 'stable'}")
print(f"LPC estimate: I = {min(r[1] for r in pts):.4f}")
json.dump([r[1:4] for r in pts if r[4] > 0], open("Hopf_unst_fold.json", "w"))
json.dump([r[1:4] for r in pts if r[4] < 0], open("Hopf_stab_fold.json", "w"))
