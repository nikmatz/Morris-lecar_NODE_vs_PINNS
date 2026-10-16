import numpy as np, json, sys
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from ml import *
p=REG[sys.argv[4]]
a,b,h=map(float,sys.argv[1:4]); NAME=sys.argv[4]
Vg=np.linspace(-10,20,300001)
def upper_eq(I):
    g=I_eq(Vg,p)-I; k=np.where(np.diff(np.sign(g)))[0][-1]; return Vg[k]-g[k]*(Vg[k+1]-Vg[k])/(g[k+1]-g[k])
def ret(v0,I,Ve,Ne):
    ev=lambda t,y,*a: y[1]-Ne; ev.terminal=True; ev.direction=1
    s=solve_ivp(rhs,[0,3000],[v0,Ne+1e-9],args=(p,I),method='LSODA',rtol=1e-10,atol=1e-12,events=ev,max_step=1.0)
    if len(s.t_events[0])==0: return np.nan,None
    return s.y_events[0][0][0]-v0,s
res=[]
for I in np.arange(a,b+1e-9,h):
    Ve=upper_eq(I); Ne=float(ninf(Ve,p))
    grid=Ve+np.concatenate([[0.02,0.05,0.1,0.2,0.4,0.7],np.arange(1,30,0.5)])
    vals=[ret(v,I,Ve,Ne)[0] for v in grid]
    root=None
    for k in range(len(grid)-1):
        if np.isfinite(vals[k]) and np.isfinite(vals[k+1]) and vals[k]<0<vals[k+1]:
            root=brentq(lambda v: ret(v,I,Ve,Ne)[0],grid[k],grid[k+1],xtol=1e-7); break
    if root is None: print(round(I,2),'none',flush=True); continue
    _,s=ret(root,I,Ve,Ne)
    s2=solve_ivp(rhs,[0,s.t[-1]],[root,Ne+1e-9],args=(p,I),method='LSODA',rtol=1e-10,atol=1e-12,max_step=0.05)
    res.append([float(I),float(s2.y[0].max()),float(s2.y[0].min())]); print(round(I,2),[round(x,2) for x in res[-1][1:]],'period %.1f'%s.t[-1],flush=True)
json.dump(res,open(f"{NAME}_unst_{a}_{b}.json","w"))
