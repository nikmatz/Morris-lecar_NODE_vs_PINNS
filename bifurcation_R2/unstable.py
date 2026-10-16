import numpy as np, json, sys
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from ml import *
name=sys.argv[1]; a,b,h=map(float,sys.argv[2:5]); upper=sys.argv[5]=="upper"; p=REG[name]
V,Ieq,tr,det=eq_branch(p)
def eqpt(I):
    idx=np.where((np.abs(Ieq-I)<0.02)&(tr<0)&(det>0))[0]
    if len(idx)==0: return None
    return V[idx[-1]] if upper else V[idx[0]]
def P(V0,I,Ne,Ve):
    # section: N=Ne, V>Ve, crossing with dN/dt>0 direction chosen by event
    ev=lambda t,y,*a: y[1]-Ne
    ev.direction=-1 # N decreasing through Ne on the V>Ve side? determine below
    s=solve_ivp(rhs,[0,5000],[V0,Ne],args=(p,I),method='LSODA',rtol=1e-10,atol=1e-12,events=ev,max_step=1.0)
    for t,y in zip(s.t_events[0],s.y_events[0]):
        if t>1e-3 and y[0]>Ve: return y[0],s
    return np.nan,s
res=[]
for I in np.arange(a,b+1e-9,h):
    Ve=eqpt(I)
    if Ve is None: continue
    Ne=ninf(Ve,p)
    # on V>Ve with N=Ne: dN/dt>0 (V above eq raises Ninf) -> first crossing back is with N decreasing on V<Ve side; full loop returns with N increasing on V>Ve
    grid=Ve+np.concatenate([[0.05,0.2,0.5],np.arange(1,80,1.5)])
    vals=[]
    for v0 in grid:
        def g(v0): 
            ev=lambda t,y: y[1]-Ne
            s=solve_ivp(lambda t,y: rhs(t,y,p,I),[0,5000],[v0,Ne+1e-9],method='LSODA',rtol=1e-10,atol=1e-12,events=ev,max_step=1.0)
            ys=[y for t,y in zip(s.t_events[0],s.y_events[0]) if y[0]>Ve and t>1]
            return (ys[0][0]-v0) if ys else np.nan
        vals.append(g(v0))
    vals=np.array(vals)
    root=None
    for k in range(len(grid)-1):
        if np.isfinite(vals[k]) and np.isfinite(vals[k+1]) and vals[k]<0<vals[k+1]:
            def g(v0):
                ev=lambda t,y: y[1]-Ne
                s=solve_ivp(lambda t,y: rhs(t,y,p,I),[0,5000],[v0,Ne+1e-9],method='LSODA',rtol=1e-10,atol=1e-12,events=ev,max_step=1.0)
                ys=[y for t,y in zip(s.t_events[0],s.y_events[0]) if y[0]>Ve and t>1]
                return ys[0][0]-v0
            root=brentq(g,grid[k],grid[k+1],xtol=1e-6); break
    if root is None: print(round(I,2),'none'); continue
    s=solve_ivp(lambda t,y: rhs(t,y,p,I),[0,400],[root,Ne+1e-9],method='LSODA',rtol=1e-10,atol=1e-12,max_step=0.2)
    # take one period
    res.append([float(I),float(s.y[0][:len(s.t)//3].max()),float(s.y[0][:len(s.t)//3].min())])
    print(round(I,2),[round(x,2) for x in res[-1][1:]])
json.dump(res,open(f"{name}_unst_{a}_{b}.json","w"))
