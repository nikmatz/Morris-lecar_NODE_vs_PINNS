import numpy as np, json, sys
from scipy.integrate import solve_ivp
from ml import *
def settle(p,I,y0,T=4000,Trec=1500):
    s=solve_ivp(rhs,[0,T],y0,args=(p,I),method='LSODA',rtol=1e-8,atol=1e-10,max_step=0.5)
    t,V=s.t,s.y[0]; m=t>T-Trec
    amp=V[m].max()-V[m].min()
    # period via upward crossings of mid level
    per=np.nan
    if amp>1:
        mid=0.5*(V[m].max()+V[m].min()); tt=t[m];vv=V[m]
        c=[tt[k] for k in range(len(vv)-1) if vv[k]<mid<=vv[k+1]]
        if len(c)>2: per=np.mean(np.diff(c))
    return s.y[:,-1],V[m].max(),V[m].min(),amp,per
name=sys.argv[1]; p=REG[name]
Is=np.round(np.arange(0,120.01,0.5),3)
out={}
for direction,seq in [("up",Is),("down",Is[::-1])]:
    y=None; res=[]
    for I in seq:
        if y is None:
            y=[-60.0,0.0] if direction=="up" else [20.0,0.4]
        y,mx,mn,amp,per=settle(p,I,y)
        res.append([float(I),float(mx),float(mn),float(amp),float(per)])
        if direction=="down" and amp<1 and I<60: # kick to probe cycle at low I handled by continuation
            pass
    out[direction]=res
json.dump(out,open(f"{name}_cyc.json","w"))
print(name,"done")
