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

Is=np.round(np.arange(0,120.01,0.5),3)
