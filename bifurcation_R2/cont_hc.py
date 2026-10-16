import numpy as np, json
from scipy.integrate import solve_ivp
from ml import *
p=REG['Homoclinic']
def settle(I,y0,T=4000,Trec=1500):
    s=solve_ivp(rhs,[0,T],y0,args=(p,I),method='LSODA',rtol=1e-9,atol=1e-11,max_step=0.5)
    t=s.t; m=t>T-Trec; V=s.y[0][m]; amp=V.max()-V.min(); per=np.nan
    if amp>1:
        mid=0.5*(V.max()+V.min()); tt=t[m]; c=[tt[k] for k in range(len(V)-1) if V[k]<mid<=V[k+1]]
        if len(c)>2: per=float(np.mean(np.diff(c)))
    return s.y[:,-1],float(V.max()),float(V.min()),float(amp),per
rows=[]
y,_,_,_,_=settle(35.5,[10,0.3])
y0=y.copy()
for I in np.round(np.arange(35.5,40.701,0.05),3):
    y,mx,mn,amp,per=settle(I,y); rows.append([float(I),mx,mn,amp,per])
    if amp<1: break
y=y0.copy()
for I in np.round(np.arange(35.45,34.98,-0.01),3):
    y,mx,mn,amp,per=settle(I,y,T=5000,Trec=2500); rows.append([float(I),mx,mn,amp,per])
    if amp<1: break
json.dump({"up":rows,"down":[]},open("Homoclinic_cont.json","w"))
ok=[r for r in rows if r[3]>1]; print("stable large cycle I range: %.2f - %.2f"%(min(r[0] for r in ok),max(r[0] for r in ok)))
print([ (r[0],round(r[4],1)) for r in sorted(ok)[:4]], [(r[0],round(r[3],2)) for r in sorted(ok)[-3:]])
