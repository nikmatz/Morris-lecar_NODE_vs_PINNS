"""R2 (Reviewer 1, comment 2): linear time rescaling does not change stiffness.
Compares solver work (NFE) on t in [0,300] ms vs tau=t/300 in [0,1], and the
component-magnitude ratio max|dV/dt| / max|dN/dt| before and after Min-Max scaling."""
import numpy as np
from scipy.integrate import solve_ivp
REG = {
 "Hopf": dict(C=20.0,g_Ca=4.4,g_K=8.0,g_L=2.0,V_Ca=120.0,V_K=-84.0,V_L=-60.0,phi=0.04,V1=-1.2,V2=18.0,V3=2.0,V4=30.0,I=90.0),
 "SNLC": dict(C=20.0,g_Ca=4.0,g_K=8.0,g_L=2.0,V_Ca=120.0,V_K=-84.0,V_L=-60.0,phi=0.067,V1=-1.2,V2=18.0,V3=12.0,V4=17.4,I=42.0),
 "Homoclinic": dict(C=20.0,g_Ca=4.0,g_K=8.0,g_L=2.0,V_Ca=120.0,V_K=-84.0,V_L=-60.0,phi=0.23,V1=-1.2,V2=18.0,V3=12.0,V4=17.4,I=35.5)}
def rhs(t,y,p):
    V,N=y; m=0.5*(1+np.tanh((V-p['V1'])/p['V2'])); n=0.5*(1+np.tanh((V-p['V3'])/p['V4'])); tau=1/np.cosh((V-p['V3'])/(2*p['V4']))
    return [(p['I']-p['g_L']*(V-p['V_L'])-p['g_Ca']*m*(V-p['V_Ca'])-p['g_K']*N*(V-p['V_K']))/p['C'], p['phi']*(n-N)/tau]
T=300.0
Y0={'Hopf':[0,0],'SNLC':[0,0],'Homoclinic':[10,0.3]}
for name,p in REG.items():
    a=solve_ivp(rhs,[0,T],Y0[name],args=(p,),method='RK45',rtol=1e-6,atol=1e-9,dense_output=True)
    b=solve_ivp(lambda s,y: [T*v for v in rhs(s,y,p)],[0,1],Y0[name],method='RK45',rtol=1e-6,atol=1e-9)
    t=np.linspace(0,T,3000); Y=a.sol(t); dY=np.array([rhs(0,y,p) for y in Y.T])
    rV,rN=np.ptp(Y[0]),np.ptp(Y[1])
    raw=np.abs(dY[:,0]).max()/np.abs(dY[:,1]).max(); sc=(np.abs(dY[:,0]).max()*T/rV)/(np.abs(dY[:,1]).max()*T/rN)
    print(f"{name:11s} NFE t:{a.nfev:5d}  NFE tau:{b.nfev:5d} | ratio raw {raw:6.1f}  scaled {sc:4.1f}")
