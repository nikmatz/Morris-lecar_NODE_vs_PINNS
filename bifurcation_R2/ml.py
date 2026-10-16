import numpy as np
from scipy.optimize import brentq
REG = {
 "Hopf": dict(C=20.0,g_Ca=4.4,g_K=8.0,g_L=2.0,V_Ca=120.0,V_K=-84.0,V_L=-60.0,phi=0.04,V1=-1.2,V2=18.0,V3=2.0,V4=30.0),
 "SNLC": dict(C=20.0,g_Ca=4.0,g_K=8.0,g_L=2.0,V_Ca=120.0,V_K=-84.0,V_L=-60.0,phi=0.067,V1=-1.2,V2=18.0,V3=12.0,V4=17.4),
 "Homoclinic": dict(C=20.0,g_Ca=4.0,g_K=8.0,g_L=2.0,V_Ca=120.0,V_K=-84.0,V_L=-60.0,phi=0.23,V1=-1.2,V2=18.0,V3=12.0,V4=17.4),
}
TRAIN = {"Hopf":90.0,"SNLC":42.0,"Homoclinic":35.5}
def minf(V,p): return 0.5*(1+np.tanh((V-p['V1'])/p['V2']))
def ninf(V,p): return 0.5*(1+np.tanh((V-p['V3'])/p['V4']))
def tau(V,p):  return 1/np.cosh((V-p['V3'])/(2*p['V4']))
def rhs(t,y,p,I):
    V,N=y
    return [(I-p['g_L']*(V-p['V_L'])-p['g_Ca']*minf(V,p)*(V-p['V_Ca'])-p['g_K']*N*(V-p['V_K']))/p['C'],
            p['phi']*(ninf(V,p)-N)/tau(V,p)]
def I_eq(V,p):  # current at which V is an equilibrium
    return p['g_Ca']*minf(V,p)*(V-p['V_Ca'])+p['g_K']*ninf(V,p)*(V-p['V_K'])+p['g_L']*(V-p['V_L'])
def jac(V,p,I):
    N=ninf(V,p); h=1e-6
    f=lambda y: np.array(rhs(0,y,p,I))
    J=np.zeros((2,2)); y=np.array([V,N])
    for k in range(2):
        e=np.zeros(2); e[k]=h; J[:,k]=(f(y+e)-f(y-e))/(2*h)
    return J
def eq_branch(p,Vgrid=np.linspace(-70,60,26001)):
    I=I_eq(Vgrid,p); tr=[];det=[]
    for V,Ii in zip(Vgrid,I):
        J=jac(V,p,Ii); tr.append(np.trace(J)); det.append(np.linalg.det(J))
    return Vgrid,I,np.array(tr),np.array(det)
def special_points(p):
    V,I,tr,det=eq_branch(p)
    dI=np.gradient(I,V)
    sn=[(I[k],V[k]) for k in range(len(V)-1) if np.sign(dI[k])!=np.sign(dI[k+1])]
    hb=[(I[k],V[k]) for k in range(len(V)-1) if np.sign(tr[k])!=np.sign(tr[k+1]) and det[k]>0]
    return sn,hb
if __name__=="__main__":
    for n,p in REG.items():
        sn,hb=special_points(p); print(n,"SN:",[(round(a,3),round(b,2)) for a,b in sn],"Hopf:",[(round(a,3),round(b,2)) for a,b in hb])
