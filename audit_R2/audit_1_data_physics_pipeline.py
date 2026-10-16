import os
import numpy as np, torch, sys, os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
import r2_experiments as R
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results_R2","data")+"/"
import pandas as pd
print("A. Ground truth accuracy vs rtol=1e-11 reference")
for n,sc in R.SCENARIOS.items():
    d=pd.read_csv(D+n+"_ground_truth.csv"); t=d.t.values
    ref=solve_ivp(R.ml_rhs,(0,300),sc['y0'],args=(sc['params'],),t_eval=t,method='DOP853',rtol=1e-11,atol=1e-12)
    eV=np.abs(ref.y[0]-d.V.values).max(); eN=np.abs(ref.y[1]-d.N.values).max()
    print(f"  {n:11s} max|dV|={eV:.2e} mV  max|dN|={eN:.2e}   V-range={np.ptp(d.V):.1f}")
print("B. PINN residual consistent with the model: residual of the exact solution")
for n,sc in R.SCENARIOS.items():
    p=sc['params']; d=pd.read_csv(D+n+"_ground_truth.csv")
    t=torch.tensor(d.t.values,dtype=torch.float64)
    # exact solution as differentiable function: use dense ODE solution via torch autograd? use analytic: compare residual formula vs ml_rhs at data points
    V=torch.tensor(d.V.values,dtype=torch.float64).reshape(-1,1); N=torch.tensor(d.N.values,dtype=torch.float64).reshape(-1,1)
    M_inf=0.5*(1+torch.tanh((V-p['V1'])/p['V2'])); N_inf=0.5*(1+torch.tanh((V-p['V3'])/p['V4'])); tau=1/torch.cosh((V-p['V3'])/(2*p['V4']))
    fV=(-p['g_Ca']*M_inf*(V-p['V_Ca'])-p['g_K']*N*(V-p['V_K'])-p['g_L']*(V-p['V_L'])+p['I'])/p['C']; fN=p['phi']*(N_inf-N)/tau
    ref=np.array([R.ml_rhs(0,[a,b],p) for a,b in zip(d.V.values,d.N.values)])
    print(f"  {n:11s} max|fV_pinn - f_true|={np.abs(fV.numpy().ravel()-ref[:,0]).max():.1e}, max|fN|={np.abs(fN.numpy().ravel()-ref[:,1]).max():.1e}")
print("C. NODE pipeline: plug the TRUE vector field (scaled coords) into the same train/eval code path")
for n,sc in R.SCENARIOS.items():
  for window in ["extrap","full"]:
    p=sc['params']; t_all,y_all=R.ground_truth(n) if False else (pd.read_csv(D+n+"_ground_truth.csv").t.values.astype(np.float32), pd.read_csv(D+n+"_ground_truth.csv")[['V','N']].values.astype(np.float32))
    tr=np.ones_like(t_all,bool) if window=="full" else t_all<=200
    t_tr,y_tr=t_all[tr],y_all[tr]; lo,hi=y_tr.min(0),y_tr.max(0); ysc=(y_tr-lo)/(hi-lo); tmin,tmax=t_tr.min(),t_tr.max(); T=tmax-tmin
    lo_t,hi_t=torch.tensor(lo),torch.tensor(hi)
    def f(tau,y):
        Y=y*(hi_t-lo_t)+lo_t; V,N=Y[...,0],Y[...,1]
        m=0.5*(1+torch.tanh((V-p['V1'])/p['V2'])); nn_=0.5*(1+torch.tanh((V-p['V3'])/p['V4'])); ta=1/torch.cosh((V-p['V3'])/(2*p['V4']))
        dV=(-p['g_Ca']*m*(V-p['V_Ca'])-p['g_K']*N*(V-p['V_K'])-p['g_L']*(V-p['V_L'])+p['I'])/p['C']; dN=p['phi']*(nn_-N)/ta
        return torch.stack([dV,dN],-1)*float(T)/(hi_t-lo_t)
    y0=torch.tensor(ysc[0],dtype=torch.float32)
    tsc=torch.tensor((t_tr-tmin)/T,dtype=torch.float32); te=torch.tensor((t_all-tmin)/T,dtype=torch.float32)
    p_train=odeint(f,y0,tsc,method='dopri5',atol=1e-5,rtol=1e-4).numpy()
    p_eval=odeint(f,y0,te,method='dopri5').numpy()*(hi-lo)+lo
    mtr=R.metrics(y_tr,p_train*(hi-lo)+lo); mev=R.metrics(y_all,p_eval)
    print(f"  {n:11s} {window:6s} train-tol: R2_V={mtr['R2_V']:.5f} RMSE_V={mtr['RMSE_V']:.3f} | eval-tol (all t): R2_V={mev['R2_V']:.5f} RMSE_V={mev['RMSE_V']:.4f}")
