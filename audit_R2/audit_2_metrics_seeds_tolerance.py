import os
import numpy as np, torch, json, glob, sys, os, pandas as pd
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')); import r2_experiments as R
from torchdiffeq import odeint
RES=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results_R2")+"/"
print("D1. Recompute metrics from saved predictions (all runs)")
bad=0;n=0
for f in glob.glob(RES+"*/*.json"):
    r=json.load(open(f)); z=np.load(f[:-5]+".npz"); tr=z['train_mask']; pa=np.nan_to_num(z['pred'],nan=1e6,posinf=1e6,neginf=-1e6)
    if 'full' in r: m=R.metrics(z['y_true'],pa); ok=abs(m['R2_V']-r['full']['R2_V'])<1e-5
    else: m=R.metrics(z['y_true'][~tr],pa[~tr]); ok=abs(m['R2_V']-r['test']['R2_V'])<1e-5
    n+=1; bad+=not ok
print(f"   {n} runs, mismatches: {bad}")
print("D2. Seeds produce different initialisations / noise identical across PINN & NODE for same seed")
for sc in ['hopf']:
    a=np.load(RES+f"noise/noise_{sc}_pinn_tanh_d3w128_s42_n05_e10000.npz")['y_train_used']; b=np.load(RES+f"noise/noise_{sc}_node_tanh_d3w128_s42_n05_e10000.npz")['y_train_used']
    c=np.load(RES+f"noise/noise_{sc}_pinn_tanh_d3w128_s43_n05_e10000.npz")['y_train_used']; g=np.load(RES+f"noise/noise_{sc}_pinn_tanh_d3w128_s42_n05_e10000.npz")['y_true'][:len(a)]
    print("   same seed PINN vs NODE noisy data identical:",np.allclose(a,b)," | seed42 vs 43 differ:",not np.allclose(a,c)," | noise std/range V: %.4f"%(np.std(a[:,0]-g[:,0])/np.ptp(g[:,0])))
    p1=np.load(RES+f"seeds/seeds_{sc}_node_tanh_d3w128_s42_n00_e10000.npz")['pred']; p2=np.load(RES+f"seeds/seeds_{sc}_node_tanh_d3w128_s43_n00_e10000.npz")['pred']
    print("   seed42 vs seed43 NODE predictions differ:",not np.allclose(p1,p2))
print("E. Solver tolerance needed for the TRUE homoclinic field (full window, scaled coords)")
d=pd.read_csv(RES+"data/homoclinic_ground_truth.csv"); t=d.t.values.astype(np.float32); y=d[['V','N']].values.astype(np.float32)
p=R.SCENARIOS['homoclinic']['params']; lo,hi=y.min(0),y.max(0); T=float(t.max()-t.min()); lo_t,hi_t=torch.tensor(lo),torch.tensor(hi)
def f(tau,yy):
    Y=yy*(hi_t-lo_t)+lo_t; V,N=Y[...,0],Y[...,1]
    m=0.5*(1+torch.tanh((V-p['V1'])/p['V2'])); nn_=0.5*(1+torch.tanh((V-p['V3'])/p['V4'])); ta=1/torch.cosh((V-p['V3'])/(2*p['V4']))
    return torch.stack([(-p['g_Ca']*m*(V-p['V_Ca'])-p['g_K']*N*(V-p['V_K'])-p['g_L']*(V-p['V_L'])+p['I'])/p['C'],p['phi']*(nn_-N)/ta],-1)*T/(hi_t-lo_t)
ysc=(y-lo)/(hi-lo); y0=torch.tensor(ysc[0]); ts=torch.tensor((t-t.min())/T)
for rt,at in [(1e-4,1e-5),(1e-5,1e-6),(1e-6,1e-7),(1e-7,1e-9)]:
    pr=odeint(f,y0,ts,method='dopri5',rtol=rt,atol=at).numpy()
    print(f"   rtol={rt:g} atol={at:g}: scaled MSE loss of true field={np.mean((pr-ysc)**2):.2e}  R2_V={R.metrics(y,pr*(hi-lo)+lo)['R2_V']:.4f}")
print("F. Final training loss of trained homoclinic NODEs (scaled MSE) for comparison")
for f_ in sorted(glob.glob(RES+"act/*.json"))+sorted(glob.glob(RES+"main/*node*.json")):
    r=json.load(open(f_)); print("  ",r['job_id'],"loss=%.2e"%r['final_loss'], "R2_V=%.3f"%r['full']['R2_V'])
