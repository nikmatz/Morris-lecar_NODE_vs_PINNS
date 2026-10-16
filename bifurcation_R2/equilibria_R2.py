"""R2: corrected equilibrium analysis (Table 3) and phase portraits (Figure 2) at the training currents.
Jacobian from eq. (jacobian_final) of the paper (the R1 script had an extra -gK(V-VK) term in a11)."""
import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ml import *
plt.rcParams.update({'font.size':9})
def J_analytic(V,p):
    C=p['C']; N=ninf(V,p)
    dm=0.5*(1-np.tanh((V-p['V1'])/p['V2'])**2)/p['V2']; dn=0.5*(1-np.tanh((V-p['V3'])/p['V4'])**2)/p['V4']
    a11=(-p['g_Ca']*dm*(V-p['V_Ca'])-p['g_Ca']*minf(V,p)-p['g_K']*N-p['g_L'])/C
    a12=-p['g_K']*(V-p['V_K'])/C; a21=p['phi']*dn/tau(V,p); a22=-p['phi']/tau(V,p)
    return np.array([[a11,a12],[a21,a22]])
def classify(ev):
    re=ev.real
    if re.max()>0 and re.min()<0: return "saddle"
    cx=abs(ev.imag).max()>1e-12
    if re.max()<0: return "stable focus" if cx else "stable node"
    return "unstable focus" if cx else "unstable node"
Y0={"Hopf":[0,0],"SNLC":[0,0],"Homoclinic":[10,0.3]}
rows=[]; fig,axs=plt.subplots(1,3,figsize=(12,3.8))
for ax,n in zip(axs,["Hopf","SNLC","Homoclinic"]):
    p=REG[n]; I=TRAIN[n]; V=np.linspace(-70,60,400001); g=I_eq(V,p)-I
    roots=[V[k]-g[k]*(V[k+1]-V[k])/(g[k+1]-g[k]) for k in range(len(V)-1) if g[k]*g[k+1]<0]
    s=solve_ivp(rhs,[0,300],Y0[n],args=(p,I),method='LSODA',rtol=1e-9,atol=1e-11,t_eval=np.linspace(0,300,3000))
    vmin,vmax=min(s.y[0].min(),min(roots))-8,max(s.y[0].max(),max(roots))+8
    nmax=max(s.y[1].max(),0.05)*1.15
    Vg,Ng=np.meshgrid(np.linspace(vmin,vmax,40),np.linspace(-0.02,nmax,40))
    dV,dN=np.array(rhs(0,[Vg,Ng],p,I))
    #ax.streamplot(Vg,Ng,dV/(np.ptp(Vg)),dN/(np.ptp(Ng)),color='0.8',density=1.0,linewidth=.5,arrowsize=.6)
    vv=np.linspace(vmin,vmax,600)
    ax.plot(vv,(I-p['g_Ca']*minf(vv,p)*(vv-p['V_Ca'])-p['g_L']*(vv-p['V_L']))/(p['g_K']*(vv-p['V_K'])),'r--',lw=1,label='$V$-nullcline')
    ax.plot(vv,ninf(vv,p),'g--',lw=1,label='$N$-nullcline')
    ax.plot(s.y[0],s.y[1],'b',lw=1,label='training trajectory')
    ax.plot(*Y0[n],'ko',ms=3)
    for r in roots:
        J=J_analytic(r,p); ev=np.linalg.eigvals(J); typ=classify(ev)
        rows.append(dict(regime=n,I=I,V=r,N=float(ninf(r,p)),trace=float(np.trace(J)),det=float(np.linalg.det(J)),
                         ev=[complex(e) for e in ev],type=typ,J=J.tolist()))
        st=typ.startswith('stable')
        ax.plot(r,ninf(r,p),'o' if typ!='saddle' else 'X',ms=7,mfc='k' if st else 'w',mec='k',zorder=6)
    ax.set_xlim(vmin,vmax); ax.set_ylim(-0.02,nmax); ax.set_xlabel('$V$ (mV)')
    ax.set_title({"Hopf":"(a) Hopf, $I=90$","SNLC":"(b) SNLC, $I=42$","Homoclinic":"(c) Homoclinic, $I=35.5$"}[n])
axs[0].set_ylabel('$N$')
from matplotlib.lines import Line2D
h=[Line2D([],[],color='r',ls='--'),Line2D([],[],color='g',ls='--'),Line2D([],[],color='b'),Line2D([],[],marker='o',ls='',mfc='k',mec='k'),
   Line2D([],[],marker='o',ls='',mfc='w',mec='k'),Line2D([],[],marker='X',ls='',mfc='w',mec='k',ms=7),Line2D([],[],marker='o',ls='',color='k',ms=3)]
fig.legend(h,['$V$-nullcline','$N$-nullcline','trajectory $t\\in[0,300]$ ms','stable equilibrium','unstable equilibrium','saddle','initial condition $y_0$'],loc='lower center',ncol=7,fontsize=8,frameon=False)
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig('ML_phase_portraits_R2.eps'); fig.savefig('ML_phase_portraits_R2.png',dpi=200)
for r in rows:
    ev=r['ev']; print(r['regime'],r['I'],'(%.4f, %.4f)'%(r['V'],r['N']),'tr=%.4f det=%.5f'%(r['trace'],r['det']),
      ', '.join(('%.4f'%e.real)+('' if abs(e.imag)<1e-12 else ('%+.4fi'%e.imag)) for e in ev), r['type'], np.round(r['J'],5).tolist())
json.dump([{k:(v if k!='ev' else [[e.real,e.imag] for e in v]) for k,v in r.items()} for r in rows],open('equilibria_R2.json','w'),indent=1)
