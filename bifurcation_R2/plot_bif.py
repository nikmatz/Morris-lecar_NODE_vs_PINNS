import numpy as np, json, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from ml import *
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':7})
LPC={"Hopf":88.29,"SNLC":115.6}; HC={"Homoclinic":35.01}
def load_stable(n):
    d=json.load(open(f'{n}_cyc.json')); rows=d['up']+d['down']
    if n=="SNLC":
        c=json.load(open('SNLC_cont.json')); rows+=c['up']+c['down']
    if n=="Homoclinic":
        f=json.load(open('Homoclinic_fine.json')); rows+=f['up']+f['down']
        c=json.load(open('Homoclinic_cont.json')); rows+=c['up']+c['down']
    rows=[r for r in rows if r[3]>1]
    if n=='Hopf': rows=[r for r in rows if r[3]>20]
    return np.array(rows)
def draw(ax,n,xl=(0,120),inset=False):
    p=REG[n]; V,I,tr,det=eq_branch(p)
    st=(tr<0)&(det>0); m=(I>=xl[0]-1)&(I<=xl[1]+1)
    Vs=np.where(st&m,V,np.nan); Vu=np.where(~st&m,V,np.nan)
    ax.plot(I,Vs,'k-',lw=1.6,label='stable equilibrium')
    ax.plot(I,Vu,'k--',lw=1.0,label='unstable equilibrium')
    c=load_stable(n); c=c[(c[:,0]>=xl[0])&(c[:,0]<=xl[1])]
    ax.plot(c[:,0],c[:,1],'o',ms=2.2,color='tab:blue',label='stable limit cycle (max/min $V$)')
    ax.plot(c[:,0],c[:,2],'o',ms=2.2,color='tab:blue')
    if n=='Hopf':  # stable cycles close to the fold, from unst_fold.py
        sf=np.array(json.load(open('Hopf_stab_fold.json'))); sf=sf[(sf[:,0]>=xl[0])&(sf[:,0]<=xl[1])]
        ax.plot(sf[:,0],sf[:,1],'o',ms=2.2,color='tab:blue'); ax.plot(sf[:,0],sf[:,2],'o',ms=2.2,color='tab:blue')
    for f in glob.glob(f'{n}_unst_*.json'):
        u=np.array(json.load(open(f)))
        # Hopf: below I = 89.2 use the fold continuation (unst_fold.py), which resolves the branch up to the LPC
        if n=='Hopf' and len(u) and not f.endswith('_fold.json'): u=u[u[:,0]>=89.2]
        if len(u):
            ax.plot(u[:,0],u[:,1],'o',ms=2.2,mfc='none',color='tab:red',label='unstable limit cycle (max/min $V$)')
            ax.plot(u[:,0],u[:,2],'o',ms=2.2,mfc='none',color='tab:red')
    sn,hb=special_points(p)
    for Ii,Vi in sn:
        if xl[0]<=Ii<=xl[1]: ax.plot(Ii,Vi,'s',ms=5,color='tab:green',zorder=5); 
    for Ii,Vi in hb:
        if xl[0]<=Ii<=xl[1]: ax.plot(Ii,Vi,'^',ms=6,color='tab:orange',zorder=5)
    return sn,hb
fig,axs=plt.subplots(1,3,figsize=(12,4.0),sharey=True)
titles={"Hopf":"(a) Hopf regime ($\\varphi=0.04$)","SNLC":"(b) SNLC regime ($\\varphi=0.067$)","Homoclinic":"(c) Homoclinic regime ($\\varphi=0.23$)"}
for ax,n in zip(axs,["Hopf","SNLC","Homoclinic"]):
    sn,hb=draw(ax,n)
    ax.set_xlim(0,120); ax.set_ylim(-65,50); ax.set_title(titles[n],fontsize=10)
    ax.set_xlabel('$I_{\\mathrm{ext}}$ ($\\mu$A/cm$^2$)'); ax.grid(alpha=.3)
    Itr=TRAIN[n]; ax.axvline(Itr,color='gray',ls=':',lw=1.2); ax.text(Itr+1,44,f'$I={Itr:g}$',fontsize=8,color='dimgray') if n!='Homoclinic' else ax.text(36.5,-52,f'$I={Itr:g}$',fontsize=8,color='dimgray')
axs[0].set_ylabel('$V$ (mV)')
# annotations
a=axs[0]; a.annotate('H (subcritical)\n$I\\approx93.9$',xy=(93.86,-25.3),xytext=(99,-38),fontsize=7,arrowprops=dict(arrowstyle='->',lw=.6))
a.annotate('LPC\n$I\\approx88.3$',xy=(88.29,24),xytext=(58,28),fontsize=7,arrowprops=dict(arrowstyle='->',lw=.6))
a.axvline(88.29,color='tab:purple',lw=.6,alpha=.5)
a=axs[1]; a.annotate('SNIC\n$I\\approx39.96$',xy=(39.96,-29.4),xytext=(5,-12),fontsize=7,arrowprops=dict(arrowstyle='->',lw=.6))
a.annotate('H (subcritical)\n$I\\approx97.6$',xy=(97.57,8.3),xytext=(88,-40),fontsize=7,arrowprops=dict(arrowstyle='->',lw=.6))
a.annotate('LPC\n$I\\approx116.0$',xy=(115.9,30.5),xytext=(97,41),fontsize=7,arrowprops=dict(arrowstyle='->',lw=.6))
a=axs[2]; a.annotate('SN\n$I\\approx39.96$',xy=(39.96,-29.4),xytext=(60,-50),fontsize=7,arrowprops=dict(arrowstyle='->',lw=.6))
a.text(3,25,'upper equilibrium stable\nfor $I>36.3$ (H, subcritical)',fontsize=7)
a.axvline(35.01,color='tab:purple',lw=.6,alpha=.6); a.text(14,-60,'HC $I\\approx35.0$',fontsize=7,color='tab:purple')
# inset for homoclinic
ins=inset_axes(axs[2],width="38%",height="38%",loc='upper right',borderpad=1.0)
draw(ins,"Homoclinic",xl=(33,42)); ins.set_xlim(33.5,41.5); ins.set_ylim(-45,22); ins.axvline(35.01,color='tab:purple',lw=.6); ins.axvline(35.5,color='gray',ls=':',lw=1.2)
ins.tick_params(labelsize=6); ins.grid(alpha=.3)
ins.text(36.5,9,'H',fontsize=7,color='tab:orange'); ins.text(35.1,-40,'HC',fontsize=7,color='tab:purple'); ins.text(39.0,-40,'SN',fontsize=7,color='tab:green'); ins.axvline(40.6,color='tab:purple',lw=.6,ls='--'); ins.text(40.65,-40,'LPC',fontsize=6,color='tab:purple')
from matplotlib.lines import Line2D
h=[Line2D([],[],color='k',lw=1.6),Line2D([],[],color='k',ls='--'),Line2D([],[],marker='o',ls='',color='tab:blue',ms=3),
   Line2D([],[],marker='o',ls='',mfc='none',color='tab:red',ms=3),Line2D([],[],marker='s',ls='',color='tab:green'),
   Line2D([],[],marker='^',ls='',color='tab:orange'),Line2D([],[],color='gray',ls=':')]
fig.legend(h,['stable equilibrium','unstable equilibrium','stable limit cycle ($V_{\\max}$, $V_{\\min}$)','unstable limit cycle ($V_{\\max}$, $V_{\\min}$)','saddle-node of equilibria (SN/SNIC)','Hopf point (H)','training current'],
           loc='lower center',ncol=4,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.01))
fig.tight_layout(rect=(0,0.1,1,1))
fig.savefig('ML_bifurcation_diagrams.eps',format='eps'); fig.savefig('ML_bifurcation_diagrams.png',dpi=200)
