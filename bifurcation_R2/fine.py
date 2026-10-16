import numpy as np, json, sys
from cycles_lib import settle
from ml import *
name=sys.argv[1]; a,b,h=map(float,sys.argv[2:5]); p=REG[name]
Is=np.round(np.arange(a,b+1e-9,h),4); out={}
for dr,seq,y0 in [("up",Is,[-60.,0.]),("down",Is[::-1],None)]:
    y=y0;res=[]
    for I in seq:
        if y is None: y=[10.,0.3]
        y,mx,mn,amp,per=settle(p,I,y,T=3000,Trec=1000)
        res.append([float(I),float(mx),float(mn),float(amp),float(per)])
    out[dr]=res
json.dump(out,open(f"{name}_fine.json","w"))
for dr in out:
    print(dr); 
    for r in out[dr]: print('  ',[round(x,3) for x in r])
