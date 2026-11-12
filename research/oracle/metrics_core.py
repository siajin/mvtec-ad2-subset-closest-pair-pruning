"""Exact FP16-tied pooled metrics from bounded-memory histograms."""
import numpy as np
from scipy import ndimage
NBINS=31744  # all finite nonnegative IEEE binary16 values

def histograms(truth,score):
 y=np.asarray(truth,bool);s=np.array(score,np.float16,copy=True)
 if not np.isfinite(s).all() or np.any(s<0):raise ValueError('finite nonnegative score required')
 s[s==0]=np.float16(0)  # IEEE -0 and +0 are one ranking tie.
 bits=s.view(np.uint16).ravel();flat=y.ravel()
 pos=np.bincount(bits[flat],minlength=NBINS).astype(np.int64)
 neg=np.bincount(bits[~flat],minlength=NBINS).astype(np.int64)
 lab,n=ndimage.label(y,structure=ndimage.generate_binary_structure(2,1))
 sizes=np.bincount(lab.ravel());weights=np.zeros(n+1,np.float64)
 if n:weights[1:]=1/sizes[1:]
 pro=np.bincount(bits,weights=weights[lab.ravel()],minlength=NBINS)
 return pos,neg,pro,int(n)

def area(x,y,limit):
 end=np.searchsorted(x,limit,side='right')
 xx=x[:end];yy=y[:end]
 if len(xx)==0:xx=np.array([0.]);yy=np.array([0.])
 if xx[-1]<limit:
  value=yy[-1]+(y[end]-yy[-1])*(limit-xx[-1])/(x[end]-xx[-1])
  xx=np.r_[xx,limit];yy=np.r_[yy,value]
 return float(np.trapezoid(yy,xx))

def rank_metrics(pos,neg,pro,nregions):
 used=(pos+neg)>0;P=int(pos.sum());N=int(neg.sum())
 if not P or not N:raise ValueError('positive and negative pooled pixels required')
 tp=np.cumsum(pos[used][::-1],dtype=np.float64);fp=np.cumsum(neg[used][::-1],dtype=np.float64)
 x=np.r_[0.,fp/N];y=np.r_[0.,tp/P]
 a=area(x,y,.05);pauc=.5*(1+(a-.00125)/(.05-.00125))
 result={'pauroc_005':float(pauc),'pixel_auroc':float(np.trapezoid(y,x)),
         'pixel_ap':float(np.sum(np.diff(np.r_[0.,tp/P])*tp/np.maximum(tp+fp,1)))}
 if nregions:
  pp=np.r_[0.,np.cumsum(pro[used][::-1],dtype=np.float64)/nregions]
  result['aupro_005']=area(x,pp,.05)/.05;result['aupro_030']=area(x,pp,.30)/.30
 return result

def confusion(y,p):
 y=np.asarray(y,bool);p=np.asarray(p,bool)
 return np.array([np.count_nonzero(y&p),np.count_nonzero(~y&p),np.count_nonzero(y&~p),np.count_nonzero(~y&~p)],np.int64)

def fixed_metrics(c):
 tp,fp,fn,tn=map(int,c);den=2*tp+fp+fn
 return {'fixed_f1':2*tp/den if den else 0.,'fixed_precision':tp/(tp+fp) if tp+fp else 0.,'fixed_recall':tp/(tp+fn) if tp+fn else 0.,'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def bootstrap_category(hist_by_method,counts_by_method,regions,seed=42,repetitions=10000,batch=128):
 """Return paired per-replicate pooled metrics; no pixel bootstrap."""
 K=len(regions);rng=np.random.default_rng(seed)
 weights=rng.multinomial(K,np.full(K,1/K),size=repetitions)
 result={m:{k:np.full(repetitions,np.nan) for k in ('pauroc_005','aupro_005','fixed_f1')} for m in hist_by_method}
 for method,(p,n,pro) in hist_by_method.items():
  keep=(p.sum(0)+n.sum(0))>0
  pp=p[:,keep][:,::-1].astype(np.float64);nn=n[:,keep][:,::-1].astype(np.float64);rr=pro[:,keep][:,::-1]
  for start in range(0,repetitions,batch):
   w=weights[start:start+batch];ph=w@pp;nh=w@nn;rh=w@rr
   tp=np.cumsum(ph,axis=1);fp=np.cumsum(nh,axis=1)
   P=ph.sum(1);N=nh.sum(1);R=w@regions
   for j in range(len(w)):
    if P[j]<=0 or N[j]<=0 or R[j]<=0:continue
    x=np.r_[0.,fp[j]/N[j]];y=np.r_[0.,tp[j]/P[j]]
    a=area(x,y,.05);result[method]['pauroc_005'][start+j]=.5*(1+(a-.00125)/(.05-.00125))
    pr=np.r_[0.,np.cumsum(rh[j])/R[j]];result[method]['aupro_005'][start+j]=area(x,pr,.05)/.05
   cc=w@counts_by_method[method];den=2*cc[:,0]+cc[:,1]+cc[:,2]
   result[method]['fixed_f1'][start:start+len(w)]=np.divide(2*cc[:,0],den,out=np.zeros(len(w)),where=den>0)
 return result,weights
