"""Exhaustive bidirectional FP64 cross-bank NN in one GEMM traversal.

Cross-bank same-coordinate rows remain legal zero neighbors. Shared FIT IDs are
filtered only for conditional summaries, never masked from the search bank.
"""
import math,time
import numpy as np
import torch
@torch.inference_mode()
def cross_nearest(a,b,chunk=4096,deadline=None,heartbeat=None):
 assert a.dtype==b.dtype==torch.float64 and a.ndim==b.ndim==2 and a.shape[1]==b.shape[1]
 assert a.device==b.device and torch.isfinite(a).all() and torch.isfinite(b).all()
 na,d=a.shape;nb=len(b);an=a.square().sum(1);bn=b.square().sum(1)
 jb=torch.full((nb,),-1,device=a.device,dtype=torch.int64);vb=torch.full((nb,),torch.inf,device=a.device,dtype=torch.float64);qa=[];ja=[];max_error=0.;max_fraction=0.;eps=torch.finfo(torch.float64).eps;gamma=d*eps/(1-d*eps);started=time.perf_counter()
 for start in range(0,na,chunk):
  if deadline is not None and time.time()>=deadline:raise TimeoutError('cross geometry compute cutoff; preserve finished pair markers')
  q=a[start:start+chunk];d2=(an[start:start+len(q),None]+bn[None,:]-2.0*(q@b.T)).clamp_min_(0)
  value,j=d2.min(1);direct=(q-b[j]).square().sum(1);bound=16*gamma*(an[start:start+len(q)]+bn[j]+2*(an[start:start+len(q)]*bn[j]).sqrt()+1);error=(value-direct).abs();assert bool((error<=bound).all()),'FP64 selected-pair roundoff bound exceeded'
  max_error=max(max_error,float(error.max()));max_fraction=max(max_fraction,float((error/bound).max()));qa.append((direct.sqrt()/math.sqrt(d)).cpu().numpy());ja.append(j.cpu().numpy().astype(np.int32))
  cv,ci=d2.min(0);improved=cv<vb;vb=torch.where(improved,cv,vb);jb=torch.where(improved,ci+start,jb)
  if heartbeat:heartbeat(done=min(start+chunk,na),total=na,elapsed=time.perf_counter()-started)
  del d2,q,direct,value
 directb=(b-a[jb]).square().sum(1);bound=16*gamma*(bn+an[jb]+2*(bn*an[jb]).sqrt()+1);error=(vb-directb).abs();assert bool((error<=bound).all()),'reverse FP64 selected-pair roundoff bound exceeded'
 max_error=max(max_error,float(error.max()));max_fraction=max(max_fraction,float((error/bound).max()))
 return np.concatenate(qa),np.concatenate(ja),(directb.sqrt()/math.sqrt(d)).cpu().numpy(),jb.cpu().numpy().astype(np.int32),{'queries_a':na,'queries_b':nb,'all_candidate_pairs_examined':na*nb,'dimension':d,'dtype':'FP64','query_chunk':chunk,'bank_chunk':'entire B bank','shared_coordinates_zero_legal':True,'self_mask':False,'tie_rule':'first stored target row, strict update across A chunks','direct_selected_pair_recomputed':True,'max_squared_reduction_error':max_error,'max_error_to_bound':max_fraction,'seconds':time.perf_counter()-started}
def synthetic_control():
 a=torch.zeros((5,1280),dtype=torch.float64);b=torch.zeros((4,1280),dtype=torch.float64);a[:,0]=torch.tensor([0.,1.,1.,4.,20.]);b[:,0]=torch.tensor([1.,1.,5.,30.]);da,ja,db,jb,audit=cross_nearest(a,b,chunk=2)
 brute=(a[:,None]-b[None,:]).square().sum(2).numpy();assert np.array_equal(ja,np.argmin(brute,axis=1)) and np.array_equal(jb,np.argmin(brute,axis=0));assert np.array_equal(da,np.sqrt(brute.min(1))/math.sqrt(1280)) and np.array_equal(db,np.sqrt(brute.min(0))/math.sqrt(1280));assert da[1]==da[2]==0
 # Rigid translation preserves all within-bank distances but changes cross location.
 c=a+8;dd,_,_,_,_=cross_nearest(a,c,chunk=3);assert dd.min()>0
 return {'status':'pass','unequal_sizes_and_last_chunk':True,'duplicate_coordinate_zero_and_first_row_ties':True,'reverse_global_min_across_chunks':True,'direct_brute_force_both_directions':True,'GT_opened':False}
