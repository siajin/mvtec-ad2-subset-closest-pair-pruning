"""Fixed-feature prototype learning and exhaustive unique-FIT projection."""
import math,time
import numpy as np
import torch
import torch.nn.functional as F

def prototype_distance(query,centers,gradient=True,query_chunk=4096,gradient_floor=1e-6):
 """Native FP32 hard-min forward, selected-pair Euclidean sparse backward."""
 dim=centers.shape[1]
 with torch.no_grad():
  bank_norm=centers.square().sum(1);values=[];indices=[]
  for s in range(0,len(query),query_chunk):
   q=query[s:s+query_chunk];qn=q.square().sum(1)
   d=(qn[:,None]+bank_norm[None,:]-2.0*(q@centers.T)).clamp_min_(0)
   v,j=d.min(1);values.append(v.sqrt()/math.sqrt(dim));indices.append(j);del d
  native=torch.cat(values);winner=torch.cat(indices)
 if not gradient:return native,winner
 selected=F.embedding(winner,centers,sparse=True)
 direct=((query-selected).square().sum(1)/dim).clamp_min(gradient_floor**2).sqrt()
 return native+(direct-direct.detach()),winner

def sparse_clip(parameters,max_norm):
 grads=[];squared=0.
 for p in parameters:
  if p.grad is None:continue
  assert p.grad.is_sparse,'Only selected rows may receive gradients'
  p.grad=p.grad.coalesce();g=p.grad.values();assert bool(torch.isfinite(g).all());grads.append(g);squared=squared+g.square().sum()
 norm=torch.sqrt(squared);factor=(max_norm/(norm+1e-12)).clamp(max=1.)
 for g in grads:g.mul_(factor)
 return float(norm),bool(norm>max_norm)

def oracle_loss(pixel_score,positive_count,normal_scale,normal_weight=.1,anomaly_margin=1.5,rank_margin=.1):
 u=pixel_score/normal_scale;a=u[:positive_count];n=u[positive_count:]
 pull=n.square().mean();push=F.relu(anomaly_margin-a).square().mean();rank=F.relu(rank_margin+n[None,:]-a[:,None]).mean()
 return normal_weight*pull+push+rank,{'pull':float(pull.detach()),'push':float(push.detach()),'rank':float(rank.detach())}

@torch.inference_mode()
def project_unique(prototypes,fit,query_chunk=512,max_rounds=128,deadline=None,heartbeat=None):
 assert prototypes.dtype==fit.dtype==torch.float64 and len(prototypes)<=len(fit)
 N,D=fit.shape;K=len(prototypes);norm=fit.square().sum(1);used=torch.zeros(N,device=fit.device,dtype=torch.bool);assigned=np.full(K,-1,np.int64);rounds=np.zeros(K,np.int16);remaining=np.arange(K);distance=np.zeros(K,np.float64);trace=[];start=time.perf_counter();max_fraction=0.
 eps=torch.finfo(torch.float64).eps;gamma=D*eps/(1-D*eps)
 for turn in range(1,max_rounds+1):
  if not len(remaining):break
  proposals=[];cost=[]
  for s in range(0,len(remaining),query_chunk):
   if deadline is not None and time.time()>=deadline:raise TimeoutError('projection compute cutoff; preserve checkpoints')
   rr=torch.as_tensor(remaining[s:s+query_chunk],device=fit.device);q=prototypes[rr];qn=q.square().sum(1);d=(qn[:,None]+norm[None,:]-2.0*(q@fit.T)).clamp_min_(0)
   if turn>1:d[:,used]=torch.inf
   v,j=d.min(1);direct=(q-fit[j]).square().sum(1);bound=16*gamma*(qn+norm[j]+2*(qn*norm[j]).sqrt()+1);error=(v-direct).abs();assert bool((error<=bound).all());max_fraction=max(max_fraction,float((error/bound).max()));proposals.append(j.cpu().numpy());cost.append(direct.cpu().numpy());del d
  target=np.concatenate(proposals);dist=np.concatenate(cost);order=np.lexsort((target,remaining,dist));_,first=np.unique(target[order],return_index=True);win=order[first];rows=remaining[win];fids=target[win];assert len(np.unique(fids))==len(fids) and np.all(assigned[rows]<0)
  assigned[rows]=fids;rounds[rows]=turn;distance[rows]=np.sqrt(dist[win]/D);used[torch.as_tensor(fids,device=fit.device)]=True;remaining=remaining[assigned[remaining]<0]
  trace.append({'round':turn,'assigned_now':len(rows),'remaining':len(remaining),'seconds':time.perf_counter()-start})
  if heartbeat:heartbeat(**trace[-1])
 assert np.all(assigned>=0),'Unique projection did not converge within declared round cap'
 assert len(np.unique(assigned))==K
 return assigned,distance,rounds,{'status':'complete','candidate_FIT_rows':N,'prototype_rows':K,'unique_FIT_ids':K,'trace':trace,'max_roundoff_bound_fraction':max_fraction,'seconds':time.perf_counter()-start,'global_assignment_optimality_claim':False}

def synthetic_control():
 # Non-tie gradient, including repeated winners; checks the hard-min derivative.
 torch.manual_seed(42);q=torch.randn(7,5,dtype=torch.float64);b=torch.randn(4,5,dtype=torch.float64,requires_grad=True);q=q+2
 full=((q[:,None]-b[None,:]).square().sum(2)/5).min(1).values.sqrt();full.sum().backward();expected=b.grad.clone();c=b.detach().clone().requires_grad_();s,j=prototype_distance(q,c,gradient=True);s.sum().backward();actual=c.grad.to_dense();err=float((expected-actual).abs().max());assert err<1e-12
 # A normal point pulls a prototype; an anomaly pushes it under the hinge.
 c=torch.tensor([[1.,0.]],requires_grad=True);q=torch.tensor([[0.,0.]]);z,_=prototype_distance(q,c);z.square().sum().backward();assert c.grad.to_dense()[0,0]>0
 c=torch.tensor([[1.,0.]],requires_grad=True);z,_=prototype_distance(q,c);F.relu(2-z).square().sum().backward();assert c.grad.to_dense()[0,0]<0
 # Exact duplicate-proposal collision: output must still contain three real IDs.
 f=torch.tensor([[0.,0.],[2.,0.],[4.,0.],[9.,0.]],dtype=torch.float64);p=torch.tensor([[.1,0.],[.1,0.],[3.9,0.]],dtype=torch.float64);ii,dd,rr,a=project_unique(p,f,query_chunk=2);assert len(np.unique(ii))==3 and sorted(ii.tolist())==[0,1,2]
 c=torch.nn.Parameter(torch.tensor([[1.,0.],[3.,0.]]));opt=torch.optim.SparseAdam([c],lr=.1);old=c.detach().clone();z,_=prototype_distance(torch.tensor([[0.,0.],[0.,0.]]),c);z.square().mean().backward();norm,clip=sparse_clip([c],1.);opt.step();assert c[0,0]<old[0,0] and torch.equal(c[1],old[1])
 return {'status':'pass','hard_min_sparse_gradient_max_abs':err,'normal_pull_and_anomaly_push_direction':True,'duplicate_winner_update':True,'unique_projection_collision_resolved':True,'projection_exact_candidates_checked':True,'torch':torch.__version__,'numpy':np.__version__}
