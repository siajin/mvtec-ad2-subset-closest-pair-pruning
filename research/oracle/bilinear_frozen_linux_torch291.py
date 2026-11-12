"""Explicit FP32 contraction order of this Linux Torch2.9.1 bilinear CPU.

Portability implementation, not a new score or resize policy. Every exported
FP32/FP16 hash must match the preserved generation reference before metric use.
Sources: PyTorch v2.9.1 aten/src/ATen/native/UpSample.h and
aten/src/ATen/native/cpu/UpSampleKernel.cpp (HelperInterpLinear, Interpolate).
Only explicit llvm.fma.f32; no fastmath/reassociation or tolerance.
"""
import numpy as np
from numba import njit,types
from numba.extending import intrinsic
from llvmlite import ir

@intrinsic
def fma32(typingctx,a,b,c):
    sig=types.float32(types.float32,types.float32,types.float32)
    def codegen(context,builder,signature,args):
        name='llvm.fma.f32';fn=builder.module.globals.get(name)
        if fn is None:fn=ir.Function(builder.module,ir.FunctionType(ir.FloatType(),[ir.FloatType()]*3),name=name)
        return builder.call(fn,args)
    return sig,codegen

@njit(cache=True)
def weights(size,out_size):
    lo=np.empty(out_size,np.int64);hi=np.empty(out_size,np.int64);w0=np.empty(out_size,np.float32);w1=np.empty(out_size,np.float32)
    scale=np.float32(size)/np.float32(out_size)
    for i in range(out_size):
        if size==out_size:lo[i]=i;hi[i]=i;w0[i]=np.float32(1);w1[i]=np.float32(0);continue
        pos=np.float32(np.float32(scale*np.float32(np.float32(i)+np.float32(.5)))-np.float32(.5));pos=max(pos,np.float32(0));x=min(int(np.floor(pos)),size-1)
        lo[i]=x;hi[i]=min(x+1,size-1);w1[i]=min(max(pos-np.float32(x),np.float32(0)),np.float32(1));w0[i]=np.float32(1)-w1[i]
    return lo,hi,w0,w1

@njit(cache=True)
def kernel(native,oh,ow):
    n,h,w=native.shape;y0,y1,hy0,hy1=weights(h,oh);x0,x1,wx0,wx1=weights(w,ow)
    out=np.empty((n,oh,ow),np.float32)
    for m in range(n):
        for y in range(oh):
            for x in range(ow):
                t0=fma32(native[m,y0[y],x0[x]],wx0[x],np.float32(native[m,y0[y],x1[x]]*wx1[x]))
                t1=fma32(native[m,y1[y],x0[x]],wx0[x],np.float32(native[m,y1[y],x1[x]]*wx1[x]))
                out[m,y,x]=fma32(t0,hy0[y],np.float32(t1*hy1[y]))
    return out

def resize(values,shape):
    a=np.asarray(values,dtype=np.float32)
    if a.ndim!=3 or min(a.shape)<=0 or len(shape)!=2 or min(shape)<=0:raise ValueError('Positive N,H,W and output shape required')
    return kernel(np.ascontiguousarray(a),int(shape[0]),int(shape[1]))
