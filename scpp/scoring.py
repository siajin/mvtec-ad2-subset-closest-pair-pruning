"""Single-layer native RMS 1NN adapter; no resize, threshold or GT evaluation."""
import math

import numpy as np
import torch

from .selection import precision_settings


@torch.inference_mode()
def nearest_scores(queries: np.ndarray, bank: np.ndarray, device='cpu', query_chunk=4096) -> np.ndarray:
    if queries.ndim != 2 or bank.ndim != 2 or queries.shape[1] != bank.shape[1] or not len(bank) or not bank.shape[1]:
        raise ValueError('queries and nonempty bank must be [N,D] arrays with the same D')
    if queries.dtype not in (np.float16, np.float32) or bank.dtype not in (np.float16, np.float32):
        raise ValueError('queries and bank must be float16 or float32')
    if not isinstance(query_chunk, int) or query_chunk < 1:
        raise ValueError('query_chunk must be positive')
    if not np.isfinite(queries).all() or not np.isfinite(bank).all():
        raise ValueError('queries and bank must be finite')
    precision_settings()
    b = torch.as_tensor(np.array(bank, copy=True), device=device).float()
    bn = b.square().sum(1)
    values = []
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        for start in range(0, len(queries), query_chunk):
            q = torch.as_tensor(np.array(queries[start:start+query_chunk], copy=True), device=device).float()
            d = (q.square().sum(1)[:,None] + bn[None,:] - 2.0*(q@b.T)).clamp_min_(0)
            values.append((d.min(1).values.sqrt()/math.sqrt(b.shape[1])).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, np.float32)
