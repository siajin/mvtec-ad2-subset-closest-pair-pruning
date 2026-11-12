"""Portable array adapter around the byte-preserved historical pruning rule.

This adapter does not reconstruct the private cache manifest or reproduce its
source order. Exact historical IDs require the original ordered feature pool.
"""
from __future__ import annotations

import numpy as np
import torch

from research.scpp.cp_select import prune_subset


def precision_settings() -> None:
    """Apply the FP32/highest, TF32-off contract without autocast."""
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def select_representatives(
    features: np.ndarray,
    target: int = 100_000,
    subsets: int = 100,
    seed: int = 42,
    device: str = 'cpu',
    query_chunk: int = 2048,
) -> tuple[np.ndarray, list[dict]]:
    """Select unique FIT row IDs, retaining the historical RNG/round rule.

    ``features`` must be the frozen FP16 pool [N,D]. Smaller target/subsets
    values support toy demonstrations and are different experimental settings.
    Rejected candidates remain eligible; only selected IDs are excluded later.
    """
    if not isinstance(features, np.ndarray) or features.ndim != 2:
        raise ValueError('features must be a numpy array of shape [N,D]')
    if features.dtype != np.float16:
        raise ValueError('features must be frozen float16; do not silently requantize a pool')
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (target, subsets, seed, query_chunk)):
        raise ValueError('target, subsets, seed and query_chunk must be integers')
    if subsets < 1 or target < 2 * subsets or target % subsets:
        raise ValueError('target must be divisible by subsets, retaining at least two per subset')
    if target > len(features) or features.shape[1] < 1 or query_chunk < 1:
        raise ValueError('invalid candidate count, feature dimension or query chunk')
    # Bound validation memory for a large mmap input.
    for start in range(0, len(features), 8192):
        if not np.isfinite(features[start:start + 8192]).all():
            raise ValueError('features contain NaN or Inf')
    precision_settings()
    rng = np.random.RandomState(seed)
    selected_mask = np.zeros(len(features), dtype=bool)
    selected = []
    trace = []
    subset_size = len(features) // subsets
    per_subset = target // subsets
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for iteration in range(subsets):
            eligible = np.flatnonzero(~selected_mask)
            candidate_ids = rng.choice(eligible, size=min(subset_size, len(eligible)), replace=False)
            kept, rounds = prune_subset(features[candidate_ids], candidate_ids, per_subset, device, query_chunk)
            if selected_mask[kept].any() or len(np.unique(kept)) != per_subset:
                raise RuntimeError('historical selector returned duplicate IDs')
            selected_mask[kept] = True
            selected.extend(kept.tolist())
            trace.append({'iteration':iteration, 'candidate_count':len(candidate_ids), 'rounds':rounds})
    return np.asarray(selected, dtype=np.int64), trace
