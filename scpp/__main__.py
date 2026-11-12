"""Use ``python -m scpp --help`` for the portable feature-array interface."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .selection import select_representatives
from .scoring import nearest_scores


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    select = sub.add_parser('select', help='SCPP selection from an ordered frozen float16 FIT pool')
    select.add_argument('--features', type=Path, required=True)
    select.add_argument('--target', type=int, default=100000)
    select.add_argument('--subsets', type=int, default=100)
    select.add_argument('--seed', type=int, default=42)
    score = sub.add_parser('score', help='single-layer global-bank RMS 1NN; not the full image evaluator')
    score.add_argument('--queries', type=Path, required=True)
    score.add_argument('--bank', type=Path, required=True)
    for p, chunk in [(select,2048), (score,4096)]:
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--device', default='cpu')
        p.add_argument('--query-chunk', type=int, default=chunk)
    args = parser.parse_args()
    if args.output.suffix != '.npy':
        parser.error('--output must end in .npy')
    meta_path = args.output.with_suffix('.json')
    if args.output.exists() or meta_path.exists():
        parser.error('output already exists; select a new output path')
    if args.device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    inputs = [args.features] if args.command == 'select' else [args.queries, args.bank]
    input_hashes = [{'name':p.name,'sha256':digest(p)} for p in inputs]
    if args.command == 'select':
        a = np.load(args.features, mmap_mode='r', allow_pickle=False)
        result, trace = select_representatives(a,args.target,args.subsets,args.seed,args.device,args.query_chunk)
    else:
        q = np.load(args.queries,mmap_mode='r',allow_pickle=False)
        b = np.load(args.bank,mmap_mode='r',allow_pickle=False)
        result = nearest_scores(q,b,args.device,args.query_chunk)
        trace = None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp = args.output.with_suffix('.npy.tmp')
    with tmp.open('wb') as stream:
        np.save(stream,result,allow_pickle=False)
    tmp.replace(args.output)
    meta = {'command':args.command,'input_files':input_hashes,'output_sha256':digest(args.output),
        'shape':list(result.shape),'torch':torch.__version__,'numpy':np.__version__,
        'device':args.device,'query_chunk':args.query_chunk,'GT_opened':False,'trace':trace}
    if args.command == 'select':
        meta.update(target=args.target,subsets=args.subsets,seed=args.seed)
    meta_path.write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'shape':list(result.shape),'GT_opened':False}))


if __name__ == '__main__':
    main()
