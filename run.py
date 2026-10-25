#!/usr/bin/env python3
"""Run the frozen EXPOSE / paired DnD-Net configurations for Table 5."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
os.environ.setdefault('MKL_NUM_THREADS','8')
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from code import evidence
from code.protocol import ROOT,configure,load_data,prepare,split,kernel_for
from code.training import fit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cell',nargs='+',help='e.g. cora_ml_u40 citeseer_p30; omitted = all 28 cells')
    ap.add_argument('--seeds',nargs='+',type=int,default=list(range(1,21)))
    ap.add_argument('--method',choices=['expose','dnd','both'],default='expose')
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/table5')
    ap.add_argument('--cache',type=Path,default=ROOT/'.cache')
    args=ap.parse_args()
    assert len(set(args.seeds))==len(args.seeds) and set(args.seeds)<=set(range(1,21))
    config_path=ROOT/'configs/table5.json'
    sealed=json.loads(config_path.read_text())
    cells=sealed['cells']
    if args.cell:
        assert set(args.cell)<=set(c['cell'] for c in cells)
        cells=[c for c in cells if c['cell'] in args.cell]
    configure()
    for c in cells:
        data=load_data(c['dataset'])
        methods=('expose','dnd') if args.method=='both' else (args.method,)
        for method in methods:
            pending=[s for s in args.seeds if not (args.out/c['cell']/f's{s}'/method/'result.json').exists()]
            if not pending:continue
            g,operator,a,ctx=prepare(data,c,args.cache,baseline=method=='dnd')
            for seed in pending:
                out=args.out/c['cell']/f's{seed}'/method
                out.mkdir(parents=True,exist_ok=True)
                assert not (out/'predictions.npz').exists(),'Partial fit exists: use a new output directory'
                ev,rng=split(data,c,seed)
                if method=='expose':
                    kernel=kernel_for(operator,ev,c)
                    risk,support=evidence.build_evidence(data.x.to(g.device),ev['train'],ev['observed'],
                        ev['targets'],kernel,ctx.num_classes,c)
                    np.savez_compressed(out/'evidence.npz',risk=risk,support=support)
                    ev.update(evidence.supervision(kernel,ev['observed'],risk,ctx.num_classes,c))
                receipt=fit(g,ctx,ev,rng,c,out,method)
                # Test labels are read only after checkpoint selection and model fitting.
                with np.load(out/'predictions.npz') as z:
                    lookup={int(node):i for i,node in enumerate(z['targets'])}
                    positions=[lookup[int(node)] for node in ev['test_ids']]
                    pred=z['probabilities'][positions].argmax(1)
                accuracy=float((pred==data.y[ev['test_ids']].numpy()).mean()*100)
                record=dict(cell=c['cell'],seed=seed,noise_seed=10,method=method,accuracy=accuracy,
                    config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),**receipt)
                (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
                print(json.dumps(record),flush=True)
            del g,operator,ctx
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=='__main__':main()
