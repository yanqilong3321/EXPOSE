#!/usr/bin/env python3
"""Aggregate new user-generated runs (no archived result data are required)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def summarize(records):
    grouped={}
    for r in records:
        grouped.setdefault((r['cell'],r['method']),[]).append(r)
    rows=[]
    for (cell,method),values in sorted(grouped.items()):
        seeds=[r['seed'] for r in values]
        assert sorted(seeds)==list(range(1,21)),f'{cell}/{method}: exactly seeds 1-20 are required, with no duplicates'
        vals=np.asarray([r['accuracy'] for r in sorted(values,key=lambda r:r['seed'])])
        rows.append(dict(cell=cell,method=method,runs=len(vals),mean=float(vals.mean()),sd=float(vals.std(ddof=0))))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,required=True,help='Directory written by run.py')
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/aggregate')
    ap.add_argument('--require-all',action='store_true',help='Require all 28 EXPOSE cells (and all 28 DnD cells if present)')
    a=ap.parse_args()
    records=[json.loads(p.read_text()) for p in a.results.rglob('result.json')]
    assert records,'No completed runs found'
    table=summarize(records)
    if a.require_all:
        expected={c['cell'] for c in json.loads((ROOT/'configs/table5.json').read_text())['cells']}
        for method in {r['method'] for r in table}:
            assert {r['cell'] for r in table if r['method']==method}==expected,f'{method}: incomplete 28-cell coverage'
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(f'Aggregated {len(records)} newly generated runs into {len(table)} method/condition rows.')


if __name__=='__main__':main()
