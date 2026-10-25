"""Portable data loading, the released split/noise protocol, and fixed propagation."""
from pathlib import Path
from types import SimpleNamespace
import gc
import numpy as np
import scipy.sparse as sp
import torch
from . import official_ops as ops
from .model import state

ROOT = Path(__file__).resolve().parents[1]


def configure():
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(dataset):
    with np.load(ROOT/'data'/f'{dataset}.npz', allow_pickle=False) as z:
        x = sp.csr_matrix((z['x_data'],z['x_indices'],z['x_indptr']),shape=tuple(z['x_shape']))
        return SimpleNamespace(x=torch.from_numpy(x.toarray()),
            y=torch.from_numpy(z['y'].copy()).long(),
            edge_index=torch.from_numpy(z['edge_index'].copy()).long())


def split(data, config, seed):
    ops.setup_seed(seed)
    n, classes = len(data.y), int(data.y.max())+1
    ti, vi, te = ops.get_split(data.y, n, classes, .05, .20)
    observed, _ = ops.noisify_with_P(data.y[ti].numpy().copy(),classes,
        config['rate'],10,config['noise'])
    rng = state()
    ev = dict(train=ti.numpy(), validation=vi.numpy(), test_ids=te.numpy(),
        targets=np.sort(np.concatenate((vi.numpy(),te.numpy()))),observed=observed,
        val_y=data.y[vi].numpy())
    return ev, rng


@torch.no_grad()
def prepare(data, config, cache, baseline=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    p = dict(config['baseline_parameters'] if baseline else config['model'])
    hops = p['lg_k'] if baseline else config['propagation_hops']
    alpha = p['alpha'] if baseline else config['propagation_alpha']
    norm_power = 0. if baseline else config['node_normalization_power']
    # The full finite PPR operator is preserved, rather than approximated by top-k.
    key=f"{config['dataset']}_k{hops}_a{alpha}_norm{norm_power}"
    path=Path(cache)/(key+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        saved=torch.load(path,map_location=device,weights_only=True)
    else:
        a=SimpleNamespace(**p)
        adjacency=ops.edge_index_to_sparse_mx(data.edge_index,len(data.y))
        _, adj=ops.process_adj(adjacency)
        g=ops.feature_propagation(adj,data.x,hops,alpha)
        eye=torch.eye(len(g),dtype=g.dtype,device=device)
        operator=eye
        adj=adj.to(device)
        for _ in range(hops):
            operator=(1-alpha)*torch.sparse.mm(adj,operator)+alpha*eye
        if norm_power:
            raw=data.x.to(device)
            normalized=raw/raw.norm(dim=1,keepdim=True).clamp_min(1e-12).pow(norm_power)
            gx=operator@normalized
            gx*=g.norm(dim=1).mean()/gx.norm(dim=1).mean().clamp_min(1e-12)
            g=gx
        a.max_n=torch.bincount(data.edge_index[0],minlength=len(g)).max()
        ni, unused, sim, nb=ops.pre_compute_local_sim_(g,data.edge_index,a,max_neighbor=a.max_n)
        del unused,eye
        saved=dict(features=g,operator=operator,node_ids=ni.to(device),similarities=sim.to(device),neighbors=nb.to(device))
        torch.save({k:v.cpu() for k,v in saved.items()},path)
        gc.collect()
    ctx=SimpleNamespace(ind_nid_m=saved['node_ids'],sim_score_m=saved['similarities'],
        nb_m=saved['neighbors'],num_classes=int(data.y.max())+1)
    return saved['features'],saved['operator'],SimpleNamespace(**p),ctx


def kernel_for(operator, ev, config):
    if config['kernel_normalization']=='torch':
        ids=torch.as_tensor(ev['targets'],device=operator.device)
        ti=torch.as_tensor(ev['train'],device=operator.device)
        h=operator[ids][:,ti].double().clamp_min(0)
        h/=h.sum(1).clamp_min(1e-12)[:,None]
        return h.cpu().numpy()
    raw=np.asarray(operator.cpu().numpy()[np.ix_(ev['targets'],ev['train'])],dtype=np.float64).clip(0)
    return raw/raw.sum(1).clip(1e-12)[:,None]
