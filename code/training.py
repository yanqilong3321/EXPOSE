"""Training accepts noisy-source labels and validation labels, never test labels."""
import copy
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model import SinglePathMLP, views, restore
from .early_stop import EarlyStopping, Stop_args
from .dndnet import DnDNet
from . import official_ops as ops


def fit(g, ctx, ev, rng, config, out, method='expose'):
    tensor=lambda x:torch.as_tensor(x,device=g.device)
    ti,vi,target=[tensor(ev[k]) for k in ('train','validation','targets')]
    vy=tensor(ev['val_y'])
    ctx.unsup_idx=torch.cat((vi,tensor(ev['test_ids'])))
    p=config['baseline_parameters'] if method=='dnd' else config['model']
    from types import SimpleNamespace
    a=SimpleNamespace(**p)
    restore(rng)
    if method=='dnd':
        model=DnDNet(len(g),g.shape[1],a.hidden,ctx.num_classes,a.dropout).to(g.device)
    else:
        model=SinglePathMLP(g.shape[1],a.hidden,ctx.num_classes,a.dropout).to(g.device)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=a.wd)
    if method=='expose':
        for group in opt.param_groups:
            group['betas']=(config['adam_beta1'],config['adam_beta2']);group['eps']=config['adam_eps']
        ui=tensor(ev['targets'][ev['eligible']])
        source=tensor(ev['source_target']).to(g.dtype)
        bank=tensor(ev['bank'][ev['eligible']]).to(g.dtype)
        weights=tensor(ev['source_ce_weights']).to(g.dtype)
    else: observed=tensor(ev['observed'])
    stopper=EarlyStopping(model,**Stop_args(patience=100,max_epochs=200))
    selected_accuracy=-float('inf');selected_loss=float('inf');selected_state=None;selected_epoch=None
    history=[];started=time.time()
    for epoch in range(1,201):
        model.train()
        if method=='expose':
            lp,lp2=views(model,g,g)
            sup=(-(source*lp[ti]).sum(1)*weights).mean()
            weight=config['target_schedule'][epoch-1]
            extra=(-(bank*lp[ui]).sum(1)).mean() if weight>0 else lp.new_zeros(())
            npa=ops.neighbor_align(lp2,ctx.ind_nid_m,ctx.nb_m,ctx,a,ctx.unsup_idx)
            if not bool(torch.isfinite(npa)):npa=lp.new_zeros(())
            loss=sup+a.lambda_ncr*npa+weight*extra
        else:
            _,pure,_,_,noisy,_=model(g,a.noisy_rate)
            with torch.no_grad():uncertainty=ops.get_uncertainty(ctx,pure,a)
            loss=(1-a.ce_b)*F.nll_loss(pure[ti],observed)+a.ce_b*F.nll_loss(noisy[ti],observed)
            loss+=a.lambda_cr*ops.fix_cr(pure[ctx.unsup_idx],noisy[ctx.unsup_idx],name='ce',w=uncertainty[ctx.unsup_idx])
            loss+=a.lambda_ncr*ops.neighbor_align(pure,ctx.ind_nid_m,ctx.nb_m,ctx,a,ctx.unsup_idx)
        assert bool(torch.isfinite(loss))
        opt.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'] if method=='expose' else 1.)
        opt.step();model.eval()
        with torch.no_grad():
            if method=='expose':
                logits=model(g)[vi]
                output=logits.softmax(-1) if config['validation_softmax'] else logits
            else:
                _,pure,_,_,noisy,_=model(g,a.noisy_rate)
                output=a.beta*pure[vi]+(1-a.beta)*noisy[vi]
                logits=output
            acc=float((output.argmax(1)==vy).double().mean())
            val_loss=float(F.cross_entropy(logits,vy))
        history.append([epoch,float(loss),acc,float(norm),val_loss])
        if acc>selected_accuracy or (acc==selected_accuracy and val_loss<selected_loss):
            selected_accuracy=acc;selected_loss=val_loss;selected_epoch=epoch
            selected_state=copy.deepcopy(model.state_dict())
        if stopper.check([acc],epoch):break
    if method=='expose' and config['checkpoint_selector']=='accuracy_then_loss':
        chosen=selected_state;best=selected_epoch
    else:chosen=stopper.best_state;best=stopper.best_epoch
    model.load_state_dict(chosen);model.eval()
    with torch.no_grad():
        if method=='expose':prediction=model(g)
        else:
            _,pure,_,_,noisy,_=model(g,a.noisy_rate)
            prediction=a.beta*pure+(1-a.beta)*noisy
        prob=prediction[target].softmax(-1).cpu().numpy()
    np.savez_compressed(out/'predictions.npz',targets=ev['targets'],probabilities=prob,history=np.asarray(history))
    torch.save(model.state_dict(),out/'model.pt')
    return dict(best_epoch=best,best_validation=history[best-1][2],epochs_executed=len(history),
        elapsed_seconds=time.time()-started,test_labels_used_for_training=False,
        trusted_labels=0,teacher=False)
