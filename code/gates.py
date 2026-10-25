import torch
from torch.nn import functional as F
VARIANTS = ('baseline', 'evidence_guard', 'agreement_guard', 'strong_blend', 'agreement_strong')
@torch.no_grad()
def independent_support(attribute_evidence, observed, source_ids, targets, ppr_kernel, nclasses):
    """Expand target evidence back to distinct labeled sources, excluding self."""
    n = len(observed)
    attribute = torch.zeros((n, nclasses), dtype=torch.float64, device=observed.device)
    covered = torch.zeros(n, dtype=torch.bool, device=observed.device)
    effective = torch.zeros((n, n), dtype=torch.float64, device=observed.device)
    for j in range(n):
        q, valid, references, counts = attribute_evidence.excluding(observed, j, nclasses)
        attribute[j] = q[source_ids[j]]
        covered[j] = valid[source_ids[j]]
        weight = ppr_kernel[:, j] * valid[targets]
        mass = weight.sum()
        if mass <= attribute_evidence.epsilon:
            continue
        by_source = references[targets] / counts[observed].clamp_min(1)[None, :]
        by_source = by_source / by_source.sum(1).clamp_min(attribute_evidence.epsilon)[:, None]
        effective[j] = (weight / mass) @ by_source
        assert effective[j, j] == 0
    usable = effective.sum(1) > 1e-12
    ess = torch.where(usable, 1 / effective.square().sum(1).clamp_min(1e-12), 0)
    assert bool((effective >= 0).all())
    assert torch.allclose(effective.sum(1)[usable], torch.ones_like(ess[usable]), atol=1e-10)
    reconstructed = effective @ F.one_hot(observed, nclasses).to(effective.dtype)
    return dict(attribute_support=attribute, attribute_covered=covered, effective_sources=ess,
                effective_kernel=effective, reconstructed_support=reconstructed, usable=usable)


@torch.no_grad()
def variant_gate(observed, support, covered, attribute_support, attribute_covered,
                 effective_sources, original_risk, variant, risk_cap=.95, minimum_sources=4.):
    """Only observed train labels and label-free/source-excluded evidence enter."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    h = F.one_hot(observed, support.shape[1]).to(support.dtype)
    b = torch.where(covered[:, None], support, h)
    top, category = b.max(1)
    second = b.topk(2, dim=1).values[:, 1]
    strength = ((top - second) / (top + second).clamp_min(1e-12)).clamp(0, 1)
    attr_category = attribute_support.argmax(1)
    attr_own = attribute_support.gather(1, observed[:, None]).squeeze(1)
    attr_other = attribute_support.clone()
    attr_other.scatter_(1, observed[:, None], -torch.inf)
    agreement = covered & attribute_covered & (category == attr_category) & (category != observed)
    agreement &= attr_other.max(1).values > attr_own
    sufficient = (effective_sources / minimum_sources).clamp(0, 1) * strength
    boosted = risk_cap * torch.sqrt((original_risk / risk_cap).clamp(0, 1))
    if variant == "baseline":
        risk = original_risk.clone()
    elif variant == "evidence_guard":
        risk = original_risk * sufficient
    elif variant == "agreement_guard":
        risk = original_risk * agreement
    elif variant == "strong_blend":
        risk = boosted
    else:
        risk = torch.where(agreement, boosted, original_risk)
    risk = torch.where(covered, risk, 0).clamp(0, risk_cap)
    assert bool(torch.isfinite(risk).all())
    return dict(risk=risk, view_agreement=agreement, evidence_strength=strength, sufficiency=sufficient)

