"""Structure-aware evidence. Inputs contain no clean/validation/test labels."""
import numpy as np
import torch
from torch.nn import functional as F
from .evidence_base import AttributeEvidence
from .frequency import FrequencyEvidence
from .gates import independent_support, variant_gate
from .source_weights import reliable_weights, class_unit_mean


@torch.no_grad()
def build_evidence(raw, train, observed, targets, kernel, classes, config):
    p = config['evidence']
    order = np.argsort(train, kind='stable')
    inverse = np.argsort(order)
    device = raw.device
    ti = torch.as_tensor(train[order], device=device)
    y = torch.as_tensor(observed[order], device=device)
    ids = torch.as_tensor(targets, device=device)
    h = torch.as_tensor(kernel[:, order], device=device)
    attribute = AttributeEvidence.build(raw, ti, p['neighbors'])
    attribute.similarity = attribute.similarity.pow(p['similarity_power'])
    attribute.ordered_similarity = attribute.ordered_similarity.pow(p['similarity_power'])
    attr = FrequencyEvidence(attribute, p['frequency_power'])
    aux = independent_support(attr, y, ti, ids, h, classes)
    support, covered = aux['reconstructed_support'], aux['usable']
    onehot = F.one_hot(y, classes).double()
    b = torch.where(covered[:, None], support, onehot)
    own = b.gather(1, y[:, None]).squeeze(1)
    other = b.clone()
    other.scatter_(1, y[:, None], -torch.inf)
    alt = other.max(1).values
    r0 = ((alt-own).clamp_min(0)/(alt+own).clamp_min(1e-12)).clamp_max(.95)
    r0 = torch.where(covered, r0, 0.)
    gate = variant_gate(y, support, covered, aux['attribute_support'],
        aux['attribute_covered'], aux['effective_sources'], r0, p['gate'], .95, 4.)
    risk = gate['risk'].cpu().numpy()[inverse]
    risk = np.minimum(.95, risk*p['risk_scale'])
    # A source cannot vote for itself, including in class-frequency counts.
    for j in (0, len(y)-1):
        before, valid, _, _ = attr.excluding(y, j, classes)
        altered = y.clone(); altered[j] = (altered[j]+1)%classes
        after, valid2, _, _ = attr.excluding(altered, j, classes)
        assert torch.equal(before, after) and torch.equal(valid, valid2)
    return risk, b.cpu().numpy()[inverse]


def source_weights(risk, observed, config):
    p = config['source_weighting']
    power, normalization = p['power'], p['normalization']
    if config['weight_implementation'] == 'legacy':
        if normalization == 'none': return np.ones(len(risk), dtype=np.float64)
        value = reliable_weights(risk, power)
        return class_unit_mean(value, observed) if normalization == 'class' else value
    value = np.clip(1-risk, .05, 1)**power
    if normalization == 'class':
        for label in np.unique(observed):
            mask = observed == label
            value[mask] /= value[mask].mean()
    return value/value.mean()


def supervision(kernel, observed, risk, classes, config):
    source = np.eye(classes, dtype=np.float64)[observed]
    weighted = kernel*np.clip(1-risk, .05, 1)[None, :]**config['aggregation_power']
    weighted /= weighted.sum(1).clip(1e-12)[:, None]
    bank = weighted@source
    return dict(risk=risk, source_target=source, bank=bank,
        eligible=kernel.sum(1)>1e-12, source_ce_weights=source_weights(risk, observed, config))
