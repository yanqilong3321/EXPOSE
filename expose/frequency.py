import torch
from torch.nn import functional as F
class FrequencyEvidence:
    def __init__(self, attribute, power):
        assert 0 <= power <= 2
        self.attribute = attribute
        self.power = power
        self.epsilon = attribute.epsilon

    def excluding(self, observed, source_position, nclasses):
        original = self.attribute.excluding(observed, source_position, nclasses)
        if self.power == 1.:
            return original
        _, _, references, counts = original
        effective_counts = counts.clamp_min(1).pow(self.power)
        onehot = F.one_hot(observed, nclasses).double()
        onehot[source_position] = 0
        votes = (references @ onehot) / effective_counts[None, :]
        denominator = votes.sum(1)
        valid = denominator > self.epsilon
        q = torch.where(valid[:, None], votes / denominator.clamp_min(self.epsilon)[:, None], 0)
        assert bool((references[:, source_position] == 0).all())
        return q, valid, references, effective_counts

