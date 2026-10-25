from dataclasses import dataclass
import torch
from torch.nn import functional as F
@dataclass
class AttributeEvidence:
    similarity: torch.Tensor
    ordering: torch.Tensor
    ordered_similarity: torch.Tensor
    source_ids: torch.Tensor
    neighbors: int
    epsilon: float

    @classmethod
    def build(cls, x, source_ids, neighbors=32, epsilon=1e-12):
        if len(source_ids) < 2:
            raise ValueError("At least two sources are needed")
        if not bool((source_ids[1:] > source_ids[:-1]).all()):
            raise ValueError("Sources must be sorted by node ID for stable ties")
        xn = F.normalize(x.to(torch.float64), dim=1, eps=epsilon)
        sim = (xn @ xn[source_ids].T).clamp(0, 1)
        order = torch.argsort(sim, dim=1, descending=True, stable=True)
        return cls(sim, order, sim.gather(1, order), source_ids, neighbors, epsilon)

    def excluding(self, observed, source_position, nclasses):
        """Exact top-k on L minus source, using cached label-free rankings."""
        keep = torch.arange(len(observed), device=observed.device) != source_position
        counts = torch.bincount(observed[keep], minlength=nclasses).to(torch.float64)
        # Stable full ranking, then delete source and nonpositive similarities.
        eligible = (self.ordering != source_position) & (self.ordered_similarity > 0)
        selected = eligible & (eligible.cumsum(1) <= self.neighbors)
        weights = torch.zeros_like(self.similarity)
        weights.scatter_(1, self.ordering, self.ordered_similarity * selected)
        onehot = F.one_hot(observed, nclasses).to(torch.float64)
        onehot[source_position] = 0
        votes = (weights @ onehot) / counts.clamp_min(1)[None, :]
        denominator = votes.sum(1)
        valid = denominator > self.epsilon
        q = torch.where(valid[:, None], votes / denominator.clamp_min(self.epsilon)[:, None], 0)
        assert bool((weights[:, source_position] == 0).all())
        return q, valid, weights, counts

