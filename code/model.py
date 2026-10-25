import torch
from torch import nn
from torch.nn import functional as F
class SinglePathMLP(nn.Module):
    """Same one-hidden-layer capacity as DnD's encoder/classifier."""

    def __init__(self, nfeat: int, hidden: int, nclass: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(nfeat, hidden)
        self.fc2 = nn.Linear(hidden, nclass)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


def state():
    return (torch.get_rng_state().clone(), [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore(saved):
    torch.set_rng_state(saved[0])
    if saved[1]:
        torch.cuda.set_rng_state_all(saved[1])


def views(model, graph, anchor):
    before = state()
    graph_logits = model(graph)
    after = state()
    restore(before)
    anchor_logits = model(anchor)
    replay = state()
    assert torch.equal(after[0], replay[0])
    assert len(after[1]) == len(replay[1])
    assert all(torch.equal(a,b) for a,b in zip(after[1],replay[1]))
    return graph_logits.log_softmax(-1), anchor_logits.log_softmax(-1)

