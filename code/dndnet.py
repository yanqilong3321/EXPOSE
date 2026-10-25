"""DnD-Net architecture, preserved for the paired baseline.
Source: https://github.com/kaize0409/DND-NET. Original rights remain with the authors.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MLP_encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(MLP_encoder, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=True)

    def forward(self, x):
        x = torch.relu(self.Linear1(x))
        return x


class MLP_classifier(nn.Module):#
    def __init__(self, nfeat, nclass, dropout):
        super(MLP_classifier, self).__init__()
        self.Linear1 = Linear(nfeat, nclass, dropout, bias=True)

    def forward(self, x):
        out = self.Linear1(x)
        return torch.log_softmax(out, dim=1), out


class DnDNet(nn.Module):
    """
    The masked PPR
    """
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout=0.5, use_bn=False):
        super(DnDNet, self).__init__()

        self.encoder = MLP_encoder(nfeat=nfeat,
                                 nhid=nhid,
                                 dropout=dropout)

        self.classifier = MLP_classifier(nfeat=nhid,
                                         nclass=nclass,
                                         dropout=dropout)

        # adaptive noise to be added.
        self.noise= nn.Parameter(torch.randn(nnodes, nfeat))

        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nfeat)
            self.bn2 = nn.BatchNorm1d(nhid)

    def forward(self, g, noisy_rate=0.1, n_id=None):
        
        x_pure, y_pure, z_pure = self.forward_pure(g)
        noisy_g = self.adding_noise(g, noisy_rate=noisy_rate, n_id=n_id)
        x_noisy, y_noisy, z_noisy  = self.forward_noisy(noisy_g)

        return x_pure, y_pure, z_pure, x_noisy, y_noisy, z_noisy
    
    def adding_noise(self, g, noisy_rate, n_id=None):

        noisy_g = g.clone()
        if n_id is None:
            noisy_g += torch.sign(noisy_g) * F.normalize(self.noise) * noisy_rate
        else:
            noisy_g += torch.sign(noisy_g) * F.normalize(self.noise[n_id]) * noisy_rate
        return noisy_g

    def forward_pure(self, x):
        if self.use_bn:
            x = self.bn1(x)
        h = self.encoder(x)
        if self.use_bn:
            h = self.bn2(h)
        y, z = self.classifier(h)

        return h, y, z

    def forward_noisy(self, x):
        if self.use_bn:
            x = self.bn1(x)
        h = self.encoder(x)
        if self.use_bn:
            h = self.bn2(h)
        y, z = self.classifier(h)
        return h, y, z
