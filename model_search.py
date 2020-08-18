import torch.nn as nn
from vanilla_model_cell import CellSearch as VanillaCell
from sigmoid_model_cell import CellSearch as SigmoidCell
from directional_model_cell import CellSearch as DirectionalCell
import torch.nn.functional as F
from genotypes import PRIMITIVES, STEPS, CONCAT, Genotype
import torch
from torch.autograd import Variable

INIT_RANGE=0.5


def get_rnn_cell(name):
    if name == "vanilla":
        return VanillaCell
    elif name == "sigmoid":
        return SigmoidCell
    elif name == "directional":
        return DirectionalCell
    else:
        raise NotImplementedError


class Network(nn.Module):
    def __init__(self, rnn_cell, init_channels, num_classes, layers, criterion, nhid):
        super(Network, self).__init__()
        self._init_channels = init_channels
        self._num_classes = num_classes
        self._nhid = nhid
        self._layers = layers
        self._criterion = criterion

        self.stem = nn.Sequential(
            nn.Conv2d(init_channels, nhid, 7, padding=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nhid),
            nn.Conv2d(nhid, nhid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nhid),
            nn.Conv2d(nhid, nhid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nhid),
        )

        self.cells = nn.ModuleList([])
        for i in range(layers):
            cell_class = get_rnn_cell(rnn_cell)
            self.cells += [cell_class(nhid, nhid//2, translation="patch_2x2")]
            self.cells += [cell_class(nhid, nhid//2, translation="patch_1x1")]

        self.classifier = nn.Linear(4*nhid, num_classes)
        self.classifier.weight.data.uniform_(-INIT_RANGE, INIT_RANGE)

        self._initialize_arch_parameters()

    def forward(self, x):
        out = self.stem(x)
        for i, rnn in enumerate(self.cells):
            if (i + 1) % 2 == 0:
                out = out.permute(0, 1, 3, 2)
                out = rnn(out)
                out = out.permute(0, 1, 3, 2)
            else:
                out = rnn(out)
        logits = self.classifier(out.contiguous().view(out.size(0), -1))
        return logits

    def _loss(self, x, target):
        return self._criterion(self(x), target)

    def _initialize_arch_parameters(self):
        k = sum(i for i in range(1, STEPS + 1))
        weights_data = torch.randn(k, len(PRIMITIVES)).mul_(1e-3)
        self.weights = Variable(weights_data.cuda(), requires_grad=True)
        self._arch_parameters = [self.weights]
        for rnn in self.cells:
            rnn.weights = self.weights

    def new(self):
        model_new = Network(self._init_channels, self._num_classes, self._layers, self._criterion,
                            self._nhid).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def arch_parameters(self):
        return self._arch_parameters

    def get_current_dist_weights(self):
        print(len(self.weights), F.softmax(self.weights, dim=-1).data.cpu().numpy())

    def genotype(self):
        def _parse(probs):
            gene = []
            start = 0
            for i in range(STEPS):
                end = start + i + 1
                W = probs[start:end].copy()
                j = sorted(range(i + 1),
                           key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((PRIMITIVES[k_best], j))
                start = end
            return gene

        gene = _parse(F.softmax(self.weights, dim=-1).data.cpu().numpy())
        genotype = Genotype(recurrent=gene, concat=range(STEPS + 1)[-CONCAT:])
        return genotype
