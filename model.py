import torch.nn as nn
from vanilla_model_cell import Cell as VanillaCell
from sigmoid_model_cell import Cell as SigmoidCell
from directional_model_cell import Cell as DirectionalCell

INIT_RANGE = 0.5


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
    def __init__(self, rnn_cell, init_channels, num_classes, layers, nhid, genotype, ds="cifar"):
        super(Network, self).__init__()
        self._init_channels = init_channels
        self._num_classes = num_classes
        self._nhid = nhid
        self._layers = layers

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
            self.cells += [cell_class(nhid, nhid // 2, translation="patch_2x2", genotype=genotype)]
            self.cells += [cell_class(nhid, nhid // 2, translation="patch_1x1", genotype=genotype)]

        lin_layer = nn.Linear(4 * nhid, num_classes)
        lin_init_range = INIT_RANGE if (ds == "cifar") else INIT_RANGE/10
        lin_layer.weight.data.uniform_(-lin_init_range, lin_init_range)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(4 * nhid),
            lin_layer
        )

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
