import torch.nn as nn
import torch
from genotypes import PRIMITIVES, STEPS, CONCAT, Genotype
import torch.nn.functional as F
from torch.autograd import Variable

INIT_RANGE=0.04


class Cell(nn.Module):
    def __init__(self, ninp, nhid, translation, genotype):
        super(Cell, self).__init__()
        self.nhid = nhid
        self.genotype = genotype
        self.translation = translation

        # genotype is None when doing arch search
        steps = len(self.genotype.recurrent) if self.genotype is not None else STEPS
        input_multiplier = self._get_param_multiplier(name=translation)
        self._W0 = nn.Parameter(torch.Tensor((input_multiplier * ninp + nhid), 2 * nhid).uniform_(-INIT_RANGE, INIT_RANGE))
        self._Ws = nn.ParameterList([
            nn.Parameter(torch.Tensor(nhid, 2 * nhid).uniform_(-INIT_RANGE, INIT_RANGE)) for _ in range(steps)
        ])

        self.activations = {
            'tanh': F.tanh,
            'relu': F.relu,
            'sigmoid': F.sigmoid,
            'identity': lambda x: x
        }

        self.bn_in = nn.BatchNorm1d(input_multiplier * ninp + nhid)
        self.bn_in_rev = nn.BatchNorm1d(input_multiplier * ninp + nhid)
        self.bn = nn.ModuleList([nn.BatchNorm1d(2 * nhid) for _ in range(steps+1)])

    def forward(self, x):
        x_temporal = self._get_translation_temporal_dynamics(x, name=self.translation)
        _in = torch.cat(torch.split(x_temporal, 1, dim=1), dim=0)[:, 0]
        rev_idx = torch.arange(_in.size(1)-1, -1, -1).long().cuda()
        _in_rev = _in[:, rev_idx]
        temporal_input = torch.cat([_in, _in_rev], dim=2)
        B, T = _in.size(0), x_temporal.size(1)

        hidden = self.init_hidden(B)
        hidden = torch.cat([hidden, hidden], dim=-1)
        hiddens = [None] * T
        for t in range(T):
            hidden = self.cell(temporal_input[:, t, :], hidden)
            hiddens[t] = hidden
        hiddens = torch.stack(hiddens).permute(1, 0, 2)
        out = hiddens.contiguous().view(1, hiddens.shape[0], hiddens.shape[1], hiddens.shape[2])
        out = torch.cat(torch.split(out, x.shape[0], dim=1), dim=0).permute(1, 3, 0, 2)
        return out.contiguous()

    def _get_param_multiplier(self, name='patch_2x2'):
        if name == 'patch_2x2':
            multiplier = 4
        elif name == 'patch_1x1':
            multiplier = 1
        else:
            raise NotImplementedError
        return multiplier

    def _get_translation_temporal_dynamics(self, x, name='patch_2x2'):
        if name == "patch_2x2":
            out = x.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5).contiguous().view(
                x.shape[0], x.shape[2]//2, x.shape[3]//2, -1)
        elif name == "patch_1x1":
            out = x.unfold(2, 1, 1).unfold(3, 1, 1).permute(0, 2, 3, 1, 4, 5).contiguous().view(
                x.shape[0], x.shape[2], x.shape[3], -1)
        else:
            raise NotImplementedError
        return out

    def _get_translation_spatial(self, x, initial_shape, name='patch_2x2'):
        if name == 'patch_2x2':
            out = x.permute(1, 2, 0).unfold(2, initial_shape[2]//2, initial_shape[3]//2)
        elif name == 'patch_1x1':
            out = x.permute(1, 2, 0).unfold(2, initial_shape[2], initial_shape[3])
        else:
            raise NotImplementedError
        return out

    def _compute_init_state(self, x, h_prev):
        xh_prev = torch.cat([x[:, :x.shape[1]//2], h_prev[:, :h_prev.shape[1]//2]], dim=-1)
        xh_prev = self.bn_in(xh_prev)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev[:, :h_prev.shape[1]//2] + c0 * (h0 - h_prev[:, :h_prev.shape[1]//2])

        xh_prev_rev = torch.cat([x[:, x.shape[1] // 2:], h_prev[:, h_prev.shape[1] // 2:]], dim=-1)
        xh_prev_rev = self.bn_in_rev(xh_prev_rev)
        c0_rev, h0_rev = torch.split(xh_prev_rev.mm(self._W0), self.nhid, dim=-1)
        c0_rev = c0_rev.sigmoid()
        h0_rev = h0_rev.tanh()
        s0_rev = h_prev[:, h_prev.shape[1] // 2:] + c0_rev * (h0_rev - h_prev[:, h_prev.shape[1] // 2:])

        return torch.cat([s0, s0_rev], dim=-1)

    def cell(self, x, h_prev):
        s0 = self._compute_init_state(x, h_prev)
        s0 = self.bn[0](s0)

        states = [s0]
        for i, (name, pred) in enumerate(self.genotype.recurrent):
            s_prev = states[pred]

            ch = s_prev[:, :s_prev.shape[1]//2].contiguous().mm(self._Ws[i])
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()

            ch_rev = s_prev[:, s_prev.shape[1] // 2:].contiguous().mm(self._Ws[i])
            c_rev, h_rev = torch.split(ch_rev, self.nhid, dim=-1)
            c_rev = c_rev.sigmoid()

            s = torch.cat(
                [s_prev[:, :s_prev.shape[1]//2] + c * (
                            self.activations[name](h) - s_prev[:, :s_prev.shape[1]//2]),
                 s_prev[:, s_prev.shape[1]//2:] + c_rev * (
                             self.activations[name](h_rev) - s_prev[:, s_prev.shape[1]//2:])],
                dim=-1)
            s = self.bn[i+1](s)
            states += [s]
        output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)  # avg pooling

        return output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(bsz, self.nhid).zero_()).cuda()


class CellSearch(Cell):
    def __init__(self, ninp, nhid, translation="patch_2x2"):
        super(CellSearch, self).__init__(ninp, nhid, translation, genotype=None)
        # [3.1.2] "we enable batch normalization in each node to prevent gradient explosion during architecture search"
        self.bn = nn.BatchNorm1d(2*nhid, affine=False)

    def cell(self, x, h_prev):
        s0 = self._compute_init_state(x, h_prev)
        s0 = self.bn(s0)
        probs = F.softmax(self.weights, dim=-1)

        offset = 0
        states = s0.unsqueeze(0)
        for i in range(STEPS):
            ch = states[:, :, :states.shape[2]//2].contiguous().view(
                -1, self.nhid
            ).mm(self._Ws[i]).view(i + 1, -1, 2 * self.nhid)
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()

            ch_rev = states[:, :, states.shape[2] // 2:].contiguous().view(
                -1, self.nhid
            ).mm(self._Ws[i]).view(i + 1, -1, 2 * self.nhid)

            c_rev, h_rev = torch.split(ch_rev, self.nhid, dim=-1)
            c_rev = c_rev.sigmoid()

            s = torch.zeros_like(s0)
            for k, name in enumerate(PRIMITIVES):
                if name == 'none':
                    continue
                unweighted = torch.cat(
                    [states[:, :, :states.shape[2]//2] + c * (self.activations[name](h) - states[:, :, :states.shape[2]//2]),
                     states[:, :, states.shape[2]//2:] + c_rev * (self.activations[name](h_rev) - states[:, :, states.shape[2]//2:])],
                    dim=-1)
                s += torch.sum(probs[offset:offset + i + 1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
            s = self.bn(s)
            states = torch.cat([states, s.unsqueeze(0)], 0)
            offset += i + 1
        output = torch.mean(states[-CONCAT:], dim=0)
        return output


