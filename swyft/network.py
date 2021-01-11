# pylint: disable=no-member, not-callable, access-member-before-definition
import math

import torch
import torch.nn as nn

from collections import defaultdict


def combine(y, z):
    """Combines data vectors y and parameter vectors z.

    z : (..., pnum, pdim)
    y : (..., ydim)

    returns: (..., pnum, ydim + pdim)

    """
    y = y.unsqueeze(-2)  # (..., 1, ydim)
    y = y.expand(*z.shape[:-1], *y.shape[-1:])  # (..., pnum, ydim)
    return torch.cat([y, z], -1)


class OnlineNormalizationLayer(nn.Module):
    def __init__(self, shape, stable: bool = False, epsilon: float = 1e-10):
        """Accumulate mean and variance online using the "parallel algorithm" algorithm from [1].

        Args:
            shape (tuple): shape of mean, variance, and std array. do not include batch dimension!
            stable (bool): (optional) compute using the stable version of the algorithm [1]
            epsilon (float): (optional) added to the computation of the standard deviation for numerical stability.

        References:
            [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        super().__init__()
        self.register_buffer("n", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_mean", torch.zeros(shape))
        self.register_buffer("_M2", torch.zeros(shape))
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.shape = shape
        self.stable = stable

    def _parallel_algorithm(self, x):
        assert x.shape[1:] == self.shape
        na = self.n.clone()
        nb = x.shape[0]
        nab = na + nb

        xa = self._mean.clone()
        xb = x.mean(dim=0)
        delta = xb - xa
        if self.stable:
            xab = (na * xa + nb * xb) / nab
        else:
            xab = xa + delta * nb / nab

        m2a = self._M2.clone()
        m2b = (
            x.var(dim=(0,), unbiased=False) * nb
        )  # do not use bessel's correction then multiply by total number of items in batch.
        m2ab = m2a + m2b + delta ** 2 * na * nb / nab
        return nab, xab, m2ab

    def forward(self, x):
        if self.training:
            self.n, self._mean, self._M2 = self._parallel_algorithm(x)
        return (x - self.mean) / self.std

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        if self.n > 1:
            return self._M2 / (self.n - 1)
        else:
            return torch.zeros_like(self._M2)

    @property
    def std(self):
        return torch.sqrt(self.var + self.epsilon)


# From: https://github.com/pytorch/pytorch/issues/36591
class LinearWithChannel(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()

        # initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, output_size, input_size))
        self.b = torch.nn.Parameter(torch.zeros(channel_size, output_size))

        # change weights to kaiming
        self.reset_parameters(self.w, self.b)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.matmul(self.w, x).squeeze(-1) + self.b


class DenseLegs(nn.Module):
    def __init__(self, ydim, pnum, pdim, p=0.0, NH=256):
        super().__init__()
        self.fcA = LinearWithChannel(ydim, NH, pnum)
        self.fcB = LinearWithChannel(NH, NH, pnum)
        self.fcC = LinearWithChannel(NH, pdim*2, pnum)
        self.fc1 = LinearWithChannel(pdim*2 + pdim, NH, pnum)
        self.fc2 = LinearWithChannel(NH, NH, pnum)
        self.fc3 = LinearWithChannel(NH, 1, pnum)
        self.drop = nn.Dropout(p=p)

        self.af = torch.relu

        # swish activation function for smooth posteriors
        self.af2 = lambda x: x * torch.sigmoid(x * 10.0)

        self.pnum = pnum

    def forward(self, y, z):
        # Defining test statistic
        #print(y.shape)
        y = y.unsqueeze(-2).repeat(1, self.pnum, 1)
        #print(y.shape)
        y = self.af(self.fcA(y))
        y = self.drop(y)
        y = self.af(self.fcB(y))
        y = self.drop(y)
        y = self.fcC(y)

        # Combination
        #x = combine(y, z)

        y = y.expand(z.shape[0], -1, -1)
        x = torch.cat([y, z], -1)
        #print(x.shape)

        x = self.af(self.fc1(x))
        x = self.drop(x)
        x = self.af(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x).squeeze(-1)
        return x


class DenseLegsOld(nn.Module):
    def __init__(self, ydim, pnum, pdim, p=0.0, NH=256):
        super().__init__()
        self.fc1 = LinearWithChannel(ydim + pdim, NH, pnum)
        self.fc2 = LinearWithChannel(NH, NH, pnum)
        self.fc3 = LinearWithChannel(NH, NH, pnum)
        self.fc4 = LinearWithChannel(NH, 1, pnum)
        self.drop = nn.Dropout(p=p)

        self.af = torch.relu

        # swish activation function for smooth posteriors
        self.af2 = lambda x: x * torch.sigmoid(x * 10.0)

    def forward(self, y, z):
        x = combine(y, z)
        x = self.af(self.fc1(x))
        x = self.drop(x)
        x = self.af(self.fc2(x))
        x = self.drop(x)
        x = self.af(self.fc3(x))
        x = self.fc4(x).squeeze(-1)
        return x


#    @staticmethod
#    def _get_configuration(combinations):
#        "Return dict (posterior dim) -> (number of cases)."
#        result = defaultdict(int)
#        for c in combinations:
#            result[len(c)] += 1
#        return result
#    
#class NetworkNew(nn.Module):
#    def __init__(self, n_features, pnum, pdim, head=None, datanorms=None, tail = DenseLegs):
#        """Base network combining z-independent head and parallel tail.
#
#        :param ydim: Number of data dimensions going into DenseLeg network
#        :param pnum: Number of posteriors to estimate
#        :param pdim: Dimensionality of posteriors
#        :param head: Head network, z-independent
#        :type head: `torch.nn.Module`, optional
#
#        The forward method of the `head` network takes data `x` as input, and
#        returns intermediate state `y`.
#        """
#        super().__init__()
#        self.n_features = n_features
#        self.pnum = pnum
#        self.pdim = pdim
#
#        self.head = head
#        self.tail = tail(n_features, pnum, pdim)
#
#
#    def _combine_z(self, z):
#        "Return dict (posterior dim) -> (list of parameter arrays)."
#        result = defaultdict(list)
#        for c in self.combinations:
#            pars = torch.stack([z[k] for k in c]).T
#            result[len(c)].append(pars)
#        return result
#
#    def forward(self, x, z):
#        f = self.head(x)
#        out = tail(f, z) self.tails(f, z)
#        return out

class Network(nn.Module):
    def __init__(self, ydim, pnum, pdim, head=None, tail = DenseLegs):
        """Base network combining z-independent head and parallel tail.

        :param ydim: Number of data dimensions going into DenseLeg network
        :param pnum: Number of posteriors to estimate
        :param pdim: Dimensionality of posteriors
        :param head: Head network, z-independent
        :type head: `torch.nn.Module`, optional

        The forward method of the `head` network takes data `x` as input, and
        returns intermediate state `y`.
        """
        super().__init__()
        self.head = head
        self.legs = tail(ydim, pnum, pdim)

#        # Set datascaling
#        if datanorms is None:
#            datanorms = [
#                torch.tensor(0.0),
#                torch.tensor(1.0),
#                torch.tensor(0.5),
#                torch.tensor(0.5),
#            ]
#        self._set_datanorms(*datanorms)
#
#    def _set_datanorms(self, x_mean, x_std, z_mean, z_std):
#        self.x_loc = torch.nn.Parameter(x_mean)
#        self.x_scale = torch.nn.Parameter(x_std)
#        self.z_loc = torch.nn.Parameter(z_mean)
#        self.z_scale = torch.nn.Parameter(z_std)

    def forward(self, x, z):
        #x = (x - self.x_loc) / self.x_scale
        #z = (z - self.z_loc) / self.z_scale

        #if self.head is not None:
        y = self.head(x)
        #else:
        #    y = x  # Use 1-dim data vector as features

        out = self.legs(y, z)
        return out


class DefaultHead(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        f = []
        for key, value in sorted(x.items()):
            f.append(value)
        f = torch.cat(f)
        return f



if __name__ == "__main__":
    pass
