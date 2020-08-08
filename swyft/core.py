# pylint: disable=no-member, not-callable
from typing import Callable
from copy import deepcopy
from warnings import warn
from contextlib import nullcontext

from collections import defaultdict
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


#######################
# Convenience functions
#######################

def sortbyfirst(x, y):
    """Sort two lists by values of first list."""
    i = np.argsort(x)
    return x[i], y[i]

def subsample(n_sub, z, replace = False):
    """Subsample lists."""
    if n_sub is None:
        return z
    if n_sub >= len(z) and not replace:
        raise ValueError("Number of sub-samples without replacement larger than sample size")
    indices = np.random.choice(len(z), size = n_sub, replace = replace)
    z_sub = [z[i] for i in indices]
    return z_sub

def combine_z(z, combinations):
    """Generate parameter combinations in last dimension. 
    Requires: z.ndim == 1. 
    output.shape == (n_posteriors, parameter shape)
    """
    if combinations is None:
        return z.unsqueeze(-1)
    else:
        return torch.stack([z[c] for c in combinations])

def set_device(gpu: bool = False) -> torch.device:
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    elif gpu and not torch.cuda.is_available():
        warn("Although the gpu flag was true, the gpu is not avaliable.")
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    return device

#########################
# Generate sample batches
#########################

def sample_hypercube(num_samples: int, num_params: int) -> Tensor:
    """Return uniform samples from the hyper cube.

    Args:
        num_samples (int): number of samples.
        num_params (int): dimension of hypercube.

    Returns:
        Tensor: random samples.
    """
    return torch.rand(num_samples, num_params)

def simulate_xz(model, list_z):
    """Generates x ~ model(z).
    
    Args:
        model (fn): foreward model, returns samples x~p(x|z).
            Both x and z have to be Tensors.
        list_z (list of Tensors): list of model parameters z.

    Returns:
        list of dict: list of dictionaries with 'x' and 'z' pairs.
    """
    list_xz = []
    for z in list_z:
        x = model(z)
        list_xz.append(dict(x=x, z=z))
    return list_xz

def get_x(list_xz):
    """Extract x from batch of samples."""
    return [xz['x'] for xz in list_xz]

def get_z(list_xz):
    """Extract z from batch of samples."""
    return [xz['z'] for xz in list_xz]


##########
# Training
##########

def loss_fn(network, xz, combinations = None):
    """Evaluate binary-cross-entropy loss function. Mean over batch.

    Args:
        network (nn.Module): network taking minibatch of samples and returing ratio estimator.
        xz (dict): batch of samples to train on.
        combinations (list, optional): determines posteriors that are generated.
            examples:
                [[0,1], [3,4]]: p(z_0,z_1) and p(z_3,z_4) are generated
                    initialize network with zdim = 2, pdim = 2
                [[0,1,5,2]]: p(z_0,z_1,z_5,z_2) is generated
                    initialize network with zdim = 1, pdim = 4

    Returns:
        Tensor: training loss.
    """ #TODO does the loss function depend on which distribution the z was drawn from? it does in SBI for the SNPE versions
    assert xz['x'].size(0) == xz['z'].size(0), "Number of x and z must be equal."
    assert xz['x'].size(0) % 2 == 0, "There must be an even number of samples in the batch for contrastive learning."
    n_batch = xz['x'].size(0)

    # Is it the removal of replacement that made it stop working?!

    # bring x into shape
    # (n_batch*2, data-shape)  - repeat twice each sample of x - there are n_batch samples
    # repetition pattern in first dimension is: [a, a, b, b, c, c, d, d, ...]
    x = xz['x']
    x = torch.repeat_interleave(x, 2, dim = 0)

    # bring z into shape
    # (n_batch*2, param-shape)  - repeat twice each sample of z - there are n_batch samples
    # repetition is alternating in first dimension: [a, b, a, b, c, d, c, d, ...]
    z = xz['z']
    z = torch.stack([combine_z(zs, combinations) for zs in z])
    zdim = len(z[0])
    z = z.view(n_batch // 2, -1, *z.shape[-1:])
    z = torch.repeat_interleave(z, 2, dim = 0)
    z = z.view(n_batch*2, -1, *z.shape[-1:])
    
    # call network
    lnL = network(x, z)
    lnL = lnL.view(n_batch // 2, 4, zdim)

    # Evaluate cross-entropy loss
    # loss = 
    # -ln( exp(lnL(x_a, z_a))/(1+exp(lnL(x_a, z_a))) )
    # -ln( exp(lnL(x_b, z_b))/(1+exp(lnL(x_b, z_b))) )
    # -ln( 1/(1+exp(lnL(x_a, z_b))) )
    # -ln( 1/(1+exp(lnL(x_b, z_a))) )
    loss  = -torch.nn.functional.logsigmoid( lnL[:,0])
    loss += -torch.nn.functional.logsigmoid(-lnL[:,1])
    loss += -torch.nn.functional.logsigmoid(-lnL[:,2])
    loss += -torch.nn.functional.logsigmoid( lnL[:,3])
    loss = loss.sum() / (n_batch // 2)

    return loss

# We have the posterior exactly because our proir is known and flat. Flip bayes theorem, we have the likelihood ratio.
# Consider that the variance of the loss from different legs causes some losses to have high coefficients in front of them.
def train(
    network, 
    train_loader,
    validation_loader,
    early_stopping_patience,
    max_epochs = None,
    lr = 1e-3,
    combinations = None,
    device=None,
    non_blocking=True
):
    """Network training loop.

    Args:
        network (nn.Module): network for ratio estimation.
        train_loader (DataLoader): DataLoader of samples.
        validation_loader (DataLoader): DataLoader of samples.
        max_epochs (int): Number of epochs.
        lr (float): learning rate.
        combinations (list, optional): determines posteriors that are generated.
            examples:
                [[0,1], [3,4]]: p(z_0,z_1) and p(z_3,z_4) are generated
                    initialize network with zdim = 2, pdim = 2
                [[0,1,5,2]]: p(z_0,z_1,z_5,z_2) is generated
                    initialize network with zdim = 1, pdim = 4
        device (str, device): Move batches to this device.
        non_blocking (bool): non_blocking in .to(device) expression.

    Returns:
        list: list of training losses.
    """
    # TODO consider that the user might want other training stats, like number of correct samples for example
    def do_epoch(loader: torch.utils.data.dataloader.DataLoader, train: bool):
        accumulated_loss = 0
        training_context = nullcontext() if train else torch.no_grad()
        with training_context:
            for batch in loader:
                optimizer.zero_grad()
                if device is not None:
                    batch = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
                loss = loss_fn(network, batch, combinations = combinations)
                if train:
                    loss.backward()
                    optimizer.step()
                accumulated_loss += loss.detach().cpu().numpy().item()
        return accumulated_loss

    max_epochs =  2 ** 31 - 1 if max_epochs is None else max_epochs
    optimizer = torch.optim.Adam(network.parameters(), lr = lr)

    n_train_batches = len(train_loader)
    n_validation_batches = len(validation_loader)
    
    train_losses, validation_losses = [], []
    epoch, fruitless_epoch, min_loss = 0, 0, float("Inf")
    while epoch <= max_epochs and fruitless_epoch < early_stopping_patience:
        network.train()
        train_loss = do_epoch(train_loader, True)
        train_losses.append(train_loss / n_train_batches)
        
        network.eval()
        validation_loss = do_epoch(validation_loader, False)
        validation_losses.append(validation_loss / n_validation_batches)

        epoch += 1
        if epoch == 0 or min_loss > validation_loss:
            fruitless_epoch = 0
            min_loss = validation_loss
            best_state_dict = deepcopy(network.state_dict())
        else:
            fruitless_epoch += 1

    return train_losses, validation_losses, best_state_dict


######################
# Posterior estimation
######################

def estimate_lnL(network, x0, list_z, normalize = True, combinations = None, n_batch = 64):
    """Return current estimate of normalized marginal 1-dim lnL.

    Args:
        network (nn.Module): trained ratio estimation network.
        x0 (torch.tensor): data.
        list_z (list): list of parameter points to evaluate.
        normalize (bool): set max(lnL) = 0.
        combinations (list, optional): parameter combinations.
        n_batch (int): minibatch size.

    Returns:
        list: List of dictionaries with component z and lnL
    """
    x0 = x0.unsqueeze(0)
    zdim = len(list_z[0]) if combinations is None else len(combinations)
    n_samples = len(list_z)

    lnL_out = []
    z_out = []
    for i in range(n_samples//n_batch+1):
        zbatch = list_z[i*n_batch:(i+1)*n_batch]
        zcomb = torch.stack([combine_z(zn, combinations) for zn in zbatch])  # TODO this raises an error when the n_samples % n_batch == 0
        tmp = network(x0, zcomb)
        lnL_out += list(tmp.detach().cpu())
        z_out += list(zcomb.cpu())

    out = []
    for i in range(zdim):
        z_i = torch.stack([z_out[j][i] for j in range(n_samples)])
        lnL_i= torch.stack([lnL_out[j][i] for j in range(n_samples)])
        if normalize:
            lnL_i -= lnL_i.max()
        out.append(dict(z=z_i, lnL=lnL_i))
    return out


##########
# Networks
##########

# From: https://github.com/pytorch/pytorch/issues/36591
class LinearWithChannel(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()

        #initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, output_size, input_size))
        self.b = torch.nn.Parameter(torch.zeros(channel_size, output_size))

        #change weights to kaiming
        self.reset_parameters(self.w, self.b)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.matmul(self.w, x).squeeze(-1) + self.b
    
def combine(y, z):
    """Combines data vectors y and parameter vectors z.
    
    z : (..., zdim, pdim)
    y : (..., ydim)
    
    returns: (..., zdim, ydim + pdim)
    
    """
    y = y.unsqueeze(-2) # (..., 1, ydim)
    y = y.expand(*z.shape[:-1], *y.shape[-1:]) # (..., zdim, ydim)
    return torch.cat([y, z], -1)

class DenseLegs(nn.Module):
    def __init__(self, ydim, zdim, pdim = 1, p = 0.0, NH = 256):
        super().__init__()
        self.fc1 = LinearWithChannel(ydim+pdim, NH, zdim)
        self.fc2 = LinearWithChannel(NH, NH, zdim)
        self.fc3 = LinearWithChannel(NH, NH, zdim)
        self.fc4 = LinearWithChannel(NH, 1, zdim)
        self.drop = nn.Dropout(p = p)

    def forward(self, y, z):
        x = combine(y, z)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x).squeeze(-1)
        return x

def get_norms(xz):
    x = get_x(xz)
    z = get_z(xz)
    x_mean = sum(x)/len(x)
    z_mean = sum(z)/len(z)
    x_var = sum([(x[i]-x_mean)**2 for i in range(len(x))])/len(x)
    z_var = sum([(z[i]-z_mean)**2 for i in range(len(z))])/len(z)
    return x_mean, x_var**0.5, z_mean, z_var**0.5

class Network(nn.Module):
    def __init__(self, xdim, zdim, pdim = 1, xz_init = None, head = None, p = 0.):
        """Base network combining z-independent head and parallel tail.

        :param xdim: Number of data dimensions going into DenseLeg network
        :param zdim: Number of latent space variables
        :param xz_init: xz Samples used for normalization
        :param head: Head network, z-independent
        :type head: `torch.nn.Module`, optional

        The forward method of the `head` network takes data `x` as input, and
        returns intermediate state `y`.
        """
        super().__init__()
        self.head = head
        self.legs = DenseLegs(xdim, zdim, pdim = pdim, p = p)
        
        if xz_init is not None:
            x_mean, x_std, z_mean, z_std = get_norms(xz_init)
        else:
            x_mean, x_std, z_mean, z_std = 0., 1., 0., 1.
        self.x_mean = torch.nn.Parameter(torch.tensor(x_mean).float())
        self.z_mean = torch.nn.Parameter(torch.tensor(z_mean).float())
        self.x_std = torch.nn.Parameter(torch.tensor(x_std).float())
        self.z_std = torch.nn.Parameter(torch.tensor(z_std).float())
    
    def forward(self, x, z):
        #TODO : Bring normalization back
        #x = (x-self.x_mean)/self.x_std
        #z = (z-self.z_mean)/self.z_std

        if self.head is not None:
            y = self.head(x)
        else:
            y = x  # Take data as features

        out = self.legs(y, z)
        return out

def iter_sample_z(n_draws, n_dim, net, x0, verbosity = False, threshold = 1e-6):
    """Generate parameter samples z~p_c(z) from constrained prior.
    
    Arguments
    ---------
    n_draws: Number of draws
    n_dim: Number of dimensions of z
    net: Trained density network
    x0: Reference data
    
    Returns
    -------
    z: list of n_dim samples with length n_draws
    """
    done = False
    zout = defaultdict(lambda: [])
    counter = np.zeros(n_dim)
    frac = np.ones(n_dim)
    while not done:
        z = torch.rand(n_draws, n_dim, device=x0.device)
        zlnL = estimate_lnL(net, x0, z)
        for i in range(n_dim):
            mask = zlnL[i]['lnL'] > np.log(threshold)
            frac[i] = sum(mask)/len(mask)
            zout[i].append(zlnL[i]['z'][mask])
            counter[i] += mask.sum()
        done = min(counter) >= n_draws
    if verbosity:
        print("Constrained posterior volume:", frac.prod())
    
    #out = list(torch.tensor([np.concatenate(zout[i])[:n_draws] for i in range(n_dim)]).T[0])
    out = list(torch.stack([torch.cat(zout[i]).squeeze(-1)[:n_draws] for i in range(n_dim)]).T)
    return out


if __name__ == "__main__":
    pass
