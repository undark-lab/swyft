# pylint: disable=no-member, not-callable
from typing import Callable
from copy import deepcopy
from warnings import warn
from contextlib import nullcontext

from collections import defaultdict
import math
import zarr
import numcodecs

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import cumtrapz
from torch import Tensor
from tqdm import tqdm

#######################
# Convenience functions
#######################

def comb2d(indices):
    output = []
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            output.append([indices[i], indices[j]])
    return output

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

def get_x(list_xz):
    """Extract x from batch of samples."""
    return [xz['x'] for xz in list_xz]

def get_z(list_xz):
    """Extract z from batch of samples."""
    return [xz['z'] for xz in list_xz]


###################################
# New data model based on datastore
###################################

class DataContainer(torch.utils.data.Dataset):
    """Simple data container class.

    Note: The noisemodel allows scheduled noise level increase during training.
    """
    def __init__(self, datastore, indices, noisemodel = None):
        super().__init__()
        # Check whether datastore is complete
        if datastore.requires_sim():
            raise RuntimeError("Datastore entries missing. Run simulator.")

        self.ds = datastore
        self.indices = indices
        self.noisemodel = noisemodel

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Obtain x, z
        i = self.indices[idx]
        x = self.ds.x[i]
        z = self.ds.z[i]
        
        # Add optional noise
        if self.noisemodel is not None:
            x = self.noisemodel(x, z)

        # Tensors
        x = torch.tensor(x).float()
        z = torch.tensor(z).float()
        
        # Done
        xz = dict(x=x, z=z)
        return xz


#################
# Datastore class
#################

class DataStore:
    """The iP3 datastore.

    Args:
        filename (str): Optional. If given, defines path of `zarr.DirectoryStore`.
    """
    def __init__(self, filename = None):
        # Open (new) datastore
        if filename is None:
            self.store = zarr.MemoryStore()
        else:
            self.store = zarr.DirectoryStore(filename)
        self.root = zarr.group(store = self.store)
        
        if 'samples' not in self.root.keys():
            print("Creating empty datastore:", filename)
            print("...don't forget to run `init` to set up storage parameters.")
            return
        
        print("Loading datastore:", filename)
        self._update()

    def _update(self):
        self.x = self.root['samples/x']
        self.z = self.root['samples/z']
        self.m = self.root['metadata/needs_sim']
        self.u = self.root['metadata/intensity']
        
    def init(self, xdim, zdim):
        """Initialize data store content dimensions.

        Args:
            zdim (int): Number of z dimensions
            xdim (tuple): Shape of x array
        """
        if 'samples' in self.root.keys():
            print("WARNING: Datastore is already initialized.")
            return self
        self.x = self.root.zeros('samples/x', shape=(0,)+xdim, chunks=(1,)+xdim, dtype='f4')
        self.z = self.root.zeros('samples/z', shape=(0,)+(zdim,), chunks=(10000,)+(zdim,), dtype='f4')
        self.m = self.root.zeros('metadata/needs_sim', shape=(0,1), chunks=(10000,)+(1,), dtype='bool')
        self.u = self.root.create('metadata/intensity', shape=(0,), dtype=object, object_codec=numcodecs.Pickle())
        print("Datastore initialized.")

        return self
        
    def _append_z(self, z):
        """Append z to datastore content and new slots for x."""
        self._update()

        # Add simulation slots
        xshape = list(self.x.shape)
        xshape[0] += len(z)
        self.x.resize(*xshape)
        
        # Add z samples
        self.z.append(z)
        
        # Register as missing
        m = np.ones((len(z),1), dtype='bool')
        self.m.append(m)
        
    def __len__(self):
        """Returns number of samples in the datastore."""
        self._update()

        return len(self.z)
        
    def intensity(self, zlist):
        """Evaluate DataStore intensity function.

        Args:
            z (array-like): list of parameter values.
        """
        self._update()

        if len(self.u) == 0:
            return np.zeros(len(zlist))
        else:
            return np.array([self.u[i](zlist) for i in range(len(self.u))]).max(axis=0)
        
    def _grow(self, p):
        """Grow number of samples in datastore."""
        # Proposed new samples z from p
        z_prop = p.sample()
        
        # Rejection sampling from proposal list
        accepted = []
        ds_intensities = self.intensity(z_prop)
        target_intensities = p(z_prop)
        for z, Ids, It in zip(z_prop, ds_intensities, target_intensities):
            rej_prob = np.minimum(1, Ids/It)
            w = np.random.rand(1)[0]
            accepted.append(rej_prob < w)
        z_accepted = z_prop[accepted, :]
        
        # Add new entries to datastore and update intensity function
        self._append_z(z_accepted)
        if len(z_accepted) > 0:
            self.u.resize(len(self.u)+1)
            self.u[-1] = p
            print("Adding %i new samples. Run simulator!"%len(z_accepted))
        else:
            print("No new simulator runs required.")

    def sample(self, p):
        """Sample from DataStore.

        Args:
            p (intensity function): Target intensity function.
        """
        self._update()

        self._grow(p)
        
        accepted = []
        zlist = self.z[:]
        I_ds = self.intensity(zlist)
        I_target = p(zlist)
        for i, z in enumerate(zlist):
            accept_prob  = I_target[i]/I_ds[i]
            if accept_prob > 1.:
                raise ValueError(accept_prob, I_target[i], I_ds[i], i)
            w = np.random.rand(1)[0]
            if accept_prob > w:
                accepted.append(i)
        return accepted
    
    def __getitem__(self, i):
        self._update()

        return self.x[i], self.z[i]
                
    def _require_sim_idx(self):
        indices = []
        m = self.m[:]
        for i in range(len(self.z)):
            if m[i]:
                indices.append(i)
        return indices

    def requires_sim(self):
        self._update()

        return len(self._require_sim_idx()) > 0
    
    def _add_sim(self, i, x):
        self.x[i] = x
        self.m[i] = False

    def simulate(self, simulator):
        """Run simulator sequentially for missing points.

        Args:
            simulator (callable): Simulator
        """
        self._update()

        idx = self._require_sim_idx()
        if len(idx) == 0:
            print("No simulations required.")
            return
        for i in tqdm(idx, desc='Simulate'):
            z = self.z[i]
            x = simulator(z)
            self._add_sim(i, x) 


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
    while epoch < max_epochs and fruitless_epoch < early_stopping_patience:
        #print("Epoch:", epoch, end = "\r")
        network.train()
        train_loss = do_epoch(train_loader, True)
        train_losses.append(train_loss / n_train_batches)
        
        network.eval()
        validation_loss = do_epoch(validation_loader, False)
        #print("Validation loss:", validation_loss)
        validation_losses.append(validation_loss / n_validation_batches)

        epoch += 1
        if epoch == 0 or min_loss > validation_loss:
            fruitless_epoch = 0
            min_loss = validation_loss
            best_state_dict = deepcopy(network.state_dict())
        else:
            fruitless_epoch += 1

    print("Total epochs:", epoch)
    #print("Validation losses:", validation_losses)

    return train_losses, validation_losses, best_state_dict


######################
# Posterior estimation
######################

# NOTE: z combinations (with pdim > 1) should not be generated here, but just
# fed it. They can be generated externally.

def eval_net(net, x0, z, n_batch = 64):
    """Evaluates network.

    Args:
        net (nn.Module): trained ratio estimation net.
        x0 (torch.tensor): data.
        z : (nsamples, pnum, pdim)
        n_batch (int): minibatch size.

    Returns:
        net output: (nsamples, pnum)
    """
    nsamples = len(z)

    out = []
    for i in tqdm(range(nsamples//n_batch+1)):
        zbatch = z[i*n_batch:(i+1)*n_batch]
        out += net(x0.unsqueeze(0), zbatch).detach().cpu()

    return torch.stack(out)


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
    
    z : (..., pnum, pdim)
    y : (..., ydim)
    
    returns: (..., pnum, ydim + pdim)
    
    """
    y = y.unsqueeze(-2) # (..., 1, ydim)
    y = y.expand(*z.shape[:-1], *y.shape[-1:]) # (..., pnum, ydim)
    return torch.cat([y, z], -1)

class DenseLegs(nn.Module):
    def __init__(self, ydim, pnum, pdim = 1, p = 0.0, NH = 256):
        super().__init__()
        self.fc1 = LinearWithChannel(ydim+pdim, NH, pnum)
        self.fc2 = LinearWithChannel(NH, NH, pnum)
        self.fc3 = LinearWithChannel(NH, NH, pnum)
        self.fc4 = LinearWithChannel(NH, 1, pnum)
        self.drop = nn.Dropout(p = p)

        self.af = torch.relu

        # swish activation function for smooth posteriors
        self.af2 = lambda x: x*torch.sigmoid(x*10.)

    def forward(self, y, z):
        x = combine(y, z)
        x = self.af(self.fc1(x))
        x = self.drop(x)
        x = self.af(self.fc2(x))
        x = self.drop(x)
        x = self.af(self.fc3(x))
        x = self.fc4(x).squeeze(-1)
        return x

def get_norms(xz, combinations = None, N = 300):
    irand = np.random.choice(len(xz), N)
    x = [xz[i]['x'] for i in irand]
    z = [xz[i]['z'] for i in irand]
    x_mean = sum(x)/len(x)
    z_mean = sum(z)/len(z)
    x_var = sum([(x[i]-x_mean)**2 for i in range(len(x))])/len(x)
    z_var = sum([(z[i]-z_mean)**2 for i in range(len(z))])/len(z)

    z_mean = combine_z(z_mean, combinations)
    z_var = combine_z(z_var, combinations)

    #print("Normalizations")
    #print("x_mean", x_mean)
    #print("x_err", x_var**0.5)
    #print("z_mean", z_mean)
    #print("z_err", z_var**0.5)

    return x_mean, x_var**0.5, z_mean, z_var**0.5

class Network(nn.Module):
    def __init__(self, ydim, pnum, pdim = 1, head = None, p = 0., datanorms = None):
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
        self.legs = DenseLegs(ydim, pnum, pdim = pdim, p = p)

        # Set datascaling
        if datanorms is None:
            datanorms = [torch.tensor(0.), torch.tensor(1.), torch.tensor(0.5), torch.tensor(0.5)]
        self._set_datanorms(*datanorms)

    def _set_datanorms(self, x_mean, x_std, z_mean, z_std):
        self.x_loc = torch.nn.Parameter(x_mean)
        self.x_scale = torch.nn.Parameter(x_std)
        self.z_loc = torch.nn.Parameter(z_mean)
        self.z_scale = torch.nn.Parameter(z_std)
    
    def forward(self, x, z):
        x = (x-self.x_loc)/self.x_scale
        z = (z-self.z_loc)/self.z_scale

        if self.head is not None:
            y = self.head(x)
        else:
            y = x  # Use 1-dim data vector as features

        out = self.legs(y, z)
        return out

def sample_constrained_hypercube(nsamples, zdim, mask):
    done = False
    zout = defaultdict(lambda: [])
    counter1 = np.zeros(zdim)  # Counter of accepted points in each z component
    counter2 = np.zeros(zdim)  # Counter of tested points in each z component
    while not done:
        z = torch.rand(nsamples, zdim)
        m = mask(z.unsqueeze(-1))
        for i in range(zdim):
            zout[i].append(z[m[:,i], i])
            counter1[i] += m[:,i].sum()
            counter2[i] += nsamples
        done = min(counter1) >= nsamples
    post_vol = np.true_divide(counter1, counter2)  # constrained posterior volume
    print("Constrained posterior volume:", post_vol.prod())
    
    out = torch.stack([torch.cat(zout[i]).squeeze(-1)[:nsamples] for i in range(zdim)]).T
    return out

# NOTE: This mask works on exactly the parameter combinations that were also
# used for the definition of the network, not plain z vectors.
class Mask:
    def __init__(self, net, x0, threshold):
        self.x0 = x0
        self.net = net
        self.threshold = threshold
        self.device = x0.device

    def __call__(self, z):
        """
        Args:
            z : (nsamples, pnum, pdim)

        Returns:
            mask : (nsamples, pnum)
        """
        z = z.to(self.device)
        lnL = get_lnL(self.net, self.x0, z).cpu()
        lnL -= lnL.max(axis=0)[0]
        return lnL > np.log(self.threshold)

def get_stats(z, p):
    # Returns central credible intervals
    zmax = z[p.argmax()]
    c = cumtrapz(p, z, initial = 0)
    res = np.interp([0.025, 0.16, 0.5, 0.84, 0.975], c, z)
    xmedian = res[2]
    xerr68 = [res[1], res[3]]
    xerr95 = [res[0], res[4]]
    return {'mode': zmax, 'median': xmedian,
            'cred68': xerr68, 'cred95': xerr95,
            'err68': (xerr68[1] - xerr68[0])/2,
            'err95': (xerr95[1] - xerr95[0])/2,
            }


####
# Intensity functions
####

class Mask1d:
    """A 1-dim multi-interval based mask class."""
    def __init__(self, intervals):
        self.intervals = np.array(intervals)  # n x 2 matrix
        
    def __call__(self, z):
        """Returns 1. if inside interval, otherwise 0."""
        m = np.zeros_like(z)
        for z0, z1 in self.intervals:
            m += np.where(z >= z0, np.where(z <= z1, 1., 0.), 0.)
        assert not any(m > 1.), "Overlapping intervals."
        return m
        
    def area(self):
        """Combined length of all intervals (AKAK 1-dim area)."""
        return (self.intervals[:, 1] - self.intervals[:, 0]).sum()
    
    def sample(self, N):
        p = self.intervals[:, 1] - self.intervals[:,0]
        p /= p.sum()
        i = np.random.choice(len(p), size = N, replace = True, p = p)
        w = np.random.rand(N)
        z = self.intervals[i,0] + w*(self.intervals[i,1] - self.intervals[i,0])
        return z


class FactorMask:
    """A d-dim factorized mask."""
    def __init__(self, masks):
        self.masks = masks
        self.d = len(masks)
    
    def __call__(self, z):
        m = [self.masks[i](z[:,i]) for i in range(self.d)]
        m = np.array(m).prod(axis=0)
        return m
    
    def area(self):
        m = [self.masks[i].area() for i in range(self.d)]
        return np.array(m).prod()
    
    def sample(self, N):
        z = np.empty((N, self.d))
        for i in range(self.d):
            z[:,i] = self.masks[i].sample(N)
        return z


class Intensity:
    """Intensity function based on d-dim mask."""
    def __init__(self, mu, mask):
        self.mu = mu
        self.mask = mask
        self.area = mask.area()
        
    def __call__(self, z):
        return self.mask(z)/self.area*self.mu
    
    def sample(self, N = None):
        if N is None:
            N = np.random.poisson(self.mu, 1)[0]
        return self.mask.sample(N)


def construct_intervals(x, y):
    """Get x intervals where y is above 0."""
    indices = np.argsort(x)
    x = x[indices]
    y = y[indices]
    m = np.where(y > 0., 1., 0.)
    m = m[1:] - m[:-1]
    i0 = np.argwhere(m == 1.)[:,0]  # Upcrossings
    i1 = np.argwhere(m == -1.)[:,0]  # Downcrossings
    
    # No crossings --> return entire interval
    if len(i0) == 0 and len(i1) == 0:
        return [[x[0], x[-1]]]
    
    # One more upcrossing than downcrossing
    # --> Treat right end as downcrossing
    if len(i0) - len(i1) == 1:
        i1 = np.append(i1, -1)
  
    # One more downcrossing than upcrossing
    # --> Treat left end as upcrossing
    if len(i0) - len(i1) == -1:
        i0 = np.append(0, i0)
      
    intervals = []
    for i in range(len(i0)):
        intervals.append([x[i0[i]], x[i1[i]]])
    
    return intervals
    
def trainloop(net, dataset, combinations = None, nbatch = 32, nworkers = 4,
        max_epochs = 50, early_stopping_patience = 1, device = 'cpu', lr_schedule = [1e-3, 1e-4, 1e-5], nl_schedule = [1.0, 1.0, 1.0]):
    print("Start training")
    nvalid = 512
    ntrain = len(dataset) - nvalid
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [ntrain, nvalid])
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=nbatch, num_workers=nworkers, pin_memory=True, drop_last=True, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=nbatch, num_workers=nworkers, pin_memory=True, drop_last=True, shuffle = False)
    # Train!

    train_loss, valid_loss = [], []
    for i, lr in enumerate(lr_schedule):
        print(f'LR iteration {i}')
        #dataset.set_noiselevel(nl_schedule[i])
        tl, vl, sd = train(net, train_loader, valid_loader,
                early_stopping_patience = early_stopping_patience, lr = lr,
                max_epochs = max_epochs, device=device, combinations =
                combinations)
        vl_minimum = min(vl)
        vl_min_idx = vl.index(vl_minimum)
        train_loss.append(tl[:vl_min_idx + 1])
        valid_loss.append(vl[:vl_min_idx + 1])
        net.load_state_dict(sd)

    return train_loss, valid_loss

def get_ratios(x0, net, dataset, combinations = None, device = 'cpu', Nmax = 1000):
    x0 = x0.to(device)
    z = get_z(dataset)[:Nmax]
    z = torch.stack(z).to(device)
    z = torch.stack([combine_z(zs, combinations) for zs in z])
    ratios = eval_net(net, x0, z)
    return z.cpu(), ratios.cpu()

