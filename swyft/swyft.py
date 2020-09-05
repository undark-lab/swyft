# pylint: disable=no-member, not-callable
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from .core import *

from copy import deepcopy

class Data(torch.utils.data.Dataset):
    def __init__(self, xz):
        super().__init__()
        self.xz = xz

    def __len__(self):
        return len(self.xz)

    def __getitem__(self, idx):
        return self.xz[idx]

def gen_train_data(model, nsamples, zdim, mask = None):
    # Generate training data
    if mask is None:
        z = sample_hypercube(nsamples, zdim)
    else:
        z = sample_constrained_hypercube(nsamples, zdim, mask)
    
    xz = simulate_xz(model, z)
    dataset = Data(xz)
    
    return dataset

def trainloop(net, dataset, combinations = None, nbatch = 8, nworkers = 4,
        max_epochs = 100, early_stopping_patience = 20, device = 'cpu'):
    nvalid = 512
    ntrain = len(dataset) - nvalid
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [ntrain, nvalid])
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=nbatch, num_workers=nworkers, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=nbatch, num_workers=nworkers, pin_memory=True, drop_last=True)
    # Train!

    train_loss, valid_loss = [], []
    for i, lr in enumerate([1e-3, 1e-4, 1e-5]):
        print(f'LR Iter {i}', end="\r")
        tl, vl, sd = train(net, train_loader, valid_loader,
                early_stopping_patience = early_stopping_patience, lr = lr,
                max_epochs = max_epochs, device=device, combinations =
                combinations)
        vl_minimum = min(vl)
        vl_min_idx = vl.index(vl_minimum)
        train_loss.append(tl[:vl_min_idx + 1])
        valid_loss.append(vl[:vl_min_idx + 1])
        net.load_state_dict(sd)

def posteriors(x0, net, dataset, combinations = None, device = 'cpu'):
    x0 = x0.to(device)
    z = torch.stack(get_z(dataset)).to(device)
    z = torch.stack([combine_z(zs, combinations) for zs in z])
    lnL = get_lnL(net, x0, z)
    return z.cpu(), lnL.cpu()

class SWYFT:
    def __init__(self, x0, model, zdim, head = None, device = 'cpu', max_epochs = 100):
        self.x0 = torch.tensor(x0).float()
        self.model = model
        self.zdim = zdim
        self.head_cls = head  # head network class
        self.device = device
        self.max_epochs = max_epochs

        # Each data_store entry has a corresponding mask entry
        # TODO: Replace with datastore eventually
        self.mask_store = []
        self.data_store = []

        # NOTE: Each trained network goes together with evaluated posteriors (evaluated on x0)
        self.post1d_store = []
        self.net1d_store = []

        # NOTE: We separate N-dim posteriors since they are not used (yet) for refining training data
        self.postNd_store = []
        self.netNd_store = []

    def _get_net(self, pnum, pdim, head = None):
        # Initialize neural network
        if self.head_cls is None and head is None:
            head = None
            ydim = len(self.x0)
        elif head is not None:
            ydim = head(self.x0.unsqueeze(0).to(self.device)).shape[1]
            print("Number of output features:", ydim)
        else:
            head = self.head_cls()
            ydim = head(self.x0.unsqueeze(0)).shape[1]
            print("Number of output features:", ydim)
        net = Network(ydim = ydim, pnum = pnum, pdim = pdim, head = head).to(self.device)
        return net

    def append_dataset(self, dataset):
        """Append dataset to data_store, assuming unconstrained prior."""
        self.data_store.append(dataset)
        self.mask_store.append(None)

    def train(self, recycle_net = True):
        """Train 1-dim posteriors."""
        # Use most recent dataset by default
        dataset = self.data_store[-1]

        if len(self.net1d_store) > 0 and recycle_net:
            #print('deepcopy, HAHAHAHhahahahaheheheeee!')
            net = deepcopy(self.net1d_store[-1])
        else:
            net = self._get_net(self.zdim, 1)

        # Train
        trainloop(net, dataset, device = self.device, max_epochs = self.max_epochs)

        # Get 1-dim posteriors
        zgrid, lnLgrid = posteriors(self.x0, net, dataset, device = self.device)

        # Store results
        self.net1d_store.append(net)
        self.post1d_store.append((zgrid, lnLgrid))

    def data(self, nsamples = 3000):
        """Generate training data on constrained prior."""
        if len(self.mask_store) == 0:
            mask = None
        else:
            last_net = self.net1d_store[-1]
            mask = Mask(last_net, self.x0.to(self.device), 1e-8)

        dataset = gen_train_data(self.model, nsamples, self.zdim, mask = mask)

        # Store dataset and mask
        self.mask_store.append(mask)
        self.data_store.append(dataset)

    def run(self, nrounds = 1):
        """Iteratively generating training data and train 1-dim posteriors."""
        for i in range(nrounds):
            if self.model is None:
                print("WARNING: No model provided. Skipping data generation.")
            else:
                self.data()
            self.train()

    def comb(self, combinations):
        """Generate N-dim posteriors."""
        # Use by default data from last 1-dim round
        dataset = self.data_store[-1]

        # Generate network
        pnum = len(combinations)
        pdim = len(combinations[0])

        head = deepcopy(self.net1d_store[-1].head)
        net = self._get_net(pnum, pdim, head = head)

        # Train!
        trainloop(net, dataset, combinations = combinations, device =
                self.device, max_epochs = self.max_epochs)

        # Get posteriors and store them internally
        zgrid, lnLgrid = posteriors(self.x0, net, dataset, combinations =
                combinations, device = self.device)

        self.postNd_store.append((combinations, zgrid, lnLgrid))
        self.netNd_store.append(net)

    def posterior(self, indices):
        """Return generated posteriors."""
        if isinstance(indices, int):
            i = indices
            return self.post1d_store[-1][0][:,i], self.post1d_store[-1][1][:,i]
        else:
            for i in range(len(self.postNd_store)-1, -1, -1):
                combinations = self.postNd_store[i][0]
                if indices in combinations:
                    j = combinations.index(indices)
                    return self.postNd_store[i][1][:,j], self.postNd_store[i][2][:,j]
            print("WARNING: Did not find requested parameter combination.")
            return None
