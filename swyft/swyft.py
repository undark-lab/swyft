# pylint: disable=no-member, not-callable
import numpy as np
from scipy.integrate import trapz

import torch
import torch.nn as nn
from .core import *

from copy import deepcopy

class Data(torch.utils.data.Dataset):
    def __init__(self, xz):
        super().__init__()
        self.xz = xz
        self.modelposthook = None

    def set_modelposthook(self, modelposthook):
        self.modelposthook = modelposthook

    def __len__(self):
        return len(self.xz)

    def __getitem__(self, idx):
        if self.modelposthook is None:
            return self.xz[idx]
        else:
            xz = self.xz[idx]
            x = self.modelposthook(xz['x'], xz['z'])
            return dict(x=x, z=xz['z'])

def gen_train_data(model, nsamples, zdim, mask = None, model_kwargs = {}):
    # Generate training data
    if mask is None:
        z = sample_hypercube(nsamples, zdim)
    else:
        z = sample_constrained_hypercube(nsamples, zdim, mask)
    
    xz = simulate_xz(model, z, model_kwargs)
    dataset = Data(xz)
    
    return dataset

def trainloop(net, dataset, combinations = None, nbatch = 8, nworkers = 4,
        max_epochs = 100, early_stopping_patience = 20, device = 'cpu'):
    print("Start training")
    nvalid = 512
    ntrain = len(dataset) - nvalid
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [ntrain, nvalid])
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=nbatch, num_workers=nworkers, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=nbatch, num_workers=nworkers, pin_memory=True, drop_last=True)
    # Train!

    train_loss, valid_loss = [], []
    for i, lr in enumerate([1e-3, 1e-4, 1e-5]):
        print(f'LR iteration {i}', end="\r")
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
    def __init__(self, x0, model, zdim, head = None, device = 'cpu'):
        self.x0 = torch.tensor(x0).float()
        self.model = model
        self.zdim = zdim
        self.head_cls = head  # head network class
        self.device = device

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

    def _get_net(self, pnum, pdim, head = None, datanorms = None):
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
        net = Network(ydim = ydim, pnum = pnum, pdim = pdim, head = head, datanorms = datanorms).to(self.device)
        return net

    def append_dataset(self, dataset):
        """Append dataset to data_store, assuming unconstrained prior."""
        self.data_store.append(dataset)
        self.mask_store.append(None)

    def train1d(self, recycle_net = True, max_epochs = 100, nbatch = 8, modelposthook = None): 
        """Train 1-dim posteriors."""
        # Use most recent dataset by default
        dataset = self.data_store[-1]
        dataset.set_modelposthook(modelposthook)

        datanorms = get_norms(dataset)

        # Start by retraining previous network
        if len(self.net1d_store) > 0 and recycle_net:
            net = deepcopy(self.net1d_store[-1])
        else:
            net = self._get_net(self.zdim, 1, datanorms = datanorms)

        # Train
        trainloop(net, dataset, device = self.device, max_epochs = max_epochs, nbatch = nbatch)

        # Get 1-dim posteriors
        zgrid, lnLgrid = posteriors(self.x0, net, dataset, device = self.device)

        # Store results
        self.net1d_store.append(net)
        self.post1d_store.append((zgrid, lnLgrid))

    def data(self, nsamples = 3000, threshold = 1e-6, model_kwargs = {}):
        """Generate training data on constrained prior."""
        if len(self.mask_store) == 0:
            mask = None
        else:
            last_net = self.net1d_store[-1]
            mask = Mask(last_net, self.x0.to(self.device), threshold)

        dataset = gen_train_data(self.model, nsamples, self.zdim, mask = mask, model_kwargs = model_kwargs)

        # Store dataset and mask
        self.mask_store.append(mask)
        self.data_store.append(dataset)

    def run(self, nrounds = 1, nsamples = 3000, threshold = 1e-6, max_epochs = 100, recycle_net = True, nbatch = 8, model_kwargs = {}, modelposthook = None):
        """Iteratively generating training data and train 1-dim posteriors."""
        for i in range(nrounds):
            if self.model is None:
                print("WARNING: No model provided. Skipping data generation.")
            else:
                self.data(nsamples = nsamples, threshold = threshold, model_kwargs = model_kwargs)
            self.train1d(recycle_net = recycle_net, max_epochs = max_epochs, nbatch = nbatch, modelposthook = modelposthook)

    def comb(self, combinations, max_epochs = 100, recycle_net = True, nbatch = 8):
        """Generate N-dim posteriors."""
        # Use by default data from last 1-dim round
        dataset = self.data_store[-1]

        # Generate network
        pnum = len(combinations)
        pdim = len(combinations[0])

        if recycle_net:
            head = deepcopy(self.net1d_store[-1].head)
            net = self._get_net(pnum, pdim, head = head)
        else:
            net = self._get_net(pnum, pdim)

        # Train!
        trainloop(net, dataset, combinations = combinations, device =
                self.device, max_epochs = max_epochs, nbatch = nbatch)

        # Get posteriors and store them internally
        zgrid, lnLgrid = posteriors(self.x0, net, dataset, combinations =
                combinations, device = self.device)

        self.postNd_store.append((combinations, zgrid, lnLgrid))
        self.netNd_store.append(net)

    def posterior(self, indices, version = -1):
        """Return generated posteriors."""
        # NOTE: 1-dim posteriors are automatically normalized
        # TODO: Normalization should be done based on prior range, not enforced by hand
        if isinstance(indices, int):
            i = indices
            # Sort for convenience
            x = self.post1d_store[version][0][:,i,0]
            y = self.post1d_store[version][1][:,i]
            isorted = np.argsort(x)
            x, y = x[isorted], y[isorted]
            y = np.exp(y)
            I = trapz(y, x)
            return x, y/I
        else:
            for i in range(len(self.postNd_store)-1, -1, -1):
                combinations = self.postNd_store[i][0]
                if indices in combinations:
                    j = combinations.index(indices)
                    return self.postNd_store[i][1][:,j], self.postNd_store[i][2][:,j]
            print("WARNING: Did not find requested parameter combination.")
            return None
