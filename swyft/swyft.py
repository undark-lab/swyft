# pylint: disable=no-member, not-callable
import numpy as np
from scipy.integrate import trapz

import torch
import torch.nn as nn
from .core import *

from copy import deepcopy

class Data(torch.utils.data.Dataset):
    """Data container class.

    Note: The noisemodel allows scheduled noise level increase during training.
    """
    def __init__(self, xz):
        super().__init__()
        self.xz = xz
        self.noisemodel = None

    def set_noisemodel(self, noisemodel):
        self.noisemodel = noisemodel
        self.noiselevel = 1.  # 0: no noise, 1: full noise

    def set_noiselevel(self, level):
        self.noiselevel = level

    def __len__(self):
        return len(self.xz)

    def __getitem__(self, idx):
        xz = self.xz[idx]
        if self.noisemodel is not None:
            x = self.noisemodel(xz['x'].numpy(), z = xz['z'].numpy(), noiselevel = self.noiselevel)
            x = torch.tensor(x).float()
            xz = dict(x=x, z=xz['z'])
        return xz

def gen_train_data(model, nsamples, zdim, mask = None):
    # Generate training data
    if mask is None:
        z = sample_hypercube(nsamples, zdim)
    else:
        z = sample_constrained_hypercube(nsamples, zdim, mask)
    
    xz = simulate_xz(model, z)
    dataset = Data(xz)
    
    return dataset

#if datastore is empty, add sims according to poisson process
#if datastore isn't empty, use constrained posterior via mask as prior  
#then grow the DataStore and sample
def update_datastore(ds, model, nsamples, zdim, mask=None):
    if mask==None:
        #use unit hypercube for prior
        pr=Prior([0.0]*zdim,[1.0]*zdim)
        ds.grow(nsamples, pr);
        z=ds.get_z_without_x()
        xz = simulate_xz(model,torch.tensor(z).float())
        x = get_x(xz)
        ds.fill_sims(x, z)
        dataset = Data(xz)
        return dataset
    else:
        z = ds.z
        zz=torch.tensor(z).float()
        m = mask(zz.unsqueeze(-1))
        #use mask to compute prior
        pr=Prior([z[:,i][m[:,i]].min() for i in range(len(z[0]))],[z[:,i][m[:,i]].max() for i in range(len(z[0]))])
        #enlarge DataStore and compute missing simulations
        ds.grow(nsamples, pr);
        z2=ds.get_z_without_x()
        xz2 = simulate_xz(model,torch.tensor(z2).float()) 
        x2 = get_x(xz2)
        ds.fill_sims(x2, z2)
        #sample the DataStore and return
        x,z=ds.sample(nsamples,pr);
        xz=[dict(x=x[i],z=torch.tensor(z[i]).float()) for i in range(len(x))]
        dataset=Data(xz)
        return dataset


def trainloop(net, dataset, combinations = None, nbatch = 32, nworkers = 4,
        max_epochs = 50, early_stopping_patience = 3, device = 'cpu', lr_schedule = [1e-3, 1e-4, 1e-5], nl_schedule = [1.0, 1.0, 1.0]):
    print("Start training")
    nvalid = 512
    ntrain = len(dataset) - nvalid
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [ntrain, nvalid])
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=nbatch, num_workers=nworkers, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=nbatch, num_workers=nworkers, pin_memory=True, drop_last=True)
    # Train!

    train_loss, valid_loss = [], []
    for i, lr in enumerate(lr_schedule):
        print(f'LR iteration {i}')
        dataset.set_noiselevel(nl_schedule[i])
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
    def __init__(self, x0, model, zdim, head = None, noisemodel = None, device = 'cpu'):
        self.x0 = torch.tensor(x0).float()
        self.model = model
        self.noisemodel = noisemodel
        self.zdim = zdim
        self.head_cls = head  # head network class
        self.device = device

        # Each data_store entry has a corresponding mask entry
        # TODO: Replace with datastore eventually
        self.mask_store = []
        self.data_store = []
        #self.ds is new DataStore class
        self.ds = DataStore()

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

    def train1d(self, recycle_net = True, max_epochs = 100, nbatch = 8, lr_schedule = [1e-3, 1e-4, 1e-5], nl_schedule = [0.1, 0.3, 1.0], early_stopping_patience = 20,nworkers=4): 
        """Train 1-dim posteriors."""
        # Use most recent dataset by default
        dataset = self.data_store[-1]

        dataset.set_noiselevel(1.)
        datanorms = get_norms(dataset)

        # Start by retraining previous network
        if len(self.net1d_store) > 0 and recycle_net:
            net = deepcopy(self.net1d_store[-1])
        else:
            net = self._get_net(self.zdim, 1, datanorms = datanorms)

        # Train
        trainloop(net, dataset, device = self.device, max_epochs = max_epochs,
                nbatch = nbatch, lr_schedule = lr_schedule, nl_schedule =
                nl_schedule, early_stopping_patience = early_stopping_patience, nworkers=nworkers)

        # Get 1-dim posteriors
        zgrid, lnLgrid = posteriors(self.x0, net, dataset, device = self.device)

        # Store results
        self.net1d_store.append(net)
        self.post1d_store.append((zgrid, lnLgrid))

    def data(self, nsamples = 3000, threshold = 1e-6):
        """Generate training data on constrained prior."""
        
        if len(self.mask_store) == 0:
            mask = None
        else:
            last_net = self.net1d_store[-1]
            mask = Mask(last_net, self.x0.to(self.device), threshold)

        
        #dataset = gen_train_data(self.model, nsamples, self.zdim, mask = mask)
        dataset = update_datastore(self.ds, self.model, nsamples, self.zdim, mask=mask)
        dataset.set_noisemodel(self.noisemodel)

        # Store dataset and mask
        self.mask_store.append(mask)
        self.data_store.append(dataset)

    def run(self, nrounds = 1, nsamples = 3000, threshold = 1e-6, max_epochs =
            100, recycle_net = True, nbatch = 8, lr_schedule = [1e-3, 1e-4,
                1e-5], nl_schedule = [0.1, 0.3, 1.0], early_stopping_patience =
            20, nworkers=4):
        """Iteratively generating training data and train 1-dim posteriors."""
        for i in range(nrounds):
            if self.model is None:
                print("WARNING: No model provided. Skipping data generation.")
            else:
                self.data(nsamples = nsamples, threshold = threshold)
            self.train1d(recycle_net = recycle_net, max_epochs = max_epochs,
                    nbatch = nbatch, lr_schedule = lr_schedule, nl_schedule =
                    nl_schedule, early_stopping_patience =
                    early_stopping_patience, nworkers=nworkers)

    def comb(self, combinations, max_epochs = 100, recycle_net = True, nbatch =
            8, lr_schedule = [1e-3, 1e-4, 1e-5], nl_schedule = [0.1, 0.3, 1.0],
            early_stopping_patience = 20, nworkers=4):
        """Generate N-dim posteriors."""
        # Use by default data from last 1-dim round
        dataset = self.data_store[-1]

        dataset.set_noiselevel(1.)
        datanorms = get_norms(dataset, combinations = combinations)

        # Generate network
        pnum = len(combinations)
        pdim = len(combinations[0])

        if recycle_net:
            head = deepcopy(self.net1d_store[-1].head)
            net = self._get_net(pnum, pdim, head = head, datanorms = datanorms)
        else:
            net = self._get_net(pnum, pdim, datanorms = datanorms)

        # Train!
        trainloop(net, dataset, combinations = combinations, device =
                self.device, max_epochs = max_epochs, nbatch = nbatch,
                lr_schedule = lr_schedule, nl_schedule = nl_schedule,
                early_stopping_patience = early_stopping_patience, nworkers=nworkers)

        # Get posteriors and store them internally
        zgrid, lnLgrid = posteriors(self.x0, net, dataset, combinations =
                combinations, device = self.device)

        self.postNd_store.append((combinations, zgrid, lnLgrid))
        self.netNd_store.append(net)

    def _prep_post_1dim(self, x, y):
        # Sort and normalize posterior
        # NOTE: 1-dim posteriors are automatically normalized
        # TODO: Normalization should be done based on prior range, not enforced by hand
        isorted = np.argsort(x)
        x, y = x[isorted], y[isorted]
        y = np.exp(y)
        I = trapz(y, x)
        return x, y/I

    def posterior(self, indices, version = -1, x0 = None):
        """Return generated posteriors."""
        if isinstance(indices, int):
            i = indices
            if x0 is None:
                x = self.post1d_store[version][0][:,i,0]
                y = self.post1d_store[version][1][:,i]
                return self._prep_post_1dim(x, y)
            else:
                net = self.net1d_store[version]
                dataset = self.data_store[version]
                x0 = torch.tensor(x0).float().to(self.device)
                x, y = posteriors(x0, net, dataset, combinations = None, device = self.device)
                x = x[:,i,0]
                y = y[:,i]
                return self._prep_post_1dim(x, y)
        else:
            for i in range(len(self.postNd_store)-1, -1, -1):
                combinations = self.postNd_store[i][0]
                if indices in combinations:
                    j = combinations.index(indices)
                    return self.postNd_store[i][1][:,j], self.postNd_store[i][2][:,j]
            print("WARNING: Did not find requested parameter combination.")
            return None
