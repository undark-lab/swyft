# pylint: disable=no-member, not-callable
import numpy as np
from scipy.integrate import trapz

import torch
import torch.nn as nn
from swyft.core import *

from copy import deepcopy

def construct_intervals(x, y):
    """Get x intervals where y is above 0."""
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

def posteriors(x0, net, dataset, combinations = None, device = 'cpu'):
    x0 = x0.to(device)
    z = torch.stack(get_z(dataset)).to(device)
    z = torch.stack([combine_z(zs, combinations) for zs in z])
    lnL = get_lnL(net, x0, z)
    return z.cpu(), lnL.cpu()


class SWYFT:
    """Main SWYFT interface."""
    def __init__(self, x0, zdim, head = None, noisemodel = None, device = 'cpu', datastore = None, parent = None, nsamples = 3000, threshold = 1e-6, res = 1e-4):
        """Initialize SWYFT.

        Args:
            x0 (array): Observational data.
            zdim (int): Number of parameters.
            head (class): Head network class.
            noisemodel (function): Function return noise.
            device (str): Device type.

        Returns:
            Instance of SWYFT.
        """
        self.x0 = torch.tensor(x0).float()
        self.noisemodel = noisemodel
        self.zdim = zdim
        self.head_cls = head  # head network class
        self.device = device

        if datastore == None:
            raise ValueError("Need datastore!")
        self.ds = datastore

        self.parent = parent

        self.intensity = None
        self.train_indices = None
        self.net1d = None
        self.post1d = None

        self.netNd = dict()
        self.postNd = dict()

        self.need_eval_post1d = True
        self.need_eval_postNd = dict()

        self._init_train_data(nsamples = nsamples, threshold = threshold, res = res)

    def _get_net(self, pnum, pdim, head = None, datanorms = None, recycle_net = False):
        # Check whether we can jump-start with using a copy of the previous network
        if self.parent is not None and recycle_net:
            net = deepcopy(self.parent.net1d)
            return net

        # Otherwise, initialize new neural network
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

    def get_dataset(self):
        """Retrieve training dataset from datastore and SWYFT object train history."""
        indices = self.train_indices
        dataset = DataDS(self.ds, indices, self.noisemodel)
        return dataset

    def train1d(self, max_epochs = 100, nbatch = 16, lr_schedule = [1e-3, 1e-4, 1e-5], nl_schedule = [1.0, 1.0, 1.0], early_stopping_patience = 3, nworkers = 0): 
        """Train 1-dim posteriors."""
        if self.net1d is None:
            self._init_net1d()
        self.need_eval_post1d = True
        net = self.net1d
        dataset = self.get_dataset()

        # Start actual training
        trainloop(net, dataset, device = self.device, max_epochs = max_epochs,
                nbatch = nbatch, lr_schedule = lr_schedule, nl_schedule =
                nl_schedule, early_stopping_patience = early_stopping_patience, nworkers=nworkers)

    def trainNd(self, combinations = None, max_epochs = 100, nbatch = 8, lr_schedule = [1e-3, 1e-4, 1e-5], nl_schedule = [1.0, 1.0, 1.0], early_stopping_patience = 3, nworkers = 0, tag = 'default'): 
        """Train 1-dim posteriors."""
        if tag not in self.netNd.keys():
            self.init_netNd(combinations, tag = tag)
        if combinations is not None:
            assert combinations == self.netNd[tag]['combinations']

        self.need_eval_postNd[tag] = True
            
        net = self.netNd[tag]['net']
        combinations = self.netNd[tag]['combinations']
        dataset = self.get_dataset()

        # Start actual training
        trainloop(net, dataset, combinations = combinations, device = self.device, max_epochs = max_epochs,
                nbatch = nbatch, lr_schedule = lr_schedule, nl_schedule =
                nl_schedule, early_stopping_patience = early_stopping_patience, nworkers=nworkers)

    def _init_train_data(self, nsamples = 3000, threshold = 1e-6, res = 1e-4):
        """Advance SWYFT internal training data history on constrained prior."""

        if self.parent is None:
            # Generate initial intensity over hypercube
            mask1d = Mask1d([[0., 1.]])
            masks_1d = [mask1d]*self.zdim
        else:
            # Generate target intensity based on previous round
            net = self.parent.net1d
            intervals_list = self.get_intervals(net, threshold = threshold, res = res)
            masks_1d = [Mask1d(tmp) for tmp in intervals_list]

        factormask = FactorMask(masks_1d)
        print("Constrained posterior area:", factormask.area())
        intensity = Intensity(nsamples, factormask)
        indices = self.ds.sample(intensity)

        # Append new training samples to train history, including intensity function
        self.intensity = intensity
        self.train_indices = indices

    def _init_net1d(self, recycle_net = False):
        """Advance SWYFT-internal net1d history."""
        # Set proper data normalizations for network initialization
        dataset = self.get_dataset()
        datanorms = get_norms(dataset)

        # Initialize network
        net = self._get_net(self.zdim, 1, datanorms = datanorms, recycle_net = recycle_net)

        # And append it to history!
        self.net1d = net

    def _eval_post1d(self):
        # Get 1-dim posteriors
        net = self.net1d
        dataset = self.get_dataset()
        z, lnL = posteriors(self.x0, net, dataset, device = self.device)

        # Store results
        self.post1d = [z, lnL]
        self.need_eval_post1d = False

    def requires_sim(self):
        """Check whether simulations are required to complete datastore."""
        return len(self.ds.require_sim()) > 0

    def run(self, nrounds = 1, nsamples = 3000, threshold = 1e-6, max_epochs =
            100, recycle_net = True, nbatch = 8, lr_schedule = [1e-3, 1e-4,
                1e-5], nl_schedule = [0.1, 0.3, 1.0], early_stopping_patience =
            20, nworkers = 4):
        """Iteratively generating training data and train 1-dim posteriors."""
        raise NotImplementedError
        #for i in range(nrounds):
        #    self.advance_train_history(nsamples = nsamples, threshold = threshold)

        #    if self.requires_sim():
        #        pass  # TODO: Run simulations if needed!

        #    self.advance_net1d_history()

        #    self.train1d(max_epochs = max_epochs,
        #            nbatch = nbatch, lr_schedule = lr_schedule, nl_schedule =
        #            nl_schedule, early_stopping_patience =
        #            early_stopping_patience, nworkers=nworkers)

        #    self.advance_post1d_history()

    @staticmethod
    def _prep_post_1dim(x, y):
        # Sort and normalize posterior
        # NOTE: 1-dim posteriors are automatically normalized
        # TODO: Normalization should be done based on prior range, not enforced by hand
        isorted = np.argsort(x)
        x, y = x[isorted], y[isorted]
        y = np.exp(y)
        I = trapz(y, x)
        return x, y/I

    def posterior(self, indices, x0 = None, tag = 'default'):
        """Return generated posteriors."""
        if isinstance(indices, int):
            if self.need_eval_post1d:
                self._eval_post1d()
            i = indices
            if x0 is None:
                x = self.post1d[0][:,i,0]
                y = self.post1d[1][:,i]
                return self._prep_post_1dim(x, y)
            else:
                raise NotImplementedError
                #net = self.net1d_history[version]
                #dataset = self.data_store[version]
                #x0 = torch.tensor(x0).float().to(self.device)
                #x, y = posteriors(x0, net, dataset, combinations = None, device = self.device)
                #x = x[:,i,0]
                #y = y[:,i]
                #return self._prep_post_1dim(x, y)
        else:
            if self.need_eval_postNd[tag]:
                self.eval_postNd(tag = tag)
            combinations = self.postNd[tag][0]
            j = combinations.index(indices)
            return self.postNd[tag][1][:,j], self.postNd[tag][2][:,j]

    def get_intervals(self, net, res = 1e-5, threshold = 1e-6):
        """Generate intervals from previous posteriors."""
        nbins = int(1./res)+1
        z = torch.linspace(0, 1, nbins).repeat(self.zdim, 1).T.unsqueeze(-1).to(self.device)
        lnL = get_lnL(net, self.x0.to(self.device), z)  
        z = z.cpu().numpy()[:,:,0]
        lnL = lnL.cpu().numpy()
        intervals_list = []
        for i in range(self.zdim):
            lnL_max = lnL[:,i].max()
            intervals = construct_intervals(z[:,i], lnL[:,i] - lnL_max - np.log(threshold))
            intervals_list.append(intervals)
        return intervals_list

    def init_netNd(self, combinations, recycle_net = False, tag = 'default'):
        """Generate N-dim posteriors."""
        # Use by default data from last 1-dim round
        dataset = self.get_dataset()
        datanorms = get_norms(dataset, combinations = combinations)
        
        # Generate network
        pnum = len(combinations)
        pdim = len(combinations[0])

        if recycle_net:
            head = deepcopy(self.net1d.head)
            net = self._get_net(pnum, pdim, head = head, datanorms = datanorms)
        else:
            net = self._get_net(pnum, pdim, datanorms = datanorms)
            
        self.netNd[tag] = dict(net=net, combinations=combinations)

    def eval_postNd(self, tag = 'default'):
        # Get posteriors and store them internally
        net = self.netNd[tag]['net']
        combinations = self.netNd[tag]['combinations']
        dataset = self.get_dataset()
        
        zgrid, lnLgrid = posteriors(self.x0, net, dataset, combinations =
                combinations, device = self.device)

        self.postNd[tag] = (combinations, zgrid, lnLgrid)
        self.need_eval_postNd[tag] = False
