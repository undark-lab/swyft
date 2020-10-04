# pylint: disable=no-member, not-callable
import numpy as np
from scipy.integrate import trapz

import torch
import torch.nn as nn
from swyft.core import *

from copy import deepcopy



class SWYFT:
    """
    `SWYFT` is the top-level interface for a single round of Nested Ratio Estimation.

    Args:
        x0 (array): Observational data.
        zdim (int): Number of parameters.
        head (class): Head network class.
        noisemodel (function): Function return noise.
        device (str): Device type.
    """
    def __init__(self, x0, zdim, head = None, noisemodel = None, device = 'cpu', datastore = None, parent = None, nsamples = 3000, threshold = 1e-7):
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

        self._init_train_data(nsamples = nsamples, threshold = threshold)

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

    def _get_dataset(self):
        """Retrieve training dataset from datastore and SWYFT object train history."""
        indices = self.train_indices
        dataset = DataDS(self.ds, indices, self.noisemodel)
        return dataset

    def train1d(self, max_epochs = 100, nbatch = 16, lr_schedule = [1e-3, 1e-4, 1e-5], nl_schedule = [1.0, 1.0, 1.0], early_stopping_patience = 1, nworkers = 0): 
        """Train 1-dim marginal posteriors.

        Args:
            max_epochs (int): Maximum number of training epochs.
            nbatch (int): Minibatch size.
            lr_schedule (list): List of learning rates.
            early_stopping_patience (int): Early stopping patience.
            nworkers (int): Number of Dataloader workers.
        """
        if self.net1d is None:
            self._init_net1d()
        self.need_eval_post1d = True
        net = self.net1d
        dataset = self._get_dataset()

        # Start actual training
        trainloop(net, dataset, device = self.device, max_epochs = max_epochs,
                nbatch = nbatch, lr_schedule = lr_schedule, nl_schedule =
                nl_schedule, early_stopping_patience = early_stopping_patience, nworkers=nworkers)

    def trainNd(self, combinations = None, max_epochs = 100, nbatch = 8, lr_schedule = [1e-3, 1e-4, 1e-5], nl_schedule = [1.0, 1.0, 1.0], early_stopping_patience = 1, nworkers = 0, tag = 'default'): 
        """Train higher-dimensional marginal posteriors.

        Args:
            combinations (list): List of posteriors of interest.
            tag (string): Tag indicating network of interest.  Default is "default".
            max_epochs (int): Maximum number of training epochs.
            nbatch (int): Minibatch size.
            lr_schedule (list): List of learning rates.
            early_stopping_patience (int): Early stopping patience.
            nworkers (int): Number of Dataloader workers.
        """

        if tag not in self.netNd.keys():
            self._init_netNd(combinations, tag = tag)
        if combinations is not None:
            assert combinations == self.netNd[tag]['combinations']

        self.need_eval_postNd[tag] = True
            
        net = self.netNd[tag]['net']
        combinations = self.netNd[tag]['combinations']
        dataset = self._get_dataset()

        # Start actual training
        trainloop(net, dataset, combinations = combinations, device = self.device, max_epochs = max_epochs,
                nbatch = nbatch, lr_schedule = lr_schedule, nl_schedule =
                nl_schedule, early_stopping_patience = early_stopping_patience, nworkers=nworkers)

    def _init_train_data(self, nsamples = 3000, threshold = 1e-7):
        """Advance SWYFT internal training data history on constrained prior."""

        if self.parent is None:
            # Generate initial intensity over hypercube
            mask1d = Mask1d([[0., 1.]])
            masks_1d = [mask1d]*self.zdim
        else:
            # Generate target intensity based on previous round
            net = self.parent.net1d
            intensity = self.parent.intensity
            intervals_list = self._get_intervals(net, intensity, threshold = threshold)
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
        dataset = self._get_dataset()
        datanorms = get_norms(dataset)

        # Initialize network
        net = self._get_net(self.zdim, 1, datanorms = datanorms, recycle_net = recycle_net)

        # And append it to history!
        self.net1d = net

    def _eval_post1d(self):
        # Get 1-dim posteriors
        net = self.net1d
        dataset = self._get_dataset()
        z, lnL = posteriors(self.x0, net, dataset, device = self.device)

        # Store results
        self.post1d = [z, lnL]
        self.need_eval_post1d = False

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
        """Retrieve estimated marginal posterior.

        Args:
            indices (int, list of ints): Parameter indices.
            x0 (array-like): Overwrites target image. Optional.
            tag (string): Tag of Ndim network.

        Returns:
            x-array, p-array
        """
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
                self._eval_postNd(tag = tag)
            combinations = self.postNd[tag][0]
            j = combinations.index(indices)
            return self.postNd[tag][1][:,j], self.postNd[tag][2][:,j]

    def _get_intervals(self, net, intensity, N = 10000, threshold = 1e-7):
        """Generate intervals from previous posteriors."""
        z = torch.tensor(intensity.sample(N = N)).float().unsqueeze(-1).to(self.device)
        lnL = get_lnL(net, self.x0.to(self.device), z)  
        z = z.cpu().numpy()[:,:,0]
        lnL = lnL.cpu().numpy()
        intervals_list = []
        for i in range(self.zdim):
            lnL_max = lnL[:,i].max()
            intervals = construct_intervals(z[:,i], lnL[:,i] - lnL_max - np.log(threshold))
            intervals_list.append(intervals)
        return intervals_list

    def _init_netNd(self, combinations, recycle_net = False, tag = 'default'):
        """Generate N-dim posteriors."""
        # Use by default data from last 1-dim round
        dataset = self._get_dataset()
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

    def _eval_postNd(self, tag = 'default'):
        # Get posteriors and store them internally
        net = self.netNd[tag]['net']
        combinations = self.netNd[tag]['combinations']
        dataset = self._get_dataset()
        
        zgrid, lnLgrid = posteriors(self.x0, net, dataset, combinations =
                combinations, device = self.device)

        self.postNd[tag] = (combinations, zgrid, lnLgrid)
        self.need_eval_postNd[tag] = False
