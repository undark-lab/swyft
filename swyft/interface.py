# pylint: disable=no-membeinr, not-callable
import numpy as np
from scipy.integrate import trapz

import torch
import torch.nn as nn
from swyft.core import *

from copy import deepcopy

class RatioEstimation:
    """
    `RatioEstimation` performs ratio estimation.
    """
    def __init__(self, zdim, traindata, combinations = None, head = None, device = 'cpu', parent = None):
        self.zdim = zdim
        self.head_cls = head  # head network class
        self.device = device
        self.traindata = traindata
        self.parent = parent
        self.net = None
        self.ratio_cache = dict()
        self.combinations = combinations

        self._init_net(self.combinations)

    def _get_dataset(self):
        return self.traindata.get_dataset()

    def _get_net(self, pnum, pdim, head = None, datanorms = None, recycle_net = False):
        # Check whether we can jump-start with using a copy of the previous network
        if self.parent is not None and recycle_net:
            net = deepcopy(self.parent.net1d)
            return net

        # Otherwise, initialize new neural network
        if self.head_cls is None and head is None:
            head = None
            ds = self._get_dataset()
            ydim = len(ds[0]['x'])
        elif head is not None:
            ydim = head(self.x0.unsqueeze(0).to(self.device)).shape[1]
            print("Number of output features:", ydim)
        else:
            head = self.head_cls()
            ydim = head(self.x0.unsqueeze(0)).shape[1]
            print("Number of output features:", ydim)
        net = Network(ydim = ydim, pnum = pnum, pdim = pdim, head = head, datanorms = datanorms).to(self.device)
        return net

    def _init_net(self, combinations, recycle_net = False, tag = 'default'):
        """Generate N-dim posteriors."""
        # Use by default data from last 1-dim round
        dataset = self._get_dataset()
        datanorms = get_norms(dataset, combinations = self.combinations)
        
        # Generate network
        pnum = len(self.combinations)
        pdim = len(self.combinations[0])

        if recycle_net:
            head = deepcopy(self.net1d.head)
            net = self._get_net(pnum, pdim, head = head, datanorms = datanorms)
        else:
            net = self._get_net(pnum, pdim, datanorms = datanorms)
            
        self.net = net

    def train(self, max_epochs = 100, nbatch = 8, lr_schedule = [1e-3, 1e-4, 1e-5], nl_schedule = [1.0, 1.0, 1.0], early_stopping_patience = 1, nworkers = 0, tag = 'default'): 
        """Train higher-dimensional marginal posteriors.

        Args:
            tag (string): Tag indicating network of interest.  Default is "default".
            max_epochs (int): Maximum number of training epochs.
            nbatch (int): Minibatch size.
            lr_schedule (list): List of learning rates.
            early_stopping_patience (int): Early stopping patience.
            nworkers (int): Number of Dataloader workers.
        """
        net = self.net
        dataset = self._get_dataset()

        # Start actual training
        trainloop(net, dataset, combinations = self.combinations, device = self.device, max_epochs = max_epochs,
                nbatch = nbatch, lr_schedule = lr_schedule, nl_schedule =
                nl_schedule, early_stopping_patience = early_stopping_patience, nworkers=nworkers)


#    def train1d(self, max_epochs = 100, nbatch = 16, lr_schedule = [1e-3, 1e-4, 1e-5], nl_schedule = [1.0, 1.0, 1.0], early_stopping_patience = 1, nworkers = 0): 
#        """Train 1-dim marginal posteriors.
#
#        Args:
#            max_epochs (int): Maximum number of training epochs.
#            nbatch (int): Minibatch size.
#            lr_schedule (list): List of learning rates.
#            early_stopping_patience (int): Early stopping patience.
#            nworkers (int): Number of Dataloader workers.
#        """
#        if self.net1d is None:
#            self._init_net1d()
#        self.need_eval_post1d = True
#        net = self.net1d
#        dataset = self._get_dataset()
#
#        # Start actual training
#        trainloop(net, dataset, device = self.device, max_epochs = max_epochs,
#                nbatch = nbatch, lr_schedule = lr_schedule, nl_schedule =
#                nl_schedule, early_stopping_patience = early_stopping_patience, nworkers=nworkers)

#    def _init_net1d(self, recycle_net = False):
#        """Advance SWYFT-internal net1d history."""
#        # Set proper data normalizations for network initialization
#        dataset = self._get_dataset()
#        datanorms = get_norms(dataset)
#
#        # Initialize network
#        net = self._get_net(self.zdim, 1, datanorms = datanorms, recycle_net = recycle_net)
#
#        # And append it to history!
#        self.net1d = net

    def _eval_posterior(self, x0):
        if x0.tobytes() in self.ratio_cache.keys():
            return
        dataset = self._get_dataset()
        z, ratios = get_ratios(torch.tensor(x0).float(), self.net, dataset, device = self.device,
                combinations = self.combinations)
        self.ratio_cache[x0.tobytes()] = [z, ratios]

# TODO: Reuse again???
#    @staticmethod
#    def _prep_post_1dim(x, y):
#        # Sort and normalize posterior
#        # NOTE: 1-dim posteriors are automatically normalized
#        # TODO: Normalization should be done based on prior range, not enforced by hand
#        isorted = np.argsort(x)
#        x, y = x[isorted], y[isorted]
#        y = np.exp(y)
#        I = trapz(y, x)
#        return x, y/I

    def posterior(self, x0, indices):
        """Retrieve estimated marginal posterior.

        Args:
            indices (int, list of ints): Parameter indices.
            x0 (array-like): Overwrites target image. Optional.

        Returns:
            x-array, p-array
        """
        self._eval_posterior(x0)

        if isinstance(indices, int):
            indices = [indices]

        j = self.combinations.index(indices)
        z, ratios = self.ratio_cache[x0.tobytes()][0][:,j], self.ratio_cache[x0.tobytes()][1][:,j]

        # 1-dim case
        if len(indices) == 1:
            z = z[:,0]
            isorted = np.argsort(z)
            z, ratios = z[isorted], ratios[isorted]

        p = np.exp(ratios)

        return z, p



class TrainData:
    """
    `TrainData` on contrained priors for Nested Ratio Estimation.

    Args:
        x0 (array): Observational data.
        zdim (int): Number of parameters.
        head (class): Head network class.
        noisemodel (function): Function return noise.
        device (str): Device type.
    """
    def __init__(self, x0, zdim, noisemodel = None, datastore = None, parent = None, nsamples = 3000, threshold = 1e-7):
        self.x0 = torch.tensor(x0).float()
        self.zdim = zdim

        if datastore == None:
            raise ValueError("Need datastore!")
        self.ds = datastore

        self.parent = parent

        self.intensity = None
        self.train_indices = None

        self.noisemodel = noisemodel

        self._init_train_data(nsamples = nsamples, threshold = threshold)

    def get_dataset(self):
        """Retrieve training dataset from datastore and SWYFT object train history."""
        indices = self.train_indices
        dataset = DataContainer(self.ds, indices, self.noisemodel)
        return dataset

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

