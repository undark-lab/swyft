import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from .core import *

class SWYFT:
    """SWYFT. Convenience class around functional methods."""
    def __init__(self, model, z_dim, x0, device = 'cpu', verbosity = True):
        self.model = model
        self.z_dim = z_dim
        self.x0 = x0 
        self.device = device

        self.xz_store = []
        self.net_store = []
        self.loss_store = []
        self.test_loss_store = []
        self.verbose = verbosity

    def square(self, n_train = [3000,3000,3000], lr = [1e-3,1e-4,1e-5], head = None,
            combinations = None, p = 0.2, n_batch = 3):

        # Instantiate network
        if head is None:
            x_dim = len(self.x0)
        else:
            x_dim = head(torch.tensor(self.x0).float().to(self.device)).shape[-1]
        zdim = len(combinations)
        pdim = len(combinations[0])
        network = Network(x_dim, zdim, pdim = pdim, head = head, p = p).to(self.device)
        xz = self.xz_store[-1]
        
        z = get_z(xz)
        network.train()

        losses = []
        for i in range(len(lr)):
            loss = train(network, xz, n_train = n_train[i], lr = lr[i],
                    device = self.device, n_batch = n_batch, combinations=combinations)
            losses += loss

        network.eval()
        out = estimate_lnL_batched(network, self.x0, z, device = self.device,
                normalize = False, combinations = combinations)
        return out, losses

    def round(self, n_sims = 3000, n_train = [3000,3000,3000], lr = [1e-3,1e-4,1e-5],
            head = None, combine = False, p = 0.2, n_batch = 3, threshold = 1e-6):
        if self.verbose:
            print("Round: ", len(self.xz_store))
        if n_sims > 0:
            n_tests = int(n_sims/10)

            # Generate new training data
            if self.verbose:
                print("Generate samples from constrained prior: z~pc(z)")
            if len(self.net_store) == 0:
                z = sample_z(n_sims, self.z_dim)  # draw from initial prior
            else:
                z = iter_sample_z(n_sims, self.z_dim, self.net_store[-1], self.x0, device = self.device, verbosity = self.verbose,
                        threshold = threshold)

            # time sink
            if self.verbose:
                print("Generate corresponding draws x ~ p(x|z)")
            xz = sample_x(self.model, z)  # generate corresponding model samples

            if combine:
                xz += self.xz_store[-1]
        else:
            # Take simply previous samples
            if self.verbose:
                print("Reusing samples from previous round.")
            xz = self.xz_store[-1]
            n_tests = int(len(xz)/10)

        # Instantiate network
        if head is None:
            x_dim = len(self.x0)
        else:
            x_dim = head(torch.tensor(self.x0).float().to(self.device)).shape[-1]
        network = Network(x_dim, self.z_dim, xz_init = xz, head = head, p = p).to(self.device)
        network.train()

        if self.verbose:
            print("Network optimization")
        # Perform optimization
        if isinstance(lr, list):
            losses = []
            for i in range(len(lr)):
                loss = train(network, xz, n_train =
                        n_train[i], lr = lr[i], device =
                        self.device, n_batch = n_batch)
                losses += loss
        else:
            losses = train(network, xz, n_train =
                    n_train, lr = lr, device =
                    self.device, n_batch = n_batch)

        # Store results
        self.xz_store.append(xz)
        self.net_store.append(network)
        self.loss_store.append(losses)

    def get_posteriors(self, nround = -1, x0 = None, error = False, z = None, n_sub = 100):
        network = self.net_store[nround]
        if z is None:
            z = get_z(self.xz_store[nround])
        x0 = x0 if x0 is not None else self.x0

        if not error:
            network.eval()
            z_lnL = estimate_lnL_batched(network, x0, z, device = self.device, normalize = True, n_sub = n_sub)
            return z_lnL
        else:
            network.train()
            z_lnL_list = []
            zsub = subsample(n_sub, z)
            for i in tqdm(range(100), desc="Estimating std"):
                z_lnL = estimate_lnL_batched(network, x0, zsub, device = self.device, normalize = False, n_sub = 100000000)
                z_lnL_list.append(z_lnL)
            std_list = []
            for j in range(len(z_lnL_list[0])):
                tmp = [z_lnL_list[i][j]['lnL'] for i in range(len(zsub))]
                mean = sum(tmp)/len(zsub)
                tmp = [(z_lnL_list[i][j]['lnL']-mean)**2 for i in range(len(zsub))]
                var = sum(tmp)/len(zsub)
                std = var**0.5
                std_list.append(std)
            return std_list

    def save(self, filename):
        """Save current state, including sampled data, loss history and fitted networks.

        :param filename: Output filename, .pt format.
        :type filename: String
        """
        obj = {'xz_store': self.xz_store,
               'loss_store': self.loss_store,
               'net_store': self.net_store
               }
        torch.save(obj, filename)

    def load(self, filename):
        """Load previously saved state, including sampled data, loss history and fitted networks.

        :param filename: Input filename, .pt format.
        :type filename: String
        """
        obj = torch.load(filename)
        self.xz_store = obj['xz_store']
        self.net_store = obj['net_store']
        self.loss_store = obj['loss_store']
