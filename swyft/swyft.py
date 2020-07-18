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

    def round(self, n_sims = 3000, n_train = [3000,3000,3000], lr = [1e-3,1e-4,1e-5], n_particles = 1,
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
            losses_test = []
            for i in range(len(lr)):
                loss, loss_test  = train(network, xz[:-n_tests-1], n_steps =
                        n_train[i], lr = lr[i], n_particles = n_particles, device =
                        self.device, xz_test = xz[-n_tests-1:], n_batch = n_batch)
                losses += loss
                losses_test += loss_test
        else:
            losses, losses_test  = train(network, xz[:-n_tests-1], n_steps =
                    n_train, lr = lr, n_particles = n_particles, device =
                    self.device, xz_test = xz[-n_tests-1:], n_batch = n_batch)

        # Store results
        self.xz_store.append(xz)
        self.net_store.append(network)
        self.loss_store.append(losses)
        self.test_loss_store.append(losses_test)

    def get_posteriors(self, nround = -1, x0 = None, MC_dropout = False):
        network = self.net_store[nround]
        if MC_dropout:
            network.train()
        else:
            network.eval()
        z = get_z(self.xz_store[nround])
        x0 = x0 if x0 is not None else self.x0
        z_lnL = estimate_lnL(network, x0, z, device = self.device)
        return z_lnL

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
