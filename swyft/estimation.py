# pylint: disable=no-member, not-callable
from warnings import warn
from copy import deepcopy
from functools import cached_property

import numpy as np
from scipy.integrate import trapz

import torch
import torch.nn as nn

from .cache import Cache
from .train import get_norms, trainloop
from .network import Network
from .eval import get_ratios, eval_net
from .intensity import construct_intervals, Mask1d, FactorMask, Intensity
from .types import Sequence, Tuple, Device, Dataset, Combinations, Array, Union
from .utils import array_to_tensor, tobytes


class RatioEstimator:
    def __init__(
        self,
        x0: Array,
        points: Dataset,
        combinations: Combinations = None,
        head: nn.Module = None,
        previous_ratio_estimator=None,
        device: Device = "cpu",
    ):
        """RatioEstimator takes a real observation and simulated points from the iP3 sample cache and handles training and posterior calculation.

        Args:
            x0 (array): real observation
            points: points dataset from the iP3 sample cache
            combinations: which combinations of z parameters to learn
            head (nn.Module): (optional) initialized module which processes observations, head(x0) = y
            previous_ratio_estimator: (optional) ratio estimator from another round
            device: (optional) default is cpu
        """
        self.x0 = array_to_tensor(x0, device=device)
        self.points = points
        self.combinations = combinations if combinations is not None else self._default_combinations()
        self.head = head
        self.prev_re = previous_ratio_estimator
        self.device = device

        self.net = None
        self.ratio_cache = {}

    @property
    def zdim(self):
        return self.points.zdim

    @property
    def xshape(self):
        assert self.points.xshape == self.x0.shape
        return self.points.xshape
    
    def _default_combinations(self):
        return [[i] for i in range(self.zdim)]

    def init_net(self, statistics=None, recycle_net: bool = False):
        """Options for custom network initialization.

        Args:
            statistics (): mean and std for x and z
            recycle_net (bool): set net with the previous ratio estimator's net
        """
        if recycle_net:
            if statistics is not None:
                warn(
                    "Since the network is being recycled, your statistics are being ignored."
                )
            return deepcopy(self.previous_re.net)

        # TODO this is an antipattern address it in network by removing pnum and pdim
        pnum = len(self.combinations)
        pdim = len(self.combinations[0])

        # TODO network should be able to handle shape or dim. right now we are forcing dim, that is bad.
        if self.head is None:
            # yshape = self.xshape
            yshape = self.xshape[0]
        else:
            # yshape = self.head(self.x0.unsqueeze(0)).shape[1:]
            yshape = self.head(self.x0.unsqueeze(0)).shape[1]
        print("yshape (shape of features between head and legs):", yshape)
        return Network(
            ydim=yshape, pnum=pnum, pdim=pdim, head=self.head, datanorms=statistics
        ).to(self.device)

    def train(
        self,
        max_epochs: int = 100,
        batch_size: int = 8,
        lr_schedule: Sequence[float] = [1e-3, 1e-4, 1e-5],
        nl_schedule: Sequence[float] = [1.0, 1.0, 1.0],
        early_stopping_patience: int = 1,
        nworkers: int = 0,
    ) -> None:
        """Train higher-dimensional marginal posteriors.

        Args:
            max_epochs (int): maximum number of training epochs
            batch_size (int): minibatch size
            lr_schedule (list): list of learning rates
            nl_schedule (list): list of noise levels
            early_stopping_patience (int): early stopping patience
            nworkers (int): number of Dataloader workers
        """
        if self.net is None:
            print("Initializing network...")
            self.net = self.init_net()
        else:
            print("Using pre-initialized network...")

        trainloop(
            self.net,
            self.points,
            combinations=self.combinations,
            device=self.device,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr_schedule=lr_schedule,
            nl_schedule=nl_schedule,
            early_stopping_patience=early_stopping_patience,
            nworkers=nworkers,
        )
        return None

    def _eval_ratios(self, x0: Array, max_n_points: int):
        """Evaluate ratios if not already evaluated.

        Args:
            x0 (array): real observation to calculate posterior
            max_n_points (int): number of points to calculate ratios
        """
        binary_x0 = tobytes(x0)
        if binary_x0 in self.ratio_cache.keys():
            pass
        else:
            z, ratios = get_ratios(
                array_to_tensor(x0, self.device),
                self.net,
                self.points,
                device=self.device,
                combinations=self.combinations,
                max_n_points=max_n_points,
            )
            self.ratio_cache[binary_x0] = {'z': z, 'ratios': ratios}
        return None

    def posterior(self, x0: Array, combination_indices: Union[int, Sequence[int]], max_n_points: int = 1000):
        """Retrieve estimated marginal posterior.

        Args:
            x0 (array): real observation to calculate posterior
            combination_indices (int, list of ints): z indices in self.combinations
            max_n_points (int): number of points to calculate ratios

        Returns:
            parameter (z) array, posterior array
        """
        self._eval_ratios(x0, max_n_points=max_n_points)

        if isinstance(combination_indices, int):
            combination_indices = [combination_indices]

        j = self.combinations.index(combination_indices)
        z = self.ratio_cache[x0.tobytes()]['z'][:, j]
        ratios = self.ratio_cache[x0.tobytes()]['ratios'][:, j]

        # 1-dim case
        if len(combination_indices) == 1:
            z = z[:, 0]
            isorted = np.argsort(z)
            z, ratios = z[isorted], ratios[isorted]
            exp_r = np.exp(ratios)
            I = trapz(exp_r, z)
            p = exp_r / I
        else:
            p = np.exp(ratios)
        return z, p


# class RatioEstimation:
#     def __init__(
#         self,
#         zdim,
#         traindata,
#         combinations=None,
#         head: nn.Module = None,
#         device="cpu",
#         previous_ratio_estimator=None,
#     ):
#         self.zdim = zdim
#         self.head_cls = head  # head network class
#         self.device = device
#         self.traindata = traindata
#         self.previous_re = previous_ratio_estimator
#         self.net = None
#         self.ratio_cache = dict()
#         self.combinations = combinations

#         self._init_net(self.combinations)

#     def _get_dataset(self):
#         return self.traindata.get_dataset()

#     def _get_net(self, pnum, pdim, head=None, datanorms=None, recycle_net=False):
#         # Check whether we can jump-start with using a copy of the previous network
#         if self.previous_re is not None and recycle_net:
#             net = deepcopy(self.previous_re.net)
#             return net

#         # Otherwise, initialize new neural network
#         if self.head_cls is None and head is None:
#             head = None
#             ds = self._get_dataset()
#             ydim = len(ds[0]["x"])
#         elif head is not None:
#             ydim = head(self.traindata.x0.unsqueeze(0).to(self.device)).shape[1]
#             print("Number of output features:", ydim)
#         else:
#             head = self.head_cls()
#             ydim = head(self.traindata.x0.unsqueeze(0)).shape[1]
#             print("Number of output features:", ydim)
#         net = Network(
#             ydim=ydim, pnum=pnum, pdim=pdim, head=head, datanorms=datanorms
#         ).to(self.device)
#         return net

#     def _init_net(self, combinations, recycle_net=False, tag="default"):
#         """Generate N-dim posteriors."""
#         # Use by default data from last 1-dim round
#         dataset = self._get_dataset()
#         datanorms = get_norms(dataset, combinations=self.combinations)

#         # Generate network
#         pnum = len(self.combinations)
#         pdim = len(self.combinations[0])

#         if recycle_net:
#             head = deepcopy(self.net.head)
#             net = self._get_net(pnum, pdim, head=head, datanorms=datanorms)
#         else:
#             net = self._get_net(pnum, pdim, datanorms=datanorms)

#         self.net = net

#     def train(
#         self,
#         max_epochs=100,
#         nbatch=8,
#         lr_schedule=[1e-3, 1e-4, 1e-5],
#         nl_schedule=[1.0, 1.0, 1.0],
#         early_stopping_patience=1,
#         nworkers=0,
#         tag="default",
#     ):
#         """Train higher-dimensional marginal posteriors.

#         Args:
#             tag (string): Tag indicating network of interest.  Default is "default".
#             max_epochs (int): Maximum number of training epochs.
#             nbatch (int): Minibatch size.
#             lr_schedule (list): List of learning rates.
#             early_stopping_patience (int): Early stopping patience.
#             nworkers (int): Number of Dataloader workers.
#         """
#         net = self.net
#         dataset = self._get_dataset()

#         # Start actual training
#         trainloop(
#             net,
#             dataset,
#             combinations=self.combinations,
#             device=self.device,
#             max_epochs=max_epochs,
#             batch_size=nbatch,
#             lr_schedule=lr_schedule,
#             nl_schedule=nl_schedule,
#             early_stopping_patience=early_stopping_patience,
#             nworkers=nworkers,
#         )

#     def _eval_ratios(self, x0, max_n_points=1000):
#         if x0.tobytes() in self.ratio_cache.keys():
#             return
#         dataset = self._get_dataset()
#         z, ratios = get_ratios(
#             torch.tensor(x0).float(),
#             self.net,
#             dataset,
#             device=self.device,
#             combinations=self.combinations,
#             max_n_points=max_n_points,
#         )
#         self.ratio_cache[x0.tobytes()] = [z, ratios]

#     def posterior(self, x0, indices, Nmax=1000):
#         """Retrieve estimated marginal posterior.

#         Args:
#             indices (int, list of ints): Parameter indices.
#             x0 (array-like): Overwrites target image. Optional.

#         Returns:
#             x-array, p-array
#         """
#         self._eval_ratios(x0, max_n_points=Nmax)

#         if isinstance(indices, int):
#             indices = [indices]

#         j = self.combinations.index(indices)
#         z, ratios = (
#             self.ratio_cache[x0.tobytes()][0][:, j],
#             self.ratio_cache[x0.tobytes()][1][:, j],
#         )

#         # 1-dim case
#         if len(indices) == 1:
#             z = z[:, 0]
#             isorted = np.argsort(z)
#             z, ratios = z[isorted], ratios[isorted]
#             exp_r = np.exp(ratios)
#             I = trapz(exp_r, z)
#             p = exp_r / I
#         else:
#             p = np.exp(ratios)
#         return z, p

#     def load_state(self, PATH):
#         self.net.load_state_dict(torch.load(PATH, map_location=self.device))

#     def save_state(self, PATH):
#         torch.save(self.net.state_dict(), PATH)


class Points(torch.utils.data.Dataset):
    """Points references (observation, parameter) pairs drawn from an inhomogenous Poisson Point Proccess (iP3) Cache.
    Points implements this via a list of indices corresponding to data contained in a cache which is provided at initialization.

    Args:
        cache (Cache): iP3 cache for zarr storage
        intensity (Intensity): inhomogenous Poisson Point Proccess intensity function on parameters
        noisehook (function): (optional) maps from (x, z) to x with noise
    """

    def __init__(self, cache: Cache, intensity, noisehook=None):
        super().__init__()
        if cache.requires_sim():
            raise RuntimeError(
                "The cache has parameters without a corresponding observation. Try running the simulator."
            )

        self.cache = cache
        self.intensity = intensity
        self.noisehook = noisehook
        self._indices = None

    def __len__(self):
        assert len(self.indices) <= len(
            self.cache
        ), "You gave more indices than there are parameter samples in the cache."
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.cache.x[i]
        z = self.cache.z[i]

        if self.noisehook is not None:
            x = self.noisehook(x, z)

        x = torch.from_numpy(x).float()
        z = torch.from_numpy(z).float()
        return {
            "x": x,
            "z": z,
        }

    @cached_property
    def zdim(self):
        assert (
            self.intensity.zdim == self.cache.zdim
        ), "The cache and intensity functions did not agree on the zdim."
        return self.intensity.zdim

    @cached_property
    def xshape(self):
        return self.cache.xshape

    @cached_property
    def indices(self):
        if self._indices is None:
            self._indices = self.cache.sample(self.intensity)

        first = self._indices[0]
        assert self.zdim == len(
            self.cache.z[first]
        ), "Item in cache did not agree with zdim."
        assert (
            self.xshape == self.cache.x[first].shape
        ), "Item in cache did not agree with xshape."

        return self._indices

    @classmethod
    def load(cls, path):
        raise NotImplementedError()


# class DataContainer(torch.utils.data.Dataset):
#     """Simple data container class.

#     Note: The noisemodel allows scheduled noise level increase during training.
#     """
#     def __init__(self, cache, indices, noisemodel=None):
#         super().__init__()
#         if cache.requires_sim():
#             raise RuntimeError("The cache has parameters without a corresponding observation. Try running the simulator.")
#         assert len(indices) <= len(cache), "You gave more indices than there are parameter samples in the cache."

#         self.cache = cache
#         self.indices = indices
#         self.noisemodel = noisemodel

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         i = self.indices[idx]
#         x = self.cache.x[i]
#         z = self.cache.z[i]

#         if self.noisemodel is not None:
#             x = self.noisemodel(x, z)

#         x = torch.from_numpy(x).float()
#         z = torch.from_numpy(z).float()

#         xz = dict(x=x, z=z)
#         return xz


# class TrainData:
#     """
#     `TrainData` on contrained priors for Nested Ratio Estimation.

#     Args:
#         x0 (array): Observational data.
#         zdim (int): Number of parameters.
#         head (class): Head network class.
#         noisehook (function): Function return noised data.
#         device (str): Device type.
#     """

#     def __init__(
#         self,
#         x0,
#         zdim,
#         noisehook=None,
#         cache=None,
#         parent=None,
#         nsamples=3000,
#         threshold=1e-7,
#     ):
#         self.x0 = torch.tensor(x0).float()
#         self.zdim = zdim

#         if cache == None:
#             raise ValueError("Need cache!")
#         self.ds = cache

#         self.parent = parent

#         self.intensity = None
#         self.train_indices = None

#         self.noisehook = noisehook

#         self._init_train_data(nsamples=nsamples, threshold=threshold)

#     def get_dataset(self):
#         """Retrieve training dataset from cache and SWYFT object train history."""
#         indices = self.train_indices
#         dataset = DataContainer(self.ds, indices, self.noisehook)
#         return dataset

#     def _init_train_data(self, nsamples=3000, threshold=1e-7):
#         """Advance SWYFT internal training data history on constrained prior."""

#         if self.parent is None:
#             # Generate initial intensity over hypercube
#             mask1d = Mask1d([[0.0, 1.0]])
#             masks_1d = [mask1d] * self.zdim
#         else:
#             # Generate target intensity based on previous round
#             net = self.parent.net
#             intensity = self.parent.traindata.intensity
#             intervals_list = self._get_intervals(net, intensity, threshold=threshold)
#             masks_1d = [Mask1d(tmp) for tmp in intervals_list]

#         factormask = FactorMask(masks_1d)
#         print("Constrained posterior area:", factormask.area())
#         intensity = Intensity(nsamples, factormask)
#         indices = self.ds.sample(intensity)

#         # Append new training samples to train history, including intensity function
#         self.intensity = intensity
#         self.train_indices = indices

#     def _get_intervals(self, net, intensity, N=10000, threshold=1e-7):
#         """Generate intervals from previous posteriors."""
#         z = (
#             torch.tensor(intensity.sample(n=N))
#             .float()
#             .unsqueeze(-1)
#             .to(self.parent.device)
#         )
#         ratios = eval_net(net, self.x0.to(self.parent.device), z)
#         z = z.cpu().numpy()[:, :, 0]
#         ratios = ratios.cpu().numpy()
#         intervals_list = []
#         for i in range(self.zdim):
#             ratios_max = ratios[:, i].max()
#             intervals = construct_intervals(
#                 z[:, i], ratios[:, i] - ratios_max - np.log(threshold)
#             )
#             intervals_list.append(intervals)
#         return intervals_list
