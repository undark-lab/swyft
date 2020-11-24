# pylint: disable=no-member, not-callable
from warnings import warn
from copy import deepcopy
from functools import wraps
from pathlib import Path
import pickle

import numpy as np
from scipy.integrate import trapz

import torch
import torch.nn as nn

from .cache import Cache
from .train import get_norms, trainloop
from .network import Network
from .eval import get_ratios, eval_net
from .intensity import construct_intervals, Mask1d, FactorMask, Intensity
from .types import (
    Sequence,
    Tuple,
    Device,
    Dataset,
    Combinations,
    Array,
    Union,
    PathType,
    Dict,
    Optional,
)
from .utils import array_to_tensor, tobytes, process_combinations


class RatioEstimator:

    _save_attrs = ["net_state_dict", "ratio_cache", "combinations"]

    def __init__(
        self,
        points: Dataset,
        combinations: Optional[Combinations] = None,
        head: Optional[nn.Module] = None,
        previous_ratio_estimator=None,
        device: Device = "cpu",
        statistics=None,
        recycle_net: bool = False,
    ):
        """RatioEstimator takes simulated points from the iP3 sample cache and handles training and posterior calculation.

        Args:
            points: points dataset from the iP3 sample cache
            combinations: which combinations of z parameters to learn
            head (nn.Module): (optional) initialized module which processes observations, head(x0) = y
            previous_ratio_estimator: (optional) ratio estimator from another round
            device: (optional) default is cpu
            statistics (): mean and std for x and z
            recycle_net (bool): set net with the previous ratio estimator's net
        """
        self.points = points
        self._combinations = combinations
        self.head = head if head is None else head.to(device)
        self.prev_re = previous_ratio_estimator
        self.device = device

        self.net = self._init_net(statistics, recycle_net)
        self.ratio_cache: Dict[bytes : np.ndarray] = {}

    @property
    def zdim(self):
        return self.points.zdim

    @property
    def xshape(self):
        return self.points.xshape

    @property
    def combinations(self):
        if self._combinations is None:
            return [[i] for i in range(self.zdim)]
        else:
            return process_combinations(self._combinations)

    def _init_net(self, statistics, recycle_net: bool):
        """Options for custom network initialization.

        Args:
            statistics: mean and std for x and z
            recycle_net (bool): set net with the previous ratio estimator's net
        """
        if recycle_net:
            if statistics is not None:
                warn(
                    "Since the network is being recycled, your statistics are being ignored."
                )
            return deepcopy(self.prev_re.net)
        # TODO this is an antipattern address it in network by removing pnum and pdim
        pnum = len(self.combinations)
        pdim = len(self.combinations[0])

        # TODO network should be able to handle shape or dim. right now we are forcing dim, that is bad.
        if self.head is None:
            # yshape = self.xshape
            yshape = self.xshape[0]
        else:
            input_x = array_to_tensor(torch.empty(1, *self.xshape, device=self.device))
            # yshape = self.head(input_x).shape[1:]
            yshape = self.head(input_x).shape[1]
        print("yshape (shape of features between head and legs):", yshape)
        return Network(
            ydim=yshape, pnum=pnum, pdim=pdim, head=self.head, datanorms=statistics
        ).to(self.device)

    def train(
        self,
        max_epochs: int = 100,
        batch_size: int = 8,
        lr_schedule: Sequence[float] = [1e-3, 1e-4, 1e-5],
        early_stopping_patience: int = 1,
        nworkers: int = 0,
        percent_validation=0.1,
    ) -> None:
        """Train higher-dimensional marginal posteriors.

        Args:
            max_epochs (int): maximum number of training epochs
            batch_size (int): minibatch size
            lr_schedule (list): list of learning rates
            early_stopping_patience (int): early stopping patience
            nworkers (int): number of Dataloader workers
            percent_validation (float): percentage to allocate to validation set
        """
        trainloop(
            self.net,
            self.points,
            combinations=self.combinations,
            device=self.device,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr_schedule=lr_schedule,
            early_stopping_patience=early_stopping_patience,
            nworkers=nworkers,
            percent_validation=percent_validation,
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
                x0,
                self.net,
                self.points,
                device=self.device,
                combinations=self.combinations,
                max_n_points=max_n_points,
            )
            self.ratio_cache[binary_x0] = {"z": z, "ratios": ratios}
        return None

    def posterior(
        self,
        x0: Array,
        combination_indices: Union[int, Sequence[int]],
        max_n_points: int = 1000,
    ):
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
        z = self.ratio_cache[x0.tobytes()]["z"][:, j]
        ratios = self.ratio_cache[x0.tobytes()]["ratios"][:, j]

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

    @property
    def net_state_dict(self):
        return self.net.state_dict()

    @property
    def _save_dict(self):
        return {attr: getattr(self, attr) for attr in RatioEstimator._save_attrs}

    def save(self, path: PathType):
        complete = {}
        complete.update(self.points._save_dict)
        complete.update(self._save_dict)

        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(complete, f)
        return None

    @classmethod
    def load(
        cls,
        cache: Cache,
        path: PathType,
        head: nn.Module = None,
        noisehook=None,
        device: Device = None,
    ):
        path = Path(path)
        with path.open("rb") as f:
            complete = pickle.load(f)

        point_attrs = {attr: complete[attr] for attr in Points._save_attrs}
        points = Points._load(cache, point_attrs, noisehook)

        re_attrs = {attr: complete[attr] for attr in RatioEstimator._save_attrs}
        instance = cls(points, re_attrs["combinations"], head, device)
        instance.net.load_state_dict(
            re_attrs["net_state_dict"],
        )
        instance.ratio_cache = re_attrs["ratio_cache"]
        return instance


class Points(torch.utils.data.Dataset):
    """Points references (observation, parameter) pairs drawn from an inhomogenous Poisson Point Proccess (iP3) Cache.
    Points implements this via a list of indices corresponding to data contained in a cache which is provided at initialization.
    """

    _save_attrs = ["intensity", "indices"]

    def __init__(self, cache: Cache, intensity, noisehook=None):
        """Create a points dataset

        Args:
            cache (Cache): iP3 cache for zarr storage
            intensity (Intensity): inhomogenous Poisson Point Proccess intensity function on parameters
            noisehook (function): (optional) maps from (x, z) to x with noise
        """
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
        ), f"You wanted {len(self.indices)} indices but there are only {len(self.cache)} parameter samples in the cache."
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

    @property
    def zdim(self):
        assert (
            self.intensity.zdim == self.cache.zdim
        ), "The cache and intensity functions did not agree on the zdim."
        return self.intensity.zdim

    @property
    def xshape(self):
        return self.cache.xshape

    @property
    def indices(self):
        if self._indices is None:
            self._indices = self.cache.sample(self.intensity)
            self._check_fidelity_to_cache(self._indices)
        return self._indices

    def _check_fidelity_to_cache(self, indices):
        first = indices[0]
        assert self.zdim == len(
            self.cache.z[first]
        ), "Item in cache did not agree with zdim."
        assert (
            self.xshape == self.cache.x[first].shape
        ), "Item in cache did not agree with xshape."
        return None

    @property
    def _save_dict(self):
        return {attr: getattr(self, attr) for attr in Points._save_attrs}

    def save(self, path: PathType):
        """Saves indices and intensity in a pickle."""
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self._save_dict, f)
        return None

    @classmethod
    def _load(cls, cache, attrs: dict, noisehook):
        instance = cls(cache, attrs["intensity"], noisehook=noisehook)
        instance._indices = attrs["indices"]
        instance._check_fidelity_to_cache(attrs["indices"])
        return instance

    @classmethod
    def load(cls, cache: Cache, path: PathType, noisehook=None):
        """Loads saved indices and intensity from a pickle. User provides cache and noisehook."""
        path = Path(path)
        with path.open("rb") as f:
            attrs = pickle.load(f)
        return cls._load(cache, attrs, noisehook)
