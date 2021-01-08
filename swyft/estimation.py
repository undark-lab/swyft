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

from .train import trainloop
from .network import Network, DenseLegs, DefaultHead
from .eval import get_ratios
from .types import (
    Sequence,
    Tuple,
    Device,
    Dataset,
    Combinations,
    Callable,
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
        head: Optional[nn.Module] = DefaultHead,
        tail: Optional[nn.Module] = DenseLegs,
        previous_ratio_estimator=None,
        device: Device = "cpu",
        statistics=None,
    ):
        """RatioEstimator takes simulated points from the iP3 sample cache and handles training and posterior calculation.

        Args:
            points: points dataset from the iP3 sample cache
            combinations: which combinations of z parameters to learn
            head: initialized module which processes observations, head(x0) = y
            previous_ratio_estimator: ratio estimator from another round. if given, reuse head.
            device: default is cpu
            statistics: x_mean, x_std, z_mean, z_std
        """
        self.points = points
        self.prev_re = previous_ratio_estimator
        self.device = device
        self.head = head().to(device)

        self.n_features = len(self.head(self.points[0]['obs']))
        self.pnum, self.pdim = self.points[0]['par'].shape

        self.tail = tail(self.n_features, self.pnum, self.pdim).to(device)

        print("Number of features from head network:", self.n_features)
        print("Number of parameters to estimate:", self.pnum)
        print("Maximum posterior dimensionality:", self.pdim)

    @property
    def zdim(self):
        return self.points.zdim

    @property
    def xshape(self):
        return self.points.xshape

    @property
    def combinations(self) -> Combinations:
        """Marginals that the ratio estimator learns."""
        if self._combinations is None:
            return [[i] for i in range(self.zdim)]
        else:
            return process_combinations(self._combinations)

    def _init_net(self, statistics: Tuple, previous_ratio_estimator):
        """Options for custom network initialization.

        Args:
            statistics: x_mean, x_std, z_mean, z_std
            previous_ratio_estimator: ratio estimator from another round. if given, reuse head.
        """
        # TODO change this to state_dict and deal with the self.head is None case more gracefully.
        if previous_ratio_estimator is not None:
            if self.head is not None:
                warn("using previous ratio estimator's head rather than yours.")
            self.head = deepcopy(self.prev_re.net.head)
        # TODO this is an antipattern address it in network by removing pnum and pdim
        #pnum = len(self.combinations)
        #pdim = len(self.combinations[0])

        # TODO network should be able to handle shape or dim. right now we are forcing dim, that is bad.
        #input_x = array_to_tensor(torch.empty(1, *self.xshape, device=self.device))
        # yshape = self.head(input_x).shape[1:]
        #yshape = self.head(input_x).shape[1]

        #print("yshape (shape of features between head and legs):", yshape)
        self.head
        return Network(
            ydim=self.n_features, pnum=self.pnum, pdim=self.pdim, head=self.head, datanorms=statistics, tail = self.tail
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
            max_epochs: maximum number of training epochs
            batch_size: minibatch size
            lr_schedule: list of learning rates
            early_stopping_patience: early stopping patience
            nworkers: number of Dataloader workers (0 for no dataloader parallelization)
            percent_validation: percentage to allocate to validation set
        """
        trainloop(
            self.head, self.tail,
            self.points,
            combinations=None,
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

    def lnL(
        self,
        obs: Array,
        par: Array,
        transform,
        max_n_points: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve estimated marginal posterior.

        Args:
            x0: real observation to calculate posterior
            combination_indices: z indices in self.combinations
            max_n_points: number of points to calculate ratios on

        Returns:
            parameter array, posterior array
        """

        obs = transform(obs=obs)['obs']
        obs = {k: v.to(self.device) for k, v in obs.items()}
        f = self.head(obs)
        z = transform(par=par)['par']
        if len(z.shape) == 2:
            lnL = self.tail(f, z.to(self.device))
            lnL = lnL.detach().cpu().numpy()
        else:
            N = len(z)
            batch_size = 1000
            lnL = []
            for i in range(N // batch_size + 1):
                zbatch = z[i * batch_size : (i + 1) * batch_size]
                lnL.append(self.tail(f, zbatch.to(self.device)).detach().cpu().numpy())
            lnL = np.vstack(lnL)

        return {k: lnL[..., i] for i, k in enumerate(transform.par_combinations)}

    def posterior(
        self,
        x0: Array,
        combination_indices: Union[int, Sequence[int]],
        max_n_points: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve estimated marginal posterior.

        Args:
            x0: real observation to calculate posterior
            combination_indices: z indices in self.combinations
            max_n_points: number of points to calculate ratios on

        Returns:
            parameter array, posterior array
        """
        self._eval_ratios(x0, max_n_points=max_n_points)

        if isinstance(combination_indices, int):
            combination_indices = [combination_indices]

        j = self.combinations.index(combination_indices)
        z = self.ratio_cache[tobytes(x0)]["z"][:, j]
        ratios = self.ratio_cache[tobytes(x0)]["ratios"][:, j]

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
        cache: "swyft.cache.Cache",
        path: PathType,
        head: nn.Module = None,
        noisehook: Callable = None,
        device: Device = None,
    ):
        """Load pickled ratio estimator.

        Args:
            cache: cache object on which the estimator was trained
            path: path to pickled ratio estimator
            head: head network utilized in pickled ratio estimator
            noisehook: maps from (x, z) to x with noise
            device
        """
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


class Points:
    """Points references (observation, parameter) pairs drawn from an inhomogenous Poisson Point Proccess (iP3) Cache.
    Points implements this via a list of indices corresponding to data contained in a cache which is provided at initialization.
    """

    _save_attrs = ["intensity", "indices"]

    def __init__(self, cache: "swyft.cache.Cache", indices, noisehook=None):
        """Create a points dataset

        Args:
            cache (Cache): iP3 cache for zarr storage
            intensity (Intensity): inhomogenous Poisson Point Proccess intensity function on parameters
            noisehook (function): (optional) maps from (x, z) to x with noise
        """
        #super().__init__()
        if cache.requires_sim():
            raise RuntimeError(
                "The cache has parameters without a corresponding observation. Try running the simulator."
            )

        self.cache = cache
        #self.intensity = intensity
        self.noisehook = noisehook
        self.indices = np.array(indices)

    def __len__(self):
        #assert len(self.indices) <= len(
        #    self.cache
        #), f"You wanted {len(self.indices)} indices but there are only {len(self.cache)} parameter samples in the cache."
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x_keys = list(self.cache.x)
        z_keys = list(self.cache.z)
        x = {k: self.cache.x[k][i] for k in x_keys}
        z = {k: self.cache.z[k][i] for k in z_keys}

        if self.noisehook is not None:
            x = self.noisehook(x, z)

        return dict(obs=x, par=z)

    def z(self):
        return {k: v[:] for k, v in self.cache.z.items()}

    @property
    def zdim(self):
        #assert (
        #    self.intensity.zdim == self.cache.zdim
        #), "The cache and intensity functions did not agree on the zdim."
        return self.cache.zdim

    @property
    def xshape(self):
        return self.cache.xshape

#    @property
#    def indices(self):
#        if self._indices is None:
#            self._indices = self.cache.sample(self.intensity)
#            self._check_fidelity_to_cache(self._indices)
#        return self._indices

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
    def load(cls, cache: "swyft.cache.Cache", path: PathType, noisehook=None):
        """Loads saved indices and intensity from a pickle. User provides cache and noisehook."""
        path = Path(path)
        with path.open("rb") as f:
            attrs = pickle.load(f)
        return cls._load(cache, attrs, noisehook)
