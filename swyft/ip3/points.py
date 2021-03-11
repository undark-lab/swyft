from typing import Callable, List

import numpy as np


class Points:
    """Points references (observation, parameter) pairs drawn from an inhomogenous Poisson Point Proccess (iP3) Cache.
    Points implements this via a list of indices corresponding to data contained in a cache which is provided at initialization.
    """

    _save_attrs = ["indices"]

    def __init__(
        self, indices: List[int], cache: "swyft.cache.Cache", noisehook: Callable = None
    ):  # noqa: F821
        """Create a points dataset

        Args:
            cache (Cache): iP3 cache for zarr storage
            intensity (Intensity): inhomogenous Poisson Point Proccess intensity function on parameters
            noisehook (function): (optional) maps from (x, z) to x with noise
        """
        if cache.any_failed:
            raise RuntimeError(
                "The cache has parameters which failed to return a simulation. Try resampling them."
            )
        elif cache.requires_sim:
            raise RuntimeError(
                "The cache has parameters without a corresponding observation. Try running the simulator."
            )
        assert (
            len(indices) != 0
        ), "You passed indices with length zero. That implies no points."

        self.cache = cache
        self.noisehook = noisehook
        self.indices = np.array(indices)

    def __len__(self):
        return len(self.indices)

    def params(self):
        return self.cache.params

    def get_range(self, indices):
        N = len(indices)
        obs_comb = {k: np.empty((N,) + v.shape) for k, v in self[0]["obs"].items()}
        par_comb = {k: np.empty((N,) + v.shape) for k, v in self[0]["par"].items()}

        for i in indices:
            p = self[i]
            for k, v in p["obs"].items():
                obs_comb[k][i] = v
            for k, v in p["par"].items():
                par_comb[k][i] = v

        return dict(obs=obs_comb, par=par_comb)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x_keys = list(self.cache.x)
        z_keys = list(self.cache.z)
        x = {k: self.cache.x[k][i] for k in x_keys}
        z = {k: self.cache.z[k][i] for k in z_keys}

        if self.noisehook is not None:
            x = self.noisehook(x, z)

        return dict(obs=x, par=z)
