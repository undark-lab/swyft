import numpy as np
import torch
from scipy.interpolate import interp1d


class NormalizeStd:
    def __init__(self, values):
        self.mean = {}
        self.std = {}

        for k, v in values.items():
            self.mean[k] = v.mean(axis=0)
            self.std[k] = v.std(axis=0).mean()

    def __call__(self, values):
        out = {}
        for k, v in values.items():
            out[k] = (v - self.mean[k]) / self.std[k]
        return out


class NormalizeScale:
    def __init__(self, values):
        self.median = {}
        self.perc = {}

        for k, v in values.items():
            median = np.percentile(v, 50, axis=0)
            perc = np.percentile(v - median, np.linspace(0, 100, 101))
            self.median[k] = median
            self.perc[k] = perc

    def __call__(self, values):
        out = {}
        for k, v in values.items():
            v = v - self.median[k]
            v = interp1d(
                self.perc[k], np.linspace(-1, 1, 101), fill_value="extrapolate"
            )(v)
            out[k] = v
        return out


Normalize = NormalizeStd


class Transform:
    def __init__(self, par_combinations, param_transform=None, obs_transform=None):
        self.obs_transform = (lambda x: x) if obs_transform is None else obs_transform
        self.param_transform = (
            (lambda z: z) if param_transform is None else param_transform
        )
        self.par_combinations = par_combinations
        self.par_comb_shape = self._get_par_comb_shape(par_combinations)

    def _get_par_comb_shape(self, par_combinations):
        n = len(par_combinations)
        m = max([len(c) for c in par_combinations])
        return (n, m)

    def _combine(self, par):
        shape = par[list(par)[0]].shape
        if len(shape) == 0:
            out = torch.zeros(self.par_comb_shape)
            for i, c in enumerate(self.par_combinations):
                pars = torch.stack([par[k] for k in c]).T
                out[i, : pars.shape[0]] = pars
        else:
            n = shape[0]
            out = torch.zeros((n,) + self.par_comb_shape)
            for i, c in enumerate(self.par_combinations):
                pars = torch.stack([par[k] for k in c]).T
                out[:, i, : pars.shape[1]] = pars
        return out

    def _tensorfy(self, x):
        return {k: torch.tensor(v).float() for k, v in x.items()}

    def __call__(self, obs=None, par=None):
        out = {}
        if obs is not None:
            tmp = self.obs_transform(obs)
            out["obs"] = self._tensorfy(tmp)
        if par is not None:
            tmp = self.param_transform(par)
            z = self._tensorfy(tmp)
            out["par"] = self._combine(z)
        return out
