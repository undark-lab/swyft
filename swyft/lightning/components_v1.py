import pylab as plt
import numpy as np
import skimage.transform
import skimage.restoration
from tqdm import tqdm
import skimage.filters
import torch
import torch.nn as nn
import typing as tp
from pyrofit.utils import pad_dims
from torch.nn.functional import grid_sample
from pyro import distributions as dist
from pyrofit.lensing.utils import get_meshgrid
from pyrofit.lensing.lenses import SPLELens
from pyrofit.lensing.sources import SersicSource, AnalyticSource
from pyrofit.utils.torchutils import _mid_many, unravel_index
from pyrofit.utils import kNN
from fft_conv_pytorch import fft_conv, FFTConv2d
import swyft
from dataclasses import dataclass
import pytorch_lightning as pl
from torch.nn import functional as F
from swyft.inference.marginalratioestimator import get_ntrain_nvalid

from typing import (
    Callable,
    Dict,
    Hashable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from toolz.dicttoolz import valmap
from torch.utils.data import DataLoader, Dataset, random_split

import swyft
import swyft.utils
from swyft.saveable import StateDictSaveable, StateDictSaveableType
from swyft.types import Array, Device, MarginalIndex, MarginalToArray, ObsType, PathType
from swyft.utils.array import array_to_tensor, dict_array_to_tensor
from swyft.utils.marginals import tupleize_marginal_indices



class RatioEstimatorGaussian1d(torch.nn.Module):
    def __init__(self, momentum = 0.1):
        super().__init__()
        self.momentum = momentum        
        self.x_mean = None
        self.z_mean = None
        self.x_var = None
        self.z_var = None
        self.xz_cov = None
        
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """2-dim Gaussian approximation to marginals and joint, assuming (B, N)."""
        if self.training or self.x_mean is None:
            # Covariance estimates must be based on joined samples only
            # NOTE: This makes assumptions about the structure of mini batches during training (J, M, M, J, J, M, M, J, ...)
            # TODO: Change to (J, M, J, M, J, M, ...) in the future
            batch_size = len(x)
            #idx = np.array([[i, i+3] for i in np.arange(0, batch_size, 4)]).flatten() 
            idx = np.arange(batch_size//2)  # TODO: Assuming (J, J, J, J, M, M, M, M) etc
            
            # Estimation w/o Bessel's correction, using simple MLE estimate (https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices)
            x_mean_batch = x[idx].mean(dim=0).detach()
            z_mean_batch = z[idx].mean(dim=0).detach()
            x_var_batch = ((x[idx]-x_mean_batch)**2).mean(dim=0).detach()
            z_var_batch = ((z[idx]-z_mean_batch)**2).mean(dim=0).detach()
            xz_cov_batch = ((x[idx]-x_mean_batch)*(z[idx]-z_mean_batch)).mean(dim=0).detach()
            
            # Momentum-based update rule
            momentum = self.momentum
            self.x_mean = x_mean_batch if self.x_mean is None else (1-momentum)*self.x_mean + momentum*x_mean_batch
            self.x_var = x_var_batch if self.x_var is None else (1-momentum)*self.x_var + momentum*x_var_batch
            self.z_mean = z_mean_batch if self.z_mean is None else (1-momentum)*self.z_mean + momentum*z_mean_batch
            self.z_var = z_var_batch if self.z_var is None else (1-momentum)*self.z_var + momentum*z_var_batch
            self.xz_cov = xz_cov_batch if self.xz_cov is None else (1-momentum)*self.xz_cov + momentum*xz_cov_batch
            
        # log r(x, z) = log p(x, z)/p(x)/p(z), with covariance given by [[x_var, xz_cov], [xz_cov, z_var]]
        xb = (x-self.x_mean)/self.x_var**0.5
        zb = (z-self.z_mean)/self.z_var**0.5
        rho = self.xz_cov/self.x_var**0.5/self.z_var**0.5
        r = -0.5*torch.log(1-rho**2) + rho/(1-rho**2)*xb*zb - 0.5*rho**2/(1-rho**2)*(xb**2 + zb**2)
        out = torch.cat([r.unsqueeze(-1), z.unsqueeze(-1).detach()], dim=-1)
        return out
    
    
@dataclass
class MeanStd:
    """Store mean and standard deviation"""
    mean: torch.Tensor
    std: torch.Tensor

    def from_samples(samples, weights = None):
        """
        Estimate mean and std deviation of samples by averaging over first dimension.
        Supports weights>=0 with weights.shape = samples.shape
        """
        if weights is None:
            weights = torch.ones_like(samples)
        mean = (samples*weights).sum(axis=0)/weights.sum(axis=0)
        res = samples - mean
        var = (res*res*weights).sum(axis=0)/weights.sum(axis=0)
        return MeanStd(mean = mean, std = var**0.5)
    
    
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self._data = kwargs
    
    def __len__(self):
        k = list(self._data.keys())[0]
        return len(self._data[k])
    
    def __getitem__(self, i):
        obs = {k: v[i] for k, v in self._data.items()}
        v = u = self._data['v'][i]
        return (obs, v, u)
    
    
def subsample_posterior(N, z, replacement = True):
    # Supports only 1-dim posteriors so far
    shape = z.shape
    z = z.view(shape[0], -1, shape[-1])
    w = z[..., 0]
    p = z[..., 1]
    wm = w.max(axis=0).values
    w = torch.exp(w-wm)
    idx = torch.multinomial(w.T, N, replacement = replacement).T
    samples = torch.gather(p, 0, idx)
    samples = samples.view(N, *shape[1:-1])
    return samples

@dataclass
class RectangleBound:
    low: torch.Tensor
    high: torch.Tensor

def get_1d_rect_bounds(samples, th = 1e-6):
    bounds = {}
    for k, v in samples.items():
        r = v.ratios
        r = r - r.max(axis=0).values  # subtract peak
        p = v.values
        #w = v[..., 0]
        #p = v[..., 1]
        all_max = p.max(dim=0).values
        all_min = p.min(dim=0).values
        constr_min = torch.where(r > np.log(th), p, all_max).min(dim=0).values
        constr_max = torch.where(r > np.log(th), p, all_min).max(dim=0).values
        #bound = torch.stack([constr_min, constr_max], dim = -1)
        bound = RectangleBound(constr_min, constr_max)
        bounds[k] = bound
    return bounds

def append_randomized(z):
    assert len(z)%2 == 0, "Cannot expand odd batch dimensions."
    n = len(z)//2
    idx = torch.randperm(n)
    z = torch.cat([z, z[n+idx], z[idx]])
    return z

def append_nonrandomized(z):
    assert len(z)%2 == 0, "Cannot expand odd batch dimensions."
    n = len(z)//2
    idx = np.arange(n)
    z = torch.cat([z, z[n+idx], z[idx]])
    return z

def valmap(m, d):
    return {k: m(v) for k, v in d.items()}

class SwyftModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._predict_condition_x = {}
        self._predict_condition_z = {}
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/KL-div": 0, "hp/JS-div": 0})
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30)
        lr_scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2, factor = 0.25), "monitor": "val_loss"}
        return dict(optimizer = optimizer, lr_scheduler = lr_scheduler)

    def _log_ratios(self, x, z):
        out = self(x, z)
        log_ratios = torch.cat([val.ratios.flatten(start_dim = 1) for val in out.values()])
        return log_ratios
    
    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def _calc_loss(self, batch, batch_idx, randomized = True):
        x, z = batch
        if randomized:
            z = valmap(append_randomized, z)
        else:
            z = valmap(append_nonrandomized, z)
        log_ratios = self._log_ratios(x, z)
        nbatch = len(log_ratios)//2
        y = torch.zeros_like(log_ratios)
        y[:nbatch, ...] = 1
        loss = F.binary_cross_entropy_with_logits(log_ratios, y, reduce = False)
        loss = loss.sum()/nbatch
        return loss
    
    def _calc_KL(self, batch, batch_idx):
        x, z = batch
        log_ratios = self._log_ratios(x, z)
        nbatch = len(log_ratios)
        loss = -log_ratios.sum()/nbatch
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, batch_idx)
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, batch_idx, randomized = False)
        lossKL = self._calc_KL(batch, batch_idx)
        self.log("hp/JS-div", loss)
        self.log("hp/KL-div", lossKL)
        return loss
    
    def _set_predict_conditions(self, condition_x, condition_z):
        self._predict_condition_x = {k: v.unsqueeze(0) for k, v in condition_x.items()}
        self._predict_condition_z = {k: v.unsqueeze(0) for k, v in condition_z.items()}
    
    def predict_step(self, batch, batch_idx):
        x, z = batch
        condition_x = swyft.utils.dict_to_device(self._predict_condition_x, self.device)
        x.update(**condition_x)
        #z.update(**self._predict_condition_z)
        return self(x, z)
    
# https://stackoverflow.com/questions/16463582/memoize-to-disk-python-persistent-memoization
#def persist_to_file():
def persist_to_file(original_func):
        def new_func(*args, file_path = None, **kwargs):
            cache = None
            if file_path is not None:
                try:
                    cache = torch.load(file_path)
                except (FileNotFoundError, ValueError):
                    pass
            if cache is None:
                cache = original_func(*args, **kwargs)
                if file_path is not None:
                    torch.save(cache, file_path)
            return cache
        return new_func
    #return decorator
    

def file_cache(fn, file_path):
    try:
        cache = torch.load(file_path)
    except (FileNotFoundError, ValueError):
        cache = None
    if cache is None:
        cache = fn()
        torch.save(cache, file_path)
    return cache
    
    
def weights_sample(N, values, weights, replacement = True):
    """Weight-based sampling with or without replacement."""
    sw = weights.shape
    sv = values.shape
    assert sw == sv[:len(sw)], "Overlapping left-handed weights and values shapes do not match: %s vs %s"%(str(sv), str(sw))
    
    w = weights.view(weights.shape[0], -1)
    idx = torch.multinomial(w.T, N, replacement = replacement).T
    si = tuple(1 for _ in range(len(sv)-len(sw)))
    idx = idx.view(N, *sw[1:], *si)
    idx = idx.expand(N, *sv[1:])
    
    samples = torch.gather(values, 0, idx)
    return samples
    
def equalize_tensors(a, b):
    n, m = len(a), len(b)
    if n == m:
        return a, b
    elif n == 1:
        shape = list(a.shape)
        shape[0] = m
        return a.expand(*shape), b
    elif m == 1:
        shape = list(b.shape)
        shape[0] = n
        return a, b.expand(*shape)
    elif n < m:
        assert m%n == 0, "Cannot equalize tensors with non-divisible batch sizes."
        shape = [1 for _ in range(a.dim())]
        shape[0] = m//n
        return a.repeat(*shape), b
    else:
        assert n%m == 0, "Cannot equalize tensors with non-divisible batch sizes."
        shape = [1 for _ in range(b.dim())]
        shape[0] = n//m
        return a, b.repeat(*shape)
    
class SwyftTrainer(pl.Trainer):
    def infer(self, model, dataloader, condition_x = {}, condition_z = {}):
        self.model._set_predict_conditions(condition_x, condition_z)
        ratio_batches = self.predict(model, dataloader)
        keys = ratio_batches[0].keys()
        d = {k: RatioSamples(
                torch.cat([r[k].values for r in ratio_batches]),
                torch.cat([r[k].ratios for r in ratio_batches])
                ) for k in keys
            }
        self.model._set_predict_conditions({}, {})  # Set it back to no conditioning
        return RatioSampleStore(**d)

class SampleStore(dict):
    def __len__(self):
        n = [len(v) for v in self.values()]
        assert all([x == n[0] for x in n]), "Inconsistent lengths in SampleStore"
        return n[0]
    
    def __getitem__(self, i):
        """For integers, return 'rows', for string returns 'columns'."""
        if isinstance(i, int):
            return {k: v[i] for k, v in self.items()}
        else:
            return super().__getitem__(i)
        
@dataclass
class RatioSamples:
    values: torch.Tensor
    ratios: torch.Tensor
    
    def __len__(self):
        assert len(self.values) == len(self.ratios), "Inconsistent RatioSamples"
        return len(self.values)
    
    def weights(self, normalize = False):
        ratios = self.ratios
        if normalize:
            ratio_max = ratios.max(axis=0).values
            weights = torch.exp(ratios-ratio_max)
            weights_total = weights.sum(axis=0)
            weights = weights/weights_total*len(weights)
        else:
            weights = torch.exp(ratios)
        return weights
    
    def sample(self, N, replacement = True):
        weights = self.weights(normalize = True)
        if not replacement and N > len(self):
            N = len(self)
        samples = weights_sample(N, self.values, weights, replacement = replacement)
        return samples

class RatioSampleStore(dict):
    def __len__(self):
        n = [len(v) for v in self.values()]
        assert all([x == n[0] for x in n]), "Inconsistent lengths in SampleStore"
        return n[0]
    
    def sample(self, N, replacement = True):
        return {k: v.sample(N, replacement = replacement) for k, v in self.items()}
        
        
# Generate new dictionary

def dictstoremap(model, dictstore):
    N = len(dictstore)
    out = []
    for i in tqdm(range(N)):
        x = model(dictstore[i])
        out.append(x)
    out = torch.utils.data.dataloader.default_collate(out) # using torch internal functionality for this, yay!
    out = {k: v.cpu() for k, v in out.items()}
    return SampleStore(out)

# Since we want to learn the src parameters, which are generated inside the generative model, 
# we simply attach the flattened src array to the parameter vectors and hope for the best

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, x_keys = None, z_keys = None, hook = None):
        self._dataset = dataset
        self._x_keys = x_keys
        self._z_keys = z_keys
        self._hook = hook

    def __len__(self):
        return len(self._dataset[list(self._dataset.keys())[0]])
    
    def __getitem__(self, i):
        d = {k: v[i] for k, v in self._dataset.items()}
        if self._hook is not None:
            d = self._hook(d)
        x_keys = self._x_keys if self._x_keys else d.keys()
        z_keys = self._z_keys if self._z_keys else d.keys()
        x = {k: d[k] for k in x_keys}
        z = {k: d[k] for k in z_keys}
        return x, z
    
class MultiplyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, M):
        self.dataset = dataset
        self.M = M
        
    def __len__(self):
        return len(self.dataset)*self.M
    
    def __getitem__(self, i):
        return self.dataset[i%self.M]
    
    
class SwyftDataModule(pl.LightningDataModule):
    def __init__(self, store, model = None, batch_size: int = 32, validation_percentage = 0.2, manual_seed = None, train_multiply = 10 ):
        super().__init__()
        self.store = store
        self.model = model
        self.batch_size = batch_size
        self.validation_percentage = validation_percentage
        self.train_multiply = train_multiply

    def setup(self, stage):
        hook = None if self.model is None else self.model.noise        
        self.dataset = DictDataset(self.store, hook = hook)#, x_keys = ['data'], z_keys=['z'])
        n_train, n_valid = get_ntrain_nvalid(self.validation_percentage, len(self.dataset))
        self.dataset_train, self.dataset_valid = random_split(self.dataset, [n_train, n_valid], generator=torch.Generator().manual_seed(42))
        self.dataset_test = DictDataset(self.store)#, x_keys = ['data'], z_keys=['z'])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_valid, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_test, batch_size=self.batch_size)
        
    
class SwyftModel:
    def _simulate(self, N, bounds = None, effective_prior = None):
        # TODO: Include conditional priors
        prior_samples = self.prior(N, bounds = bounds)
        if effective_prior:
            for k in effective_prior.keys():
                samples = effective_prior[k].sample(N)
                prior_samples[k] = samples[k]
        simulated_samples = dictstoremap(self.slow, prior_samples)
        samples_tot = dict(**prior_samples, **simulated_samples)
        return SampleStore(samples_tot)
    
    def sample(self, N, bounds = None, effective_prior = None):
        sims = self._simulate(N, bounds = bounds, effective_prior = effective_prior)
        data = dictstoremap(self.fast, sims)
        out = dict(**data, **sims)
        return SampleStore(out)
    
    def noise(self, S):
        S = S.copy()
        D = self.fast(S)
        S.update(D)
        return S#dict(**D, **S)
    
    def __call__(self, S):
        D = self.slow(S)
        D = dict(**D, **S)
        E = self.fast(D)
        return dict(**D, **E)
