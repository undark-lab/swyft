from dataclasses import dataclass, field
from toolz.dicttoolz import valmap
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
    Any,
)
import numpy as np
import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.cli import instantiate_class

import yaml

from swyft.lightning.samples import *
from swyft.plot.mass import get_empirical_z_score

import scipy
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import torchist


# Main classes


class OptimizerInit:
    def __init__(self, optim_constructor = torch.optim.Adam, optim_args = {"lr": 1e-3},
                 scheduler_constructor = torch.optim.lr_scheduler.ReduceLROnPlateau,
                scheduler_args = {"factor": 0.3, "patience": 5}):
        self.optim_constructor = optim_constructor
        self.optim_args = optim_args
        self.scheduler_constructor = scheduler_constructor
        self.scheduler_args = scheduler_args
        
    def __call__(self, params):
        optimizer = self.optim_constructor(params, **self.optim_args)
        lr_scheduler = {"scheduler": self.scheduler_constructor(
            optimizer, **self.scheduler_args), "monitor": "val_loss"}
        return dict(optimizer = optimizer, lr_scheduler = lr_scheduler)
    
    
class AdamOptimizerInit(OptimizerInit):
    def __init__(self, lr = 1e-3, lrs_factor = 0.3, lrs_patience = 5):
        super().__init__(
            optim_constructor = torch.optim.Adam,
            optim_args = {"lr": lr},
            scheduler_constructor = torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_args = {"factor": lrs_factor, "patience": lrs_patience}
        )


class SwyftParameterError(Exception):
    pass


class SwyftModule(pl.LightningModule):
    def __init__(self):
        r"""
        Handles training of logratio estimators.
        """
        super().__init__()
        self.optimizer_init = AdamOptimizerInit()

#    def configure_callbacks(self):
#        callbacks = []
#        if self._swyft_module_config["lr_monitor"]:
#            callbacks.append(LearningRateMonitor())
#        if self._swyft_module_config["early_stopping"]:
#            early_stop = EarlyStopping(monitor="val_loss", min_delta=0.0,
#                    patience=self._swyft_module_config["early_stopping_patience"],
#                    verbose=False, mode="min")
#            callbacks.append(early_stop)
#        return callbacks

    def configure_optimizers(self):
        return self.optimizer_init(self.parameters())
 
      
    def _logratios(self, x, z):
        out = self(x, z)
        if isinstance(out, dict):
            out = {k: v for k, v in out.items() if k[:4] != 'aux_'}
            logratios = torch.cat([val.logratios.flatten(start_dim = 1) for val in out.values()], dim=1)
        elif isinstance(out, list) or isinstance(out, tuple):
            out = [v for v in out if hasattr(v, 'logratios')]
            logratios = torch.cat([val.logratios.flatten(start_dim = 1) for val in out], dim=1)
        else:
            logratios = out.logratios.flatten(start_dim = 1)
        return logratios
    
    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized = False)
        self.log("val_loss", loss, prog_bar=True, on_step = False, on_epoch = True)
        return loss

    def _calc_loss(self, batch, randomized = True):
        """Calcualte batch-averaged loss summed over ratio estimators.

        Note: The expected loss for an untrained classifier (with f = 0) is
        subtracted.  The initial loss is hence usually close to zero.
        """
        if isinstance(batch, list):  # multiple dataloaders provided, using second one for contrastive samples
            A = batch[0]
            B = batch[1]
        else:  # only one dataloader provided, using same samples for constrative samples
            A = batch
            B = valmap(lambda z: torch.roll(z, 1, dims = 0), A)

        # Concatenate positive samples and negative (contrastive) examples
        x = A
        z = {}
        for key in B:
            z[key] = torch.cat([A[key], B[key]])

        num_pos = len(list(x.values())[0])          # Number of positive examples
        num_neg = len(list(z.values())[0])-num_pos  # Number of negative examples

        logratios = self._logratios(x, z)  # Generates concatenated flattened list of all estimated log ratios
        y = torch.zeros_like(logratios)
        y[:num_pos, ...] = 1
        pos_weight = torch.ones_like(logratios[0])*num_neg/num_pos
        loss = F.binary_cross_entropy_with_logits(logratios, y, reduction = 'none', pos_weight = pos_weight)
        num_ratios = loss.shape[1]
        loss = loss.sum()/num_neg  # Calculates batched-averaged loss
        return loss - 2*np.log(2.)*num_ratios
    
    def training_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log("train_loss", loss, on_step = True, on_epoch = False)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized = False)
        self.log("test_loss", loss, on_epoch = True, on_step = False)
        return loss
    
    def predict_step(self, batch, *args, **kwargs):
        A = batch[0]
        B = batch[1]
        return self(A, B)


class SwyftTrainer(pl.Trainer):
    """Training of SwyftModule, a thin layer around lightning.Trainer."""
    def infer(self, model, A, B, return_sample_ratios = True, batch_size = 1024):
        """Run through model in inference mode.

        Args:
            A: Sample, Samples, or dataloader for samples A.
            B: Sample, Samples, or dataloader for samples B.
            batch_size: batch_size used for Samples provided.

        Returns:
            Concatenated network output
        """
        if isinstance(A, Sample):
            dl1 = Samples({k: [v] for k, v in A.items()}).get_dataloader(batch_size = 1)
        elif isinstance(A, Samples):
            dl1 = A.get_dataloader(batch_size = batch_size)
        else:
            dl1 = A
        if isinstance(B, Sample):
            dl2 = Samples({k: [v] for k, v in B.items()}).get_dataloader(batch_size = 1)
        elif isinstance(B, Samples):
            dl2 = B.get_dataloader(batch_size = batch_size)
        else:
            dl2 = B
        dl = CombinedLoader([dl1, dl2], mode = 'max_size_cycle')
        ratio_batches = self.predict(model, dl)
        if return_sample_ratios:
            if isinstance(ratio_batches[0], dict):
                keys = ratio_batches[0].keys()
                d = {k: LogRatioSamples(
                        torch.cat([r[k].logratios for r in ratio_batches]),
                        torch.cat([r[k].params for r in ratio_batches]),
                        ratio_batches[0][k].parnames
                        ) for k in keys if k[:4] != "aux_"
                    }
                return SampleRatios(**d)
            elif isinstance(ratio_batches[0], list) or isinstance(ratio_batches[0], tuple):
                d = [LogRatioSamples(
                        torch.cat([r[i].logratios for r in ratio_batches]),
                        torch.cat([r[i].params for r in ratio_batches]),
                        ratio_batches[0][i].parnames
                        ) for i in range(len(ratio_batches[0]))
                        if hasattr(ratio_batches[0][i], 'logratios')  # Should we better check for Ratio class?
                    ]
                return d
            else:
                d = LogRatioSamples(
                        torch.cat([r.logratios for r in ratio_batches]),
                        torch.cat([r.params for r in ratio_batches]),
                        ratio_batches[0].parnames
                        ) 
                return d
        else:
            return ratio_batches

    def test_coverage(self, model, A, B, batch_size = 1024):
        """Estimate empirical mass.

        Args:
            model: network
            A: truth samples
            B: prior samples
            batch_size: batch sized used during network evaluation

        Returns:
            Dict of CoverageSamples objects.
        """
        print("WARNING: This estimates the mass of highest-likelihood intervals.")
        repeat = len(B)//batch_size + (len(B)%batch_size>0)
        pred0 = self.infer(model, A.get_dataloader(batch_size=32), A.get_dataloader(batch_size=32))
        pred1 = self.infer(model, A.get_dataloader(batch_size=1, repeat = repeat), B.get_dataloader(batch_size = batch_size))

        def get_pms(p0, p1):
            n0 = len(p0)
            ratios = p1.logratios.reshape(n0, -1, *p1.logratios.shape[1:])  # (n_examples, n_samples_per_example, *per_event_ratio_shape)
            vs = []
            ms = []
            for i in range(n0):
                ratio0 = p0.logratios[i]
                value0 = p0.params[i]
                m = _calc_mass(ratio0, ratios[i])
                vs.append(value0)
                ms.append(m)
            masses = torch.stack(ms, dim = 0)
            params = torch.stack(vs, dim = 0)
            out = CoverageSamples(params, masses, p0.parnames)
            return out

        if isinstance(pred0, tuple):
            out = tuple([get_pms(pred0[i], pred1[i]) for i in range(len(pred0))])
        elif isinstance(pred0, list):
            out = [get_pms(pred0[i], pred1[i]) for i in range(len(pred0))]
        elif isinstance(pred0, dict):
            out = {k: get_pms(pred0[k], pred1[k]) for k in pred0.keys()}
        else:
            out = get_pms(pred0, pred1)

        return out
    

@dataclass
class CoverageSamples:
    """Handles estimated probability masses from coverage samples."""
    params: torch.Tensor
    prob_masses: torch.Tensor
    parnames: np.array

    def _get_matching_masses(self, *args):
        for i, pars in enumerate(self.parnames):
            if set(pars) == set(args):
                return self.prob_masses[:,i]
        return None

    def estimate_coverage(self, params, z_max = 3.5, bins = 50):
        """Estimate expected coverage of credible intervals.

        Args:
            z_max: upper limit (default 3.5)
            bins (int): number of bins used when tabulating z-score

        Returns:
            Array (bins, 4): [nominal z, empirical z, low_err empirical z, hi_err empirical z]
        """
        m = self._get_matching_masses(params)
        if m is None:
            raise SwyftParameterError("Requested parameters not available:", params)
        z0, z1, z2 = get_empirical_z_score(m, z_max, bins, interval_z_score = 1.0)
        z0 = np.tile(z0, (*z1.shape[:-1], 1))
        z0 = np.reshape(z0, (*z0.shape, 1))
        z1 = z1.reshape(*z1.shape, 1)
        z = np.concatenate([z0, z1, z2], axis=-1)
        return z


@dataclass
class LogRatioSamples:
    r"""Dataclass for storing samples of estimated log-ratio values in Swyft.
    
    Args:
        logratios: Estimated log-ratios, :math:`(\text{minibatch}, *\text{logratios_shape})`
        params: Corresponding parameter valuess, :math:`(\text{minibatch}, *\text{logratios_shape}, *\text{params_shape})`
	parnames: Array of parameter names, :math:`(*\text{logratios_shape})`
	metadata: Optional meta-data from inference network etc.
    """
    logratios: torch.Tensor
    params: torch.Tensor
    parnames: np.array
    metadata: dict[str, Any] = field(default_factory = dict)

#    @property
#    def ratios(self):
#        print("WARNING: 'ratios' deprecated")
#        return self.logratios

#    @property
#    def values(self):
#        print("WARNING: 'values' deprecated")
#        return self.params
    
    def __len__(self):
        """Returns number of stored ratios (minibatch size)."""
        assert len(self.params) == len(self.logratios), "Inconsistent Ratios"
        return len(self.params)

#    @property
#    def weights(self):
#        print("WARNING: weights is deprecated.")
#        return self._get_weights(normalize = True)

#    @property
#    def unnormalized_weights(self):
#        print("WARNING: unnormalized_weights is deprecated.")
#        return self._get_weights(normalize = False)

#    def _get_weights(self, normalize: bool = False):
#        """Calculate weights based on ratios.
#
#        Args:
#            normalize: If true, normalize weights to sum to one.  If false, return weights = exp(logratios).
#        """
#        logratios = self.logratios
#        if normalize:
#            logratio_max = logratios.max(axis=0).values
#            weights = torch.exp(logratios-logratio_max)
#            weights_total = weights.sum(axis=0)
#            weights = weights/weights_total*len(weights)
#        else:
#            weights = torch.exp(logratios)
#        return weights
    
#    def sample(self, N, replacement = True):
#        """Subsample params based on normalized weights.
#
#        Args:
#            N: Number of samples to generate
#            replacement: Sample with replacement.  Default is true, which corresponds to generating samples from the posterior.
#
#        Returns:
#            Tensor with samples (n_samples, ..., n_param_dims)
#        """
#        print("WARNING: sample method is deprecated.")
#        weights = self._get_weights(normalized = True)
#        if not replacement and N > len(self):
#            N = len(self)
#        samples = weights_sample(N, self.params, weights, replacement = replacement)
#        return samples


# Utilitiy

def _collection_mask(coll, mask_fn):
    def mask(item):
        if isinstance(item, list) or isinstance(item, tuple) or isinstance(item, dict):
            return True
        return mask_fn(item)

    if isinstance(coll, list):
        return [_collection_mask(item, mask_fn) for item in coll if mask(item)]
    elif isinstance(coll, tuple):
        return tuple([_collection_mask(item, mask_fn) for item in coll if mask(item)])
    elif isinstance(coll, dict):
        return {k: _collection_mask(item, mask_fn) for k, item in coll.items() if mask(item)}
    else:
        return coll if mask(coll) else None

def _collection_map(coll, map_fn):
    if isinstance(coll, list):
        return [_collection_map(item, map_fn) for item in coll]
    elif isinstance(coll, tuple):
        return tuple([_collection_map(item, map_fn) for item in coll])
    elif isinstance(coll, dict):
        return {k: _collection_map(item, map_fn) for k, item in coll.items()}
    else:
        return map_fn(coll)


def _collection_select(coll, err, fn, *args, **kwargs):
    if isinstance(coll, list):
        for item in coll:
            try:
                return _collection_select(item, err, fn, *args, **kwargs)
            except SwyftParameterError:
                pass
    elif isinstance(coll, tuple):
        for item in coll:
            try:
                return _collection_select(item, err, fn, *args, **kwargs)
            except SwyftParameterError:
                pass
    elif isinstance(coll, dict):
        for item in coll.values():
            try:
                return _collection_select(item, err, fn, *args, **kwargs)
            except SwyftParameterError:
                pass
    else:
        try:
            bar = getattr(coll, fn) if fn else coll
            return bar(*args, **kwargs)
        except SwyftParameterError:
            pass
    raise SwyftParameterError(err)


# Convenience

def estimate_coverage(coverage_samples, params, z_max = 3.5, bins = 50):
    return _collection_select(coverage_samples, "Requested parameters not available: %s"%(params,),
            "estimate_coverage", params, z_max = z_max, bins = bins)


def get_weighted_samples(lrs_coll, params: Union[str, Sequence[str]]):
    """Returns weighted samples for particular parameter combination.

    Args:
        params: (List of) parameter names

    Returns:
        (torch.Tensor, torch.Tensor): Parameter and weight tensors
    """
    params = params if isinstance(params, list) else [params]
    if not(isinstance(lrs_coll, list) or isinstance(lrs_coll, tuple)):
        lrs_coll = [lrs_coll]
    for l in lrs_coll:
        for i, pars in enumerate(l.parnames):
            if all(x in pars for x in params):
                idx = [list(pars).index(x) for x in params]
                params = l.params[:,i, idx]
                weights = _get_weights(l.logratios, normalize = True)[:,i]
                return params, weights
    raise SwyftParameterError("Requested parameters not available:", *params)

def _calc_mass(r0, r):
    p = torch.exp(r - r.max(axis=0).values)
    p /= p.sum(axis=0)
    m = r > r0
    return (p*m).sum(axis=0)

#def weights_sample(N, values, weights, replacement = True):
#    """Weight-based sampling with or without replacement."""
#    sw = weights.shape
#    sv = values.shape
#    assert sw == sv[:len(sw)], "Overlapping left-handed weights and values shapes do not match: %s vs %s"%(str(sv), str(sw))
#    
#    w = weights.view(weights.shape[0], -1)
#    idx = torch.multinomial(w.T, N, replacement = replacement).T
#    si = tuple(1 for _ in range(len(sv)-len(sw)))
#    idx = idx.view(N, *sw[1:], *si)
#    idx = idx.expand(N, *sv[1:])
#    
#    samples = torch.gather(values, 0, idx)
#    return samples

def best_from_yaml(filepath):
    """Get best model from tensorboard log. Useful for reloading trained networks.

    Args:
        filepath: Filename of yaml file (assumed to be saved with to_yaml from ModelCheckpoint)

    Returns:
        path to best model
    """
    try:
        with open(filepath) as f:
            best_k_models = yaml.load(f, Loader = yaml.FullLoader)    
    except FileNotFoundError:
        return None
    val_loss = np.inf
    path = None
    for k, v in best_k_models.items():
        if v < val_loss:
            path = k
            val_loss = v
    return path

def _pdf_from_weighted_samples(v, w, bins = 50, smooth = 0, v_aux = None):
    """Take weighted samples and turn them into a pdf on a grid.
    
    Args:
        bins
    """
    ndim = v.shape[-1]
    if v_aux is None:
        return weighted_smoothed_histogramdd(v, w, bins = bins, smooth = smooth)
    else:
        h, xy = weighted_smoothed_histogramdd(v_aux, None, bins = bins, smooth = smooth)
        if ndim == 2:
            X, Y = np.meshgrid(xy[:,0], xy[:,1])
            n = len(xy)
            out = scipy.interpolate.griddata(v, w, (X.flatten(), Y.flatten()), method = 'cubic', fill_value = 0.).reshape(n, n)
            return out, xy
        elif ndim == 1:
            out = scipy.interpolate.griddata(v[:,0], w, xy[:,0], method = 'cubic', fill_value = 0.)
            return out, xy
        else:
            raise KeyError("Not supported")
    
def weighted_smoothed_histogramdd(v, w, bins = 50, smooth = 0):
    ndim = v.shape[-1]
    if ndim == 1:
        low, upp = v.min(), v.max()
        h =  torchist.histogramdd(v, bins, weights = w, low = low, upp = upp)
        h /= len(v)*(upp-low)/bins
        edges = torch.linspace(low, upp, bins+1)
        x = (edges[1:] + edges[:-1])/2
        if smooth > 0:
            h = torch.tensor(gaussian_filter1d(h, smooth))
        return h, x.unsqueeze(-1)
    elif ndim == 2:
        low = v.min(axis=0).values
        upp = v.max(axis=0).values
        h = torchist.histogramdd(v, bins = bins, weights = w, low = low, upp = upp)
        h /= len(v)*(upp[0]-low[0])*(upp[1]-low[1])/bins**2
        x = torch.linspace(low[0], upp[0], bins+1)
        y = torch.linspace(low[1], upp[1], bins+1)
        x = (x[1:] + x[:-1])/2
        y = (y[1:] + y[:-1])/2
        xy = torch.vstack([x, y]).T
        if smooth > 0:
            h = torch.tensor(gaussian_filter(h*1., smooth))
        return h, xy

def get_pdf(lrs_coll, params: Union[str, Sequence[str]], aux = None, bins: int = 50, smooth: float = 0.):
    """Generate binned PDF based on input 

    Args:
        lrs_coll: Collection of LogRatioSamples objects.
        params: Parameter names
        bins: Number of bins
        smooth: Apply Gaussian smoothing

    Returns:
        np.array, np.array: Returns densities and parameter grid.
    """
    z, w = get_weighted_samples(lrs_coll, params)
    if aux is not None:
        z_aux, _ = get_weighted_samples(aux, params)
    else:
        z_aux = None
    return _pdf_from_weighted_samples(z, w, bins = bins, smooth = smooth, v_aux = z_aux)


def _get_weights(logratios, normalize: bool = False):
    """Calculate weights based on ratios.

    Args:
        normalize: If true, normalize weights to sum to one.  If false, return weights = exp(logratios).
    """
    if normalize:
        logratio_max = logratios.max(axis=0).values
        weights = torch.exp(logratios-logratio_max)
        weights_total = weights.sum(axis=0)
        weights = weights/weights_total*len(weights)
    else:
        weights = torch.exp(logratios)
    return weights
    
