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
)
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import random_split

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.trainer.supporters import CombinedLoader

import yaml

from swyft.lightning.samples import *
from swyft.plot.mass import get_empirical_z_score

import scipy
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import torchist


class SwyftModule(pl.LightningModule):
    def __init__(self, 
            lr: float = 1e-3,
            lrs_factor: float = 0.1,
            lr_monitor: bool = True,
            early_stopping: bool = True,
            early_stopping_patience: int = 3,
            lrs_patience: int = 5
            ):
        r"""

        Handles training of logratio estimators.

        The main way to use a ``SwyftModule''.

        Arguments:

            lr: The initial learning rate.
            lrs_factor: The learning rate decay.
            lrs_patience: The learning rate decay patience.
        """
        super().__init__()
        self._swyft_module_config = dict(lr = lr, lrs_factor = lrs_factor, lr_monitor = lr_monitor,
                lrs_patience = lrs_patience, early_stopping = early_stopping,
                early_stopping_patience = early_stopping_patience)
        #self.save_hyperparameters()
#        self._predict_condition_x = {}
#        self._predict_condition_z = {}

    def on_train_start(self):
        pass
        #self.logger.log_hyperparams(self.hparams, {"hp/KL-div": -1, "hp/JS-div": -1})

    def configure_callbacks(self):
        callbacks = []
        if self._swyft_module_config["lr_monitor"]:
            callbacks.append(LearningRateMonitor())
        if self._swyft_module_config["early_stopping"]:
            early_stop = EarlyStopping(monitor="val_loss", min_delta=0.0,
                    patience=self._swyft_module_config["early_stopping_patience"],
                    verbose=False, mode="min")
            callbacks.append(early_stop)
        return callbacks
        
    def on_train_end(self):
        pass

        # TODO: Convenience
#        for cb in self.trainer.callbacks:
#            if isinstance(cb, pl.callbacks.model_checkpoint.ModelCheckpoint):
#                cb.to_yaml()  # Saves path of best_k_models to yaml file


    def configure_optimizers(self):
        lr = self._swyft_module_config["lr"]
        lrs_patience = self._swyft_module_config["lrs_patience"]
        lrs_factor = self._swyft_module_config["lrs_factor"]

        optimizer = torch.optim.Adam(self.parameters(), lr)
        lr_scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lrs_factor, patience=lrs_patience), "monitor": "val_loss"}
        return dict(optimizer = optimizer, lr_scheduler = lr_scheduler)
      
#    def configure_optimizers(self):
#        return default_optimizers(self.parameters())
#        optimizer = torch.optim.Adam(self.parameters(), self.hparams.lr)
#
#        # TODO: Convenience
#        lr_scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
#            optimizer, factor=self.hparams.lrs_factor, patience=self.hparams.lrs_patience), "monitor": "val_loss"}
#
#        return dict(optimizer = optimizer, lr_scheduler = lr_scheduler)

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
    
    def _calc_KL(self, batch, batch_idx):
        # TODO: Convenience
        x = batch
        z = batch
        logratios = self._logratios(x, z)
        nbatch = len(logratios)
        loss = -logratios.sum()/nbatch
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log("train_loss", loss, on_step = True, on_epoch = False)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized = False)
        self.log("test_loss", loss, on_epoch = True, on_step = False)

        #self.log("hp_metric", loss, on_step = False, on_epoch = True)

        #self.log("hp/JS-div", loss)
        #lossKL = self._calc_KL(batch, batch_idx)
        #self.log("hp/KL-div", lossKL)

        return loss
    
#    def _set_predict_conditions(self, condition_x, condition_z):
#        self._predict_condition_x = {k: v.unsqueeze(0) for k, v in condition_x.items()}
#        self._predict_condition_z = {k: v.unsqueeze(0) for k, v in condition_z.items()}
        
#    def set_conditions(self, conditions):
#        self._predict_condition_x = conditions
    
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
                        torch.cat([r[k].params for r in ratio_batches]),
                        torch.cat([r[k].logratios for r in ratio_batches]),
                        ratio_batches[0][k].parnames
                        ) for k in keys if k[:4] != "aux_"
                    }
                return SampleRatios(**d)
            elif isinstance(ratio_batches[0], list) or isinstance(ratio_batches[0], tuple):
                d = [LogRatioSamples(
                        torch.cat([r[i].params for r in ratio_batches]),
                        torch.cat([r[i].logratios for r in ratio_batches]),
                        ratio_batches[0][i].parnames
                        ) for i in range(len(ratio_batches[0]))
                        if hasattr(ratio_batches[0][i], 'logratios')  # Should we better check for Ratio class?
                    ]
                return d
            else:
                d = LogRatioSamples(
                        torch.cat([r.params for r in ratio_batches]),
                        torch.cat([r.logratios for r in ratio_batches]),
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
                m = calc_mass(ratio0, ratios[i])
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

#    def estimate_coverage_dict(self):
#        out = {}
#        for i, pars in enumerate(self.parnames):
#            pars = tuple(pars)
#            m = self.prob_masses[:,i]
#            z0, z1, z2 = get_empirical_z_score(m, 3.5, 50, 1.0)
#            out[pars] = np.array([np.interp([1.0, 2.0, 3.0], z0, z2[:,0]),
#                np.interp([1.0, 2.0, 3.0], z0, z1), np.interp([1.0, 2.0, 3.0],
#                    z0, z2[:,1])]).T
#        return out

    def estimate_coverage(self, *args, z_max = 3.5, bins = 50):
        """Estimate expected coverage of credible intervals.

        Args:
            z_max: upper limit (default 3.5)
            bins (int): number of bins used when tabulating z-score

        Returns:
            Array (bins, 4): [nominal z, empirical z, low_err empirical z, hi_err empirical z]
        """
        m = self._get_matching_masses(*args)
        if m is None:
            raise SwyftParameterError("Requested parameters not available:", *args)
        z0, z1, z2 = get_empirical_z_score(m, z_max, bins, interval_z_score = 1.0)
        z0 = np.tile(z0, (*z1.shape[:-1], 1))
        z0 = np.reshape(z0, (*z0.shape, 1))
        z1 = z1.reshape(*z1.shape, 1)
        z = np.concatenate([z0, z1, z2], axis=-1)
        return z

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

def estimate_coverage(coverage_samples, *args, z_max = 3.5, bins = 50):
    return _collection_select(coverage_samples, "Requested parameters not available: %s"%(args,),
            "estimate_coverage", *args, z_max = z_max, bins = bins)

@dataclass
class LogRatioSamples:
    """Handles logratios and the corresponding parameter values.
    
    A dictionary of Ratios is expected to be returned by ratio estimation networks.

    Args:
        values: tensor of values for which the ratios were estimated, (nbatch, *shape_ratios, *shape_params)
        ratios: tensor of estimated ratios, (nbatch, *shape_ratios)
    """
    params: torch.Tensor
    logratios: torch.Tensor
    parnames: np.array
    metadata: dict = field(default_factory = dict)

    @property
    def ratios(self):
        print("WARNING: 'ratios' deprecated")
        return self.logratios

    @property
    def values(self):
        print("WARNING: 'values' deprecated")
        return self.params
    
    def __len__(self):
        """Number of stored ratios."""
        assert len(self.params) == len(self.logratios), "Inconsistent Ratios"
        return len(self.params)

    @property
    def weights(self):
        return self._get_weights(normalize = True)

    @property
    def unnormalized_weights(self):
        return self._get_weights(normalize = False)
    
    def _get_weights(self, normalize = False):
        """Calculate weights based on ratios.

        Args:
            normalize: If true, normalize weights to sum to one.  If false, return weights = exp(ratios).
        """
        logratios = self.logratios
        if normalize:
            logratio_max = logratios.max(axis=0).values
            weights = torch.exp(logratios-logratio_max)
            weights_total = weights.sum(axis=0)
            weights = weights/weights_total*len(weights)
        else:
            weights = torch.exp(logratios)
        return weights
    
    def sample(self, N, replacement = True):
        """Subsample params based on normalized weights.

        Args:
            N: Number of samples to generate
            replacement: Sample with replacement.  Default is true, which corresponds to generating samples from the posterior.

        Returns:
            Tensor with samples (n_samples, ..., n_param_dims)
        """
        weights = self.weights
        if not replacement and N > len(self):
            N = len(self)
        samples = weights_sample(N, self.params, weights, replacement = replacement)
        return samples

def get_weighted_samples(loglike, *args):
    """Returns weighted samples for particular parameter combination.

    Args:
        *args: Parameter names

    Returns:
        (parameter tensor, weight tensor)
    """
    if not(isinstance(loglike, list) or isinstance(loglike, tuple)):
        loglike = [loglike]
    for l in loglike:
        for i, pars in enumerate(l.parnames):
            if all(x in pars for x in args):
                idx = [list(pars).index(x) for x in args]
                return l.params[:,i, idx], l.weights[:,i]
    raise SwyftParameterError("Requested parameters not available:", *args)

class SwyftParameterError(Exception):
    pass


def calc_mass(r0, r):
    p = torch.exp(r - r.max(axis=0).values)
    p /= p.sum(axis=0)
    m = r > r0
    return (p*m).sum(axis=0)

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


def tensorboard_config(save_dir = "./lightning_logs", name = None, version = None, patience = 3):
    """Generates convenience configuration for Trainer object.

    Args:
        save_dir: Save-directory for tensorboard logs
        name: tensorboard logs name
        version: tensorboard logs version
        patience: early-stopping patience

    Returns:
        Configuration dictionary
    """
    tbl = pl_loggers.TensorBoardLogger(save_dir = save_dir, name = name, version = version, default_hp_metric = False)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=patience, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    return dict(logger = tbl, callbacks = [lr_monitor, early_stop_callback, checkpoint_callback])

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

def get_best_model(tbl):
    """Get best model from tensorboard log. Useful for reloading trained networks.

    Args:
        tbl: Tensorboard log instance

    Returns:
        path to best model
    """
    try:
        with open(tbl.experiment.get_logdir()+"/checkpoints/best_k_models.yaml") as f:
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

def pdf_from_weighted_samples(v, w, bins = 50, smooth = 0, v_aux = None):
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
        edges = torch.linspace(low, upp, bins+1)
        x = (edges[1:] + edges[:-1])/2
        if smooth > 0:
            h = torch.tensor(gaussian_filter1d(h, smooth))
        return h, x.unsqueeze(-1)
    elif ndim == 2:
        low = v.min(axis=0).values
        upp = v.max(axis=0).values
        h = torchist.histogramdd(v, bins = bins, weights = w, low = low, upp = upp)
        x = torch.linspace(low[0], upp[0], bins+1)
        y = torch.linspace(low[1], upp[1], bins+1)
        x = (x[1:] + x[:-1])/2
        y = (y[1:] + y[:-1])/2
        xy = torch.vstack([x, y]).T
        if smooth > 0:
            h = torch.tensor(gaussian_filter(h*1., smooth))
        return h, xy


    
    
def get_pdf(loglike, *args, aux = None, bins = 50, smooth = 0):
    z, w = get_weighted_samples(loglike, *args)
    if aux is not None:
        z_aux, _ = get_weighted_samples(aux, *args)
    else:
        z_aux = None
    return pdf_from_weighted_samples(z, w, bins = bins, smooth = smooth, v_aux = z_aux)


#def default_optimizers(parameters, lr = 1e-3, lrs_factor = 0.1, lrs_patience = 5):
#    optimizer = torch.optim.Adam(parameters, lr)
#
#    lr_scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
#        optimizer, factor=lrs_factor, patience=lrs_patience), "monitor": "val_loss"}
#
#    return dict(optimizer = optimizer, lr_scheduler = lr_scheduler)
#
#def default_callbacks(
#        lr_monitor = True,
#        early_stopping = True, 
#        early_stopping_patience = 2, 
#        model_checkpoint = True,
#        save_last = True, save_top_k = 1, save_weights_only = False):
#    """Generates convenience configuration for Trainer object.
#
#    Args:
#        save_dir: Save-directory for tensorboard logs
#        name: tensorboard logs name
#        version: tensorboard logs version
#        patience: early-stopping patience
#
#    Returns:
#        Configuration dictionary
#    """
#    callbacks = []
#    if lr_monitor:
#        callbacks.append(LearningRateMonitor())
#    if early_stopping:
#        early_stop = EarlyStopping(monitor="val_loss", min_delta=0.0,
#                patience=early_stopping_patience, verbose=False, mode="min")
#        callbacks.append(early_stop)
#    if model_checkpoint:
#        checkpoint = ModelCheckpoint(monitor="val_loss", save_last = save_last, save_top_k = save_top_k, save_weights_only = save_weights_only)
#        callbacks.append(checkpoint)
#    return callbacks
