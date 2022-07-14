from abc import abstractmethod
import math
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
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split
import pytorch_lightning as pl
from tqdm import tqdm
import swyft
import swyft.utils
from swyft.inference.marginalratioestimator import get_ntrain_nvalid
import yaml

import zarr
import fasteners
from dataclasses import dataclass
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

from swyft.networks.standardization import OnlineStandardizingLayer


class SwyftModule(pl.LightningModule):
    """Handles training of ratio estimators."""
    def __init__(self, lr = 1e-3, lrs_factor = 0.1, lrs_patience = 5):
        """Instantiates SwyftModule.

        Args:
            lr: learning rate
            lrs_factor: learning rate decay
            lrs_patience: learning rate decay patience
        """
        super().__init__()
        self.save_hyperparameters()
        self._predict_condition_x = {}
        self._predict_condition_z = {}

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/KL-div": -1, "hp/JS-div": -1})
        
    def on_train_end(self):
        for cb in self.trainer.callbacks:
            if isinstance(cb, pl.callbacks.model_checkpoint.ModelCheckpoint):
                cb.to_yaml()
      
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.lr)
        lr_scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.hparams.lrs_factor, patience=self.hparams.lrs_patience), "monitor": "val_loss"}
        return dict(optimizer = optimizer, lr_scheduler = lr_scheduler)

    def _log_ratios(self, x, z):
        out = self(x, z)
        out = {k: v for k, v in out.items() if k[:4] != 'aux_'}
        log_ratios = torch.cat([val.ratios.flatten(start_dim = 1) for val in out.values()], dim=1)
        return log_ratios
    
    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized = False)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def _calc_loss(self, batch, randomized = True):
        """Calcualte batch-averaged loss summed over ratio estimators.

        Note: The expected loss for an untrained classifier (with f = 0) is subtracted.  The initial loss is hence usually close to zero.
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

        log_ratios = self._log_ratios(x, z)  # Generates concatenated flattened list of all estimated log ratios
        y = torch.zeros_like(log_ratios)
        y[:num_pos, ...] = 1
        pos_weight = torch.ones_like(log_ratios[0])*num_neg/num_pos
        loss = F.binary_cross_entropy_with_logits(log_ratios, y, reduction = 'none', pos_weight = pos_weight)
        num_ratios = loss.shape[1]
        loss = loss.sum()/num_neg  # Calculates batched-averaged loss
        return loss - 2*np.log(2.)*num_ratios
    
    def _calc_KL(self, batch, batch_idx):
        x = batch
        z = batch
        log_ratios = self._log_ratios(x, z)
        nbatch = len(log_ratios)
        loss = -log_ratios.sum()/nbatch
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized = False)
        lossKL = self._calc_KL(batch, batch_idx)
        self.log("hp/JS-div", loss)
        #self.log("hp_metric", loss)
        self.log("hp/KL-div", lossKL)
        return loss
    
    def _set_predict_conditions(self, condition_x, condition_z):
        self._predict_condition_x = {k: v.unsqueeze(0) for k, v in condition_x.items()}
        self._predict_condition_z = {k: v.unsqueeze(0) for k, v in condition_z.items()}
        
    def set_conditions(self, conditions):
        self._predict_condition_x = conditions
    
    def predict_step(self, batch, *args, **kwargs):
        A = batch[0]
        B = batch[1]
        return self(A, B)


class SwyftTrainer(pl.Trainer):
    """Training of SwyftModule, a thin layer around lightning.Trainer."""
    def infer(self, model, A, B, return_sample_ratios = True):
        """Run through model in inference mode.

        Args:
            A: sample or dataloader for samples A.
            B: sample or dataloader for samples B.

        Returns:
            Concatenated network output
        """
        if isinstance(A, dict):
            dl1 = Samples({k: [v] for k, v in A.items()}).get_dataloader(batch_size = 1)
        else:
            dl1 = A
        if isinstance(B, dict):
            dl2 = Samples({k: [v] for k, v in B.items()}).get_dataloader(batch_size = 1)
        else:
            dl2 = B
        dl = CombinedLoader([dl1, dl2], mode = 'max_size_cycle')
        ratio_batches = self.predict(model, dl)
        if return_sample_ratios:
            keys = ratio_batches[0].keys()
            d = {k: Ratios(
                    torch.cat([r[k].values for r in ratio_batches]),
                    torch.cat([r[k].ratios for r in ratio_batches])
                    ) for k in keys if k[:4] != "aux_"
                }
            return SampleRatios(**d)
        else:
            return ratio_batches
    
    def estimate_mass(self, model, A, B, batch_size = 1024):
        """Estimate empirical mass.

        Args:
            model: network
            A: truth samples
            B: prior samples
            batch_size: batch sized used during network evaluation

        Returns:
            Dict of PosteriorMass objects.
        """
        repeat = len(B)//batch_size + (len(B)%batch_size>0)
        pred0 = self.infer(model, A.get_dataloader(batch_size=32), A.get_dataloader(batch_size=32))
        pred1 = self.infer(model, A.get_dataloader(batch_size=1, repeat = repeat), B.get_dataloader(batch_size = batch_size))
        n0 = len(pred0)
        out = {}
        for k, v in pred1.items():
            ratios = v.ratios.reshape(n0, -1, *v.ratios.shape[1:])
            vs = []
            ms = []
            for i in range(n0):
                ratio0 = pred0[k].ratios[i]
                value0 = pred0[k].values[i]
                m = calc_mass(ratio0, ratios[i])
                vs.append(value0)
                ms.append(m)
            masses = torch.stack(ms, dim = 0)
            values = torch.stack(vs, dim = 0)
            out[k] = PosteriorMass(values, masses)
        return out


@dataclass
class PosteriorMass:
    """Handles masses and the corresponding parameter values."""
    values: None
    masses: None

@dataclass
class Ratios:
    """Handles ratios and the corresponding parameter values.
    
    A dictionary of Ratios is expected to be returned by ratio estimation networks.

    Args:
        values: tensor of values for which the ratios were estimated, (nbatch, *shape_ratios, *shape_params)
        ratios: tensor of estimated ratios, (nbatch, *shape_ratios)
    """
    values: torch.Tensor
    ratios: torch.Tensor
    metadata: dict = field(default_factory = dict)
    
    def __len__(self):
        """Number of stored ratios."""
        assert len(self.values) == len(self.ratios), "Inconsistent Ratios"
        return len(self.values)
    
    def weights(self, normalize = False):
        """Calculate weights based on ratios.

        Args:
            normalize: If true, normalize weights to sum to one.  If false, return weights = exp(ratios).
        """
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
        """Subsample values based on normalized weights.

        Args:
            N: Number of samples to generate
            replacement: Sample with replacement.  Default is true, which corresponds to generating samples from the posterior.
        """
        weights = self.weights(normalize = True)
        if not replacement and N > len(self):
            N = len(self)
        samples = weights_sample(N, self.values, weights, replacement = replacement)
        return samples


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

