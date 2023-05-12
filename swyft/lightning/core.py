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

try:
    from pytorch_lightning.trainer.supporters import CombinedLoader
except ImportError:
    from pytorch_lightning.utilities import CombinedLoader


# from pytorch_lightning.cli import instantiate_class

import yaml

from swyft.lightning.data import *
from swyft.plot.mass import get_empirical_z_score
from swyft.lightning.utils import (
    OptimizerInit,
    AdamOptimizerInit,
    SwyftParameterError,
    _collection_mask,
    _collection_flatten,
)

import scipy
from scipy.ndimage import gaussian_filter1d, gaussian_filter

# import torchist


#############
# SwyftModule
#############


class SwyftModule(pl.LightningModule):
    r"""This is the central Swyft LightningModule for handling the training of logratio estimators.

    Derived classes are supposed to overwrite the `forward` method in order to implement specific inference tasks.

    The attribute `optimizer_init` points to the optimizer initializer (default is `AdamOptimizerInit`).

    .. note::

       The forward method takes as arguments the sample batches `A` and `B`,
       which typically include all sample variables.  Joined samples correspond to
       A=B, whereas marginal samples correspond to samples A != B.

    Example usage:

    .. code-block:: python

       class MyNetwork(swyft.SwyftModule):
           def __init__(self):
               self.optimizer_init = AdamOptimizerInit(lr = 1e-4)
               self.mlp = swyft.LogRatioEstimator_1dim(4, 4)

           def forward(A, B);
               x = A['x']
               z = A['z']
               logratios = self.mlp(x, z)
               return logratios
    """

    def __init__(self):
        super().__init__()
        self.optimizer_init = AdamOptimizerInit()

    def configure_optimizers(self):
        return self.optimizer_init(self.parameters())

    def _get_logratios(self, out):
        if isinstance(out, dict):
            out = {k: v for k, v in out.items() if k[:4] != "aux_"}
            logratios = torch.cat(
                [val.logratios.flatten(start_dim=1) for val in out.values()], dim=1
            )
        elif isinstance(out, list) or isinstance(out, tuple):
            out = [v for v in out if hasattr(v, "logratios")]
            if out == []:
                return None
            logratios = torch.cat(
                [val.logratios.flatten(start_dim=1) for val in out], dim=1
            )
        elif isinstance(out, swyft.LogRatioSamples):
            logratios = out.logratios.flatten(start_dim=1)
        else:
            logratios = None
        return logratios

    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized=False)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def _calc_loss(self, batch, randomized=True):
        """Calcualte batch-averaged loss summed over ratio estimators.

        Note: The expected loss for an untrained classifier (with f = 0) is
        subtracted.  The initial loss is hence usually close to zero.
        """
        if isinstance(
            batch, list
        ):  # multiple dataloaders provided, using second one for contrastive samples
            A = batch[0]
            B = batch[1]
        else:  # only one dataloader provided, using same samples for constrative samples
            A = batch
            B = valmap(lambda z: torch.roll(z, 1, dims=0), A)

        # Concatenate positive samples and negative (contrastive) examples
        x = A
        z = {}
        for key in B:
            z[key] = torch.cat([A[key], B[key]])

        num_pos = len(list(x.values())[0])  # Number of positive examples
        num_neg = len(list(z.values())[0]) - num_pos  # Number of negative examples

        out = self(x, z)  # Evaluate network
        loss_tot = 0

        logratios = self._get_logratios(
            out
        )  # Generates concatenated flattened list of all estimated log ratios
        if logratios is not None:
            y = torch.zeros_like(logratios)
            y[:num_pos, ...] = 1
            pos_weight = torch.ones_like(logratios[0]) * num_neg / num_pos
            loss = F.binary_cross_entropy_with_logits(
                logratios, y, reduction="none", pos_weight=pos_weight
            )
            num_ratios = loss.shape[1]
            loss = loss.sum() / num_neg  # Calculates batched-averaged loss
            loss = loss - 2 * np.log(2.0) * num_ratios
            loss_tot += loss

        aux_losses = self._get_aux_losses(out)
        if aux_losses is not None:
            loss_tot += aux_losses.sum()

        return loss_tot

    def _get_aux_losses(self, out):
        flattened_out = _collection_flatten(out)
        filtered_out = [v for v in flattened_out if isinstance(v, swyft.AuxLoss)]
        if len(filtered_out) == 0:
            return None
        else:
            losses = torch.cat([v.loss.unsqueeze(-1) for v in filtered_out], dim=1)
            return losses

    def training_step(self, batch, batch_idx):
        loss = self._calc_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._calc_loss(batch, randomized=False)
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch, *args, **kwargs):
        A = batch[0]
        B = batch[1]
        return self(A, B)


#################
# LogRatioSamples
#################


@dataclass
class AuxLoss:
    r"""Datacloss for storing aditional loss functions that are minimized during optimization"""
    loss: torch.Tensor
    name: str


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
    metadata: dict = field(default_factory=dict)

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


#########
# Trainer
#########


class SwyftTrainer(pl.Trainer):
    """Base class: pytorch_lightning.Trainer

    It provides training functionality for swyft.SwyftModule. The functionality
    is identical to `pytorch_lightning.Trainer`, see corresponding documentation
    for more details.

    Two additional methods are defined:

    - `infer` for performing parameter inference tasks with a trained network
    - `test_coverage` for performing coverage tests
    """

    def infer(
        self, model, A, B, return_sample_ratios: bool = True, batch_size: int = 1024
    ):
        """Run through model in inference mode.

        Args:
            A: Sample, Samples, or dataloader for samples A.
            B: Sample, Samples, or dataloader for samples B.
            return_sample_ratios: If true (default), return results as collated collection of `LogRatioSamples` objects.  Otherwise, return batches.
            batch_size: batch_size used for Samples provided.

        Returns:
            Concatenated network output
        """
        if isinstance(A, Sample):
            dl1 = Samples({k: [v] for k, v in A.items()}).get_dataloader(batch_size=1)
        elif isinstance(A, Samples):
            dl1 = A.get_dataloader(batch_size=batch_size)
        else:
            dl1 = A
        if isinstance(B, Sample):
            dl2 = Samples({k: [v] for k, v in B.items()}).get_dataloader(batch_size=1)
        elif isinstance(B, Samples):
            dl2 = B.get_dataloader(batch_size=batch_size)
        else:
            dl2 = B
        dl = CombinedLoader([dl1, dl2], mode="max_size_cycle")
        ratio_batches = self.predict(model, dl)
        if return_sample_ratios:
            if isinstance(ratio_batches[0], dict):
                keys = ratio_batches[0].keys()
                d = {
                    k: LogRatioSamples(
                        torch.cat([r[k].logratios for r in ratio_batches]),
                        torch.cat([r[k].params for r in ratio_batches]),
                        ratio_batches[0][k].parnames,
                    )
                    for k in keys
                    if k[:4] != "aux_"
                }
                return d
            elif isinstance(ratio_batches[0], list) or isinstance(
                ratio_batches[0], tuple
            ):
                d = [
                    LogRatioSamples(
                        torch.cat([r[i].logratios for r in ratio_batches]),
                        torch.cat([r[i].params for r in ratio_batches]),
                        ratio_batches[0][i].parnames,
                    )
                    for i in range(len(ratio_batches[0]))
                    if hasattr(
                        ratio_batches[0][i], "logratios"
                    )  # Should we better check for Ratio class?
                ]
                return d
            else:
                d = LogRatioSamples(
                    torch.cat([r.logratios for r in ratio_batches]),
                    torch.cat([r.params for r in ratio_batches]),
                    ratio_batches[0].parnames,
                )
                return d
        else:
            return ratio_batches

    def test_coverage(self, model, A, B, batch_size=1024, logratio_noise=True):
        """Estimate empirical mass.

        Args:
            model: network
            A: truth samples
            B: prior samples
            batch_size: batch sized used during network evaluation
            logratio_noise: Add a small amount of noise to log-ratio estimates, which stabilizes mass estimates for classification tasks.

        Returns:
            Dict of CoverageSamples objects.
        """

        print("WARNING: This estimates the mass of highest-likelihood intervals.")
        repeat = len(B) // batch_size + (len(B) % batch_size > 0)
        pred0 = self.infer(
            model, A.get_dataloader(batch_size=32), A.get_dataloader(batch_size=32)
        )
        pred1 = self.infer(
            model,
            A.get_dataloader(batch_size=1, repeat=repeat),
            B.get_dataloader(batch_size=batch_size),
        )

        def get_pms(p0, p1):
            n0 = len(p0)
            ratios = p1.logratios.reshape(
                n0, -1, *p1.logratios.shape[1:]
            )  # (n_examples, n_samples_per_example, *per_event_ratio_shape)
            vs = []
            ms = []
            for i in range(n0):
                ratio0 = p0.logratios[i]
                value0 = p0.params[i]
                m = _calc_mass(ratio0, ratios[i], add_noise=logratio_noise)
                vs.append(value0)
                ms.append(m)
            masses = torch.stack(ms, dim=0)
            params = torch.stack(vs, dim=0)
            out = CoverageSamples(masses, params, p0.parnames)
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


def _calc_mass(r0, r, add_noise=False):
    if add_noise:
        r = r + torch.rand_like(r) * 1e-3
        r0 = r0 + torch.rand_like(r0) * 1e-3
    p = torch.exp(r - r.max(axis=0).values)
    p /= p.sum(axis=0)
    m = r > r0
    return (p * m).sum(axis=0)


#################
# CoverageSamples
#################


@dataclass
class CoverageSamples:
    r"""Dataclass for storing probability masses samples from coverage tests.

    Args:
        prob_masses: Tensor of probability masses in the range [0, 1], :math:`(\text{minibatch}, *\text{logratios_shape})`
        params: Corresponding parameter valuess, :math:`(\text{minibatch}, *\text{logratios_shape}, *\text{params_shape})`
        parnames: Array of parameter names, :math:`(*\text{logratios_shape})`
    """
    prob_masses: torch.Tensor
    params: torch.Tensor
    parnames: np.array

    def _get_matching_masses(self, parnames):
        parnames = [parnames] if isinstance(parnames, str) else parnames
        for i, pars in enumerate(self.parnames):
            if set(pars) == set(parnames):
                return self.prob_masses[:, i]
        return None

    def estimate_coverage(
        self, parnames: Union[str, Sequence[str]], z_max: float = 3.5, bins: int = 50
    ):
        """Estimate expected coverage of credible intervals on a grid of credibility values.

        Args:
            parnames: Names of parameters
            z_max: upper limit on the credibility level (default 3.5)
            bins (int): number of bins used when tabulating z-score

        Returns:
            np.array (bins, 4): Array columns correspond to [nominal z, empirical z, low_err empirical z, hi_err empirical z]
        """
        m = self._get_matching_masses(parnames)
        if m is None:
            raise SwyftParameterError("Requested parameters not available:", parnames)
        z0, z1, z2 = get_empirical_z_score(m, z_max, bins, interval_z_score=1.0)
        z0 = np.tile(z0, (*z1.shape[:-1], 1))
        z0 = np.reshape(z0, (*z0.shape, 1))
        z1 = z1.reshape(*z1.shape, 1)
        z = np.concatenate([z0, z1, z2], axis=-1)
        return z
