import logging
from typing import Callable, Dict, Optional, Sequence, Union
from warnings import warn

import numpy as np
import torch

import swyft
from swyft.inference.ratios import RatioEstimator
from swyft.inference.train import TrainOptions
from swyft.networks import DefaultHead, DefaultTail
from swyft.types import (
    Array,
    Device,
    MarginalsType,
    ObsType,
    PathType,
    PNamesType,
    PoIType,
    RatiosType,
)
from swyft.utils import tupleize_marginals

log = logging.getLogger(__name__)


class Posteriors:
    """Main inference class.

    Args:
        dataset: Dataset for which we want to perform inference.

    .. note::
        The dataset will be used to extract parameter names (`pnames`), the
        prior and its bound. It will be then set as default dataset for
        training.
    """

    def __init__(self, dataset: "swyft.Dataset") -> None:
        self._pnames = dataset.pnames
        self._trunc_prior = swyft.TruncatedPrior(dataset.prior, bound=dataset.bound)
        self._ratios = {}
        self._dataset = dataset

    @property
    def pnames(self) -> PNamesType:
        """Parameter names. Inherited from dataset."""
        return self._pnames

    def add(
        self,
        marginals: MarginalsType,
        head: Callable[..., "swyft.Module"] = DefaultHead,
        tail: Callable[..., "swyft.Module"] = DefaultTail,
        head_args: dict = {},
        tail_args: dict = {},
        device: Device = "cpu",
    ) -> None:
        """Add marginals.

        Args:
            marginals
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
        """
        marginals = tupleize_marginals(marginals)
        re = RatioEstimator(
            marginals,
            device=device,
            head=head,
            tail=tail,
            head_args=head_args,
            tail_args=tail_args,
        )
        self._ratios[marginals] = re

    def to(
        self, device: Device, marginals: Optional[MarginalsType] = None
    ) -> "Posteriors":
        """Move networks to device.

        Args:
            device: Targeted device.
            marginals: Optional, only move networks related to specific marginals.
        """
        if marginals is not None:
            marginals = tupleize_marginals(marginals)
            self._ratios[marginals].to(device)
        else:
            for _, v in self._ratios.items():
                v.to(device)
        return self

    @property
    def dataset(self) -> "swyft.Dataset":
        """Default training dataset."""
        return self._dataset

    def set_dataset(self, dataset: "swyft.Dataset") -> None:
        """Set default training dataset."""
        self._dataset = dataset

    def train(
        self,
        marginals: MarginalsType,
        batch_size: int = 64,
        validation_size: float = 0.1,
        early_stopping_patience: int = 5,
        max_epochs: int = 30,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: dict = dict(lr=1e-3),
        scheduler: Callable[
            ..., torch.optim.lr_scheduler._LRScheduler
        ] = torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args: dict = dict(factor=0.1, patience=5),
        nworkers: int = 2,
        non_blocking: bool = True,
    ) -> None:
        """Train marginals.

        Args:
            batch_size (int): Batch size...
            TODO
        """
        if self._dataset is None:
            print("ERROR: No dataset specified.")
            return
        if self._dataset.requires_sim:
            print("ERROR: Not all points in the dataset are simulated yet.")
            return

        marginals = tupleize_marginals(marginals)
        re = self._ratios[marginals]

        trainoptions = TrainOptions(
            batch_size=batch_size,
            validation_size=validation_size,
            early_stopping_patience=early_stopping_patience,
            max_epochs=max_epochs,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            nworkers=nworkers,
            non_blocking=non_blocking,
        )

        re.train(self._dataset, trainoptions)

    def train_diagnostics(self, marginals: MarginalsType):
        marginals = tupleize_marginals(marginals)
        return self._ratios[marginals].train_diagnostics()

    def sample(
        self, N: int, obs0: ObsType, n_batch: int = 100
    ) -> Dict[str, Union[np.ndarray, RatiosType, PNamesType]]:
        """Resturn weighted posterior samples for given observation.

        Args:
            N: Number of samples to return
            obs0: Observation of interest
            n_batch: number of samples to produce in each batch
        """
        v = self._trunc_prior.sample(N)  # prior samples

        # Unmasked original wrongly normalized log_prob densities
        # log_probs = self._trunc_prior.log_prob(v)
        u = self._trunc_prior.prior.u(v)

        ratios = self._eval_ratios(
            obs0, u, n_batch=n_batch
        )  # evaluate lnL for reference observation
        weights = {}
        for k, val in ratios.items():
            weights[k] = np.exp(val)
        return dict(v=v, weights=weights, pnames=self.pnames)

    #    # TODO: Still needs to be fixed?
    #    def _rejection_sample(
    #        self,
    #        N: int,
    #        obs0: ObsType,
    #        excess_factor: int = 10,
    #        maxiter: int = 1000,
    #        n_batch: int = 10_000,
    #        PoI: Sequence[PoIType] = None,
    #    ) -> MarginalType:
    #        """Samples from each marginal using rejection sampling.
    #
    #        Args:
    #            N: number of samples in each marginal to output
    #            obs0: target observation
    #            excess_factor: N_to_reject = excess_factor * N
    #            maxiter: maximum loop attempts to draw N
    #            n_batch: how many proposed samples are drawn at once
    #            PoI: selection of parameters of interest
    #
    #        Returns:
    #            Marginal posterior samples. keys are marginal tuples, values are samples/
    #
    #        Reference:
    #            Section 23.3.3
    #            Machine Learning: A Probabilistic Perspective
    #            Kevin P. Murphy
    #        """
    #
    #        weighted_samples = self.sample(N=excess_factor * N, obs0=obs0, n_batch=10_000)
    #
    #        maximum_log_likelihood_estimates = {
    #            k: np.log(np.max(v)) for k, v in weighted_samples["weights"].items()
    #        }
    #
    #        PoI = set(weighted_samples["weights"].keys()) if PoI is None else PoI
    #        collector = {k: [] for k in PoI}
    #        out = {}
    #
    #        # Do the rejection sampling.
    #        # When a particular key hits the necessary samples, stop calculating on it to reduce cost.
    #        # Send that key to out.
    #        counter = 0
    #        remaining_param_tuples = PoI
    #        while counter < maxiter:
    #            # Calculate chance to keep a sample
    #
    #            log_prob_to_keep = {
    #                pt: np.log(weighted_samples["weights"][pt])
    #                - maximum_log_likelihood_estimates[pt]
    #                for pt in remaining_param_tuples
    #            }
    #
    #            # Draw and determine if samples are kept
    #            to_keep = {
    #                pt: np.less_equal(np.log(np.random.rand(*v.shape)), v)
    #                for pt, v in log_prob_to_keep.items()
    #            }
    #
    #            # Collect samples for every tuple of parameters, if there are enough, add them to out.
    #            for param_tuple in remaining_param_tuples:
    #                kept_all_params = weighted_samples["v"][to_keep[param_tuple]]
    #                kept_params = kept_all_params[..., param_tuple]
    #                collector[param_tuple].append(kept_params)
    #                concatenated = np.concatenate(collector[param_tuple])[:N]
    #                if len(concatenated) == N:
    #                    out[param_tuple] = concatenated
    #
    #            # Remove the param_tuples which we already have in out, thus not to calculate for them anymore.
    #            for param_tuple in out.keys():
    #                if param_tuple in remaining_param_tuples:
    #                    remaining_param_tuples.remove(param_tuple)
    #                    log.debug(f"{len(remaining_param_tuples)} param tuples remaining")
    #
    #            if len(remaining_param_tuples) > 0:
    #                weighted_samples = self.sample(
    #                    N=excess_factor * N, obs0=obs0, n_batch=n_batch
    #                )
    #            else:
    #                return out
    #            counter += 1
    #        warn(
    #            f"Max iterations {maxiter} reached there were not enough samples produced in {remaining_param_tuples}."
    #        )
    #        return out

    @property
    def bound(self) -> "swyft.bounds.Bound":
        return self._trunc_prior.bound

    @property
    def prior(self) -> "swyft.bounds.Prior":
        return self._trunc_prior.prior

    def truncate(self, marginals: MarginalsType, obs0: ObsType) -> "swyft.bounds.Bound":
        """Generate and return new bound object."""
        marginals = tupleize_marginals(marginals)
        bound = swyft.Bound.from_Posteriors(marginals, self, obs0)
        print("Bounds: Truncating...")
        print("Bounds: ...done. New volue is V=%.4g" % bound.volume)
        return bound

    def _eval_ratios(self, obs: ObsType, v: Array, n_batch: int = 100) -> RatiosType:
        result = {}
        for _, rc in self._ratios.items():
            ratios = rc.ratios(obs, v, n_batch=n_batch)
            result.update(ratios)
        return result

    def state_dict(self) -> dict:
        state_dict = dict(
            trunc_prior=self._trunc_prior.state_dict(),
            ratios={k: v.state_dict() for k, v in self._ratios.items()},
            pnames=self._pnames,
        )
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict, dataset: "swyft.Dataset" = None):
        obj = Posteriors.__new__(Posteriors)
        obj._trunc_prior = swyft.TruncatedPrior.from_state_dict(
            state_dict["trunc_prior"]
        )
        obj._ratios = {
            k: RatioEstimator.from_state_dict(v)
            for k, v in state_dict["ratios"].items()
        }
        obj._pnames = state_dict["pnames"]
        obj._dataset = dataset
        return obj

    @classmethod
    def load(cls, filename: PathType, dataset: "swyft.Dataset" = None):
        sd = torch.load(filename)
        return cls.from_state_dict(sd, dataset=dataset)

    def save(self, filename: PathType) -> None:
        """Save a posterior.

        Args:
            filename: Filename

        .. note::
            What will be saved are: parameter names, the prior and the bound,
            as well as all networks.  We will NOT save the dataset, which can
            be however specified during `load` if necessary.
        """
        sd = self.state_dict()
        torch.save(sd, filename)
