import logging
from warnings import warn

import numpy as np
import torch

import swyft
from swyft.types import Array
from swyft.utils import tupelize_marginals
from .ratios import RatioEstimator
from swyft.networks import DefaultHead, DefaultTail

log = logging.getLogger(__name__)

# TODO: Deprecate!
class PosteriorCollection:
    def __init__(self, rc, prior):
        """Marginal container initialization.

        Args:
            rc (RatioCollection)
            prior (BoundPrior)
        """
        self._rc = rc
        self._prior = prior

    def sample(self, N, obs, device=None, n_batch=10_000):
        """Resturn weighted posterior samples for given observation.

        Args:
            obs (dict): Observation of interest.
            N (int): Number of samples to return.
        """

        v = self._prior.sample(N)  # prior samples

        # Unmasked original wrongly normalized log_prob densities
        # log_probs = self._prior.log_prob(v)
        u = self._prior.prior.u(v)

        ratios = self._rc.ratios(
            obs, u, device=device, n_batch=n_batch
        )  # evaluate lnL for reference observation

        weights = {}
        for k, val in ratios.items():

            weights[k] = np.exp(val)

        return dict(params=v, weights=weights)

    def rejection_sample(
        self,
        N: int,
        obs: dict,
        excess_factor: int = 100,
        maxiter: int = 1000,
        device=None,
        n_batch=10_000,
        param_tuples=None,
    ):
        """Samples from each marginal using rejection sampling.

        Args:
            N (int): number of samples in each marginal to output
            obs (dict): target observation
            excess_factor (int, optional): N_to_reject = excess_factor * N . Defaults to 100.
            maxiter (int, optional): maximum loop attempts to draw N. Defaults to 1000.

        Returns:
            Dict[str, np.ndarray]: keys are marginal tuples, values are samples

        Reference:
            Section 23.3.3
            Machine Learning: A Probabilistic Perspective
            Kevin P. Murphy
        """

        weighted_samples = self.sample(
            N=excess_factor * N, obs=obs, device=device, n_batch=10_000
        )

        maximum_log_likelihood_estimates = {
            k: np.log(np.max(v)) for k, v in weighted_samples["weights"].items()
        }

        param_tuples = (
            set(weighted_samples["weights"].keys())
            if param_tuples is None
            else param_tuples
        )
        collector = {k: [] for k in param_tuples}
        out = {}

        # Do the rejection sampling.
        # When a particular key hits the necessary samples, stop calculating on it to reduce cost.
        # Send that key to out.
        counter = 0
        remaining_param_tuples = param_tuples
        while counter < maxiter:
            # Calculate chance to keep a sample

            log_prob_to_keep = {
                pt: np.log(weighted_samples["weights"][pt])
                - maximum_log_likelihood_estimates[pt]
                for pt in remaining_param_tuples
            }

            # Draw and determine if samples are kept
            to_keep = {
                pt: np.less_equal(np.log(np.random.rand(*v.shape)), v)
                for pt, v in log_prob_to_keep.items()
            }

            # Collect samples for every tuple of parameters, if there are enough, add them to out.
            for param_tuple in remaining_param_tuples:
                kept_all_params = weighted_samples["params"][to_keep[param_tuple]]
                kept_params = kept_all_params[..., param_tuple]
                collector[param_tuple].append(kept_params)
                concatenated = np.concatenate(collector[param_tuple])[:N]
                if len(concatenated) == N:
                    out[param_tuple] = concatenated

            # Remove the param_tuples which we already have in out, thus not to calculate for them anymore.
            for param_tuple in out.keys():
                if param_tuple in remaining_param_tuples:
                    remaining_param_tuples.remove(param_tuple)
                    log.debug(f"{len(remaining_param_tuples)} param tuples remaining")

            if len(remaining_param_tuples) > 0:
                weighted_samples = self.sample(
                    N=excess_factor * N, obs=obs, device=device, n_batch=n_batch
                )
            else:
                return out
            counter += 1
        warn(
            f"Max iterations {maxiter} reached there were not enough samples produced in {remaining_param_tuples}."
        )
        return out

    def state_dict(self):
        return dict(rc=self._rc.state_dict(), prior=self._prior.state_dict())

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(
            RatioCollection.from_state_dict(state_dict["rc"]),
            swyft.Prior.from_state_dict(state_dict["prior"]),
        )

    @classmethod
    def load(cls, filename):
        sd = torch.load(filename)
        return cls.from_state_dict(sd)

    def save(self, filename):
        sd = self.state_dict()
        torch.save(sd, filename)


class Posteriors:
    def __init__(self, prior, bound=None):
        self._trunc_prior = swyft.TruncatedPrior(prior, bound=bound)
        self._ratios = {}

    def add(
        self,
        marginals,
        head=DefaultHead,
        tail=DefaultTail,
        head_args: dict = {},
        tail_args: dict = {},
        device="cpu",
    ):
        """Add marginals.

        Args:
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
        """
        marginals = tupelize_marginals(marginals)
        re = RatioEstimator(
            marginals,
            device=device,
            head=head,
            tail=tail,
            head_args=head_args,
            tail_args=tail_args,
        )
        self._ratios[marginals] = re

    def to(self, device, marginals=None):
        if marginals is not None:
            marginals = tupelize_marginals(marginals)
            self._ratios[marginals].to(device)
        else:
            for _, v in self._ratios.items():
                v.to(device)
        return self

    def train(self, marginals, dataset, 
        batch_size=64,
        validation_size=0.1,
        early_stopping_patience=5,
        max_epochs=30,
        optimizer=torch.optim.Adam,
        optimizer_args=dict(lr=1e-3),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args=dict(factor=0.1, patience=5),
        nworkers=2,
        non_blocking=True,
        ):
        """Train marginals.

        Args:
            train_args (dict): Training keyword arguments.
        """
        if dataset.requires_sim:
            print("ERROR: Not all points in the dataset are simulated yet.")
            return
        marginals = tupelize_marginals(marginals)
        re = self._ratios[marginals]
        re.train(dataset, 
                batch_size=batch_size,
                validation_size=validation_size,
                early_stopping_patience=early_stopping_patience,
                max_epochs=max_epochs,
                optimizer=optimizer,
                optimizer_args=optimizer_args,
                scheduler=scheduler,
                scheduler_args=scheduler_args,
                nworkers=nworkers,
                non_blocking=non_blocking)

    def train_diagnostics(self, marginals):
        return self._ratios[marginals].train_diagnostics()

    def sample(self, N, obs0):
        """Resturn weighted posterior samples for given observation.

        Args:
            obs0 (dict): Observation of interest.
            N (int): Number of samples to return.
        """
        v = self._trunc_prior.sample(N)  # prior samples

        # Unmasked original wrongly normalized log_prob densities
        # log_probs = self._trunc_prior.log_prob(v)
        u = self._trunc_prior.prior.u(v)

        ratios = self._eval_ratios(obs0, u)  # evaluate lnL for reference observation
        weights = {}
        for k, val in ratios.items():
            weights[k] = np.exp(val)
        return dict(params=v, weights=weights)

    #    def sample(self, N, obs, device=None, n_batch=10_000):
    #        post = PosteriorCollection(self.ratios, self._trunc_prior)
    #        samples = post.sample(N, obs, device=device, n_batch=n_batch)
    #        return samples

    # TODO: Import from PosteriorCollection
    def rejection_sample(
        self,
        N: int,
        obs: dict,
        excess_factor: int = 100,
        maxiter: int = 1000,
        device=None,
        n_batch=10_000,
    ):
        """Samples from each marginal using rejection sampling.

        Args:
            N (int): number of samples in each marginal to output
            obs (dict): target observation
            excess_factor (int, optional): N_to_reject = excess_factor * N . Defaults to 100.
            maxiter (int, optional): maximum loop attempts to draw N. Defaults to 1000.

        Returns:
            Dict[str, np.ndarray]: keys are marginal tuples, values are samples

        Reference:
            Section 23.3.3
            Machine Learning: A Probabilistic Perspective
            Kevin P. Murphy
        """
        post = PosteriorCollection(self.ratios, self._trunc_prior)
        return post.rejection_sample(
            N=N,
            obs=obs,
            excess_factor=excess_factor,
            maxiter=maxiter,
            device=device,
            n_batch=n_batch,
        )

    @property
    def bound(self):
        return self._trunc_prior.bound

    @property
    def prior(self):
        return self._trunc_prior.prior

    def truncate(self, partition, obs0):
        """Generate and return new bound object."""
        partition = tupelize_marginals(partition)
        bound = swyft.Bound.from_Posteriors(partition, self, obs0)
        print("Bounds: Truncating...")
        print("Bounds: ...done. New volue is V=%.4g" % bound.volume)
        return bound

    def _eval_ratios(self, obs: Array, params: Array, n_batch=100):
        result = {}
        for marginals, rc in self._ratios.items():
            ratios = rc.ratios(obs, params, n_batch=n_batch)
            result.update(ratios)
        return result

    def state_dict(self):
        state_dict = dict(
            trunc_prior=self._trunc_prior.state_dict(),
            ratios={k: v.state_dict() for k, v in self._ratios.items()},
        )
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict):
        obj = Posteriors.__new__(Posteriors)
        obj._trunc_prior = swyft.TruncatedPrior.from_state_dict(
            state_dict["trunc_prior"]
        )
        obj._ratios = {
            k: RatioEstimator.from_state_dict(v)
            for k, v in state_dict["ratios"].items()
        }
        return obj

    @classmethod
    def load(cls, filename):
        sd = torch.load(filename)
        return cls.from_state_dict(sd)

    def save(self, filename):
        sd = self.state_dict()
        torch.save(sd, filename)
