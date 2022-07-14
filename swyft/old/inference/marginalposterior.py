from typing import Optional, Tuple, TypeVar
from warnings import warn

import numpy as np
import torch
from torch.types import Device
from torch.utils.data.dataset import Dataset

from swyft.bounds import Bound
from swyft.inference.marginalratioestimator import MarginalRatioEstimator
from swyft.prior import Prior, PriorTruncator
from swyft.types import Array, MarginalIndex, MarginalToArray, ObsType
from swyft.utils import tupleize_marginal_indices
from swyft.weightedmarginals import WeightedMarginalSamples

MarginalPosteriorType = TypeVar("MarginalPosteriorType", bound="MarginalPosterior")


class MarginalPosterior:
    def __init__(
        self,
        marginal_ratio_estimator: MarginalRatioEstimator,
        prior: Prior,
        bound: Optional[Bound] = None,
    ) -> None:
        """a trained marginal ratio estimator and a prior allows access to the posterior

        Args:
            marginal_ratio_estimator: trained marginal ratio estimator
            prior: associated prior. if swyft.Prior is provided then create an untruncated prior to sample from.
        """
        super().__init__()
        self.marginal_ratio_estimator = marginal_ratio_estimator
        self.prior = prior
        self.bound = bound
        self.prior_truncator = PriorTruncator(self.prior, self.bound)

    @property
    def marginal_indices(self) -> MarginalIndex:
        return tupleize_marginal_indices(self.marginal_ratio_estimator.marginal_indices)

    @property
    def device(self) -> Device:
        return self.marginal_ratio_estimator.device

    def truncate(
        self, n_samples: int, observation: ObsType, threshold: float = -13.0
    ) -> Bound:
        return Bound.from_marginal_posterior(
            n_samples=n_samples,
            observation=observation,
            marginal_posterior=self,
            threshold=threshold,
        )

    def empirical_mass(
        self,
        n_observations: int,
        n_posterior_samples: int,
        dataset: Dataset,
    ) -> Tuple[MarginalToArray, MarginalToArray]:
        """compute the empirical and nominal masses

        Args:
            n_observations: number of observations to estimate with
            n_posterior_samples: number of samples to estimate the mass for a certain observation
            dataset: indexable torch dataset which outputs `observation, u, v = dataset[i]`

        Returns:
            empirical mass dict and nominal mass dict for every marginal
        """
        empirical_mass = {marginal: [] for marginal in self.marginal_indices}
        nominal_mass = {
            marginal: np.linspace(1 / n_observations, 1, n_observations)
            for marginal in self.marginal_indices
        }

        for _ in range(n_observations):
            ind = np.random.randint(n_observations)
            observation_o, _, v_o = dataset[ind]
            logw_o = self.marginal_ratio_estimator.log_ratio(
                observation_o, torch.atleast_2d(v_o)
            )
            logw_s = self.weighted_sample(n_posterior_samples, observation_o)

            w_o = {
                marginal_index: np.exp(logw) for marginal_index, logw in logw_o.items()
            }
            w_s = {
                marginal_index: np.exp(logw)
                for marginal_index, logw in logw_s.weights.items()
            }
            for marginal_index, weight_o in w_o.items():
                is_above_true_param_density = w_s[marginal_index] >= weight_o
                sum_above_true_param_density = w_s[marginal_index][
                    is_above_true_param_density
                ].sum()
                percent_above_true_param_density = (
                    sum_above_true_param_density / w_s[marginal_index].sum()
                )
                empirical_mass[marginal_index].append(percent_above_true_param_density)
        for marginal_index, mass in empirical_mass.items():
            empirical_mass[marginal_index] = np.asarray(sorted(mass))

        return empirical_mass, nominal_mass

    def log_prob(
        self, observation: ObsType, v: Array, batch_size: Optional[int] = None
    ) -> MarginalToArray:
        """estimate the log posterior probability of an observation and parameters v

        Args:
            observation: a single observation
            v: parameters to estimate the log proability
            batch_size: batch_size to estimate the log ratio. None attempts to estimate all log ratios at once

        Returns:
            dictionary with marginal_indices keys and log posterior values.
        """
        log_prior = np.atleast_2d(self.prior_truncator.log_prob(v))
        marginal_log_prior = {
            k: log_prior[..., i] for i, k in enumerate(self.marginal_indices)
        }
        log_ratios = self.marginal_ratio_estimator.log_ratio(
            observation, v, batch_size=batch_size
        )
        return {
            marginal_index: log_ratios[marginal_index]
            + marginal_log_prior[marginal_index]
            for marginal_index in self.marginal_indices
        }

    def weighted_sample(
        self, n_samples: int, observation: ObsType, batch_size: Optional[int] = None
    ) -> WeightedMarginalSamples:
        """sample from the prior and estimate the log ratio weights

        Args:
            n_samples: number of desired samples
            observation: observation in the posterior
            batch_size: batch_size to estimate the log ratio. None attempts to estimate all log ratios at once

        Returns:
            dictionary with keys = marginal_indices U {"v"}
                "v" maps to the parameters drawn from the prior (n_samples, n_parameters).
                each marginal index maps to the log ratio (n_samples, len(marginal_index)).
        """
        v = self.prior_truncator.sample(n_samples)
        logweight = self.marginal_ratio_estimator.log_ratio(
            observation, v, batch_size=batch_size
        )
        return WeightedMarginalSamples(logweight, v)

    def sample(
        self,
        n_samples: int,
        observation: ObsType,
        batch_size: Optional[int] = None,
        excess_factor: int = 10,
        maxiter: int = 1000,
    ) -> MarginalToArray:
        """sample from the posterior using rejection sampling

        Args:
            n_samples: number of samples in each marginal to output
            observation: target observation
            batch_size: batch_size to estimate the log ratio. None attempts to estimate all log ratios at once
            excess_factor: n_samples_to_reject = excess_factor * n_samples
            maxiter: maximum loop attempts to draw n_samples

        Returns:
            Marginal posterior samples. keys are marginal tuples, values are samples.

        Reference:
            Section 23.3.3
            Machine Learning: A Probabilistic Perspective
            Kevin P. Murphy
        """
        # From an initial draw, estimate the log MAP
        weighted_samples = self.weighted_sample(
            excess_factor * n_samples, observation, batch_size
        )

        marginal_log_MAP = {
            marginal_index: weighted_samples.get_logweight(marginal_index).max()
            for marginal_index in self.marginal_indices
        }

        # define the work left to do, the place to put it, and the end result / output.
        remaining_marginal_indices = set(marginal_log_MAP.keys())
        collector = {k: [] for k in remaining_marginal_indices}
        out = {}

        # Do the rejection sampling.
        # When a particular key hits the necessary samples, stop calculating on it to reduce cost.
        # Send that key to out.
        counter = 0
        while counter < maxiter:
            # Calculate chance to keep a sample
            log_prob_to_keep = {
                marginal_index: weighted_samples.get_logweight(marginal_index)
                - marginal_log_MAP[marginal_index]
                for marginal_index in remaining_marginal_indices
            }

            # Draw and determine if samples are kept
            to_keep = {
                marginal_index: np.less_equal(
                    np.log(np.random.rand(*log_prob.shape)), log_prob
                )
                for marginal_index, log_prob in log_prob_to_keep.items()
            }

            # Collect samples for every tuple of parameters, if there are enough, add them to out.
            for marginal_index in remaining_marginal_indices:
                all_parameters_to_keep = weighted_samples.v[to_keep[marginal_index]]
                marginal_to_keep = all_parameters_to_keep[..., marginal_index]
                collector[marginal_index].append(marginal_to_keep)
                concatenated = np.concatenate(collector[marginal_index])[:n_samples]
                if len(concatenated) == n_samples:
                    out[marginal_index] = concatenated

            # Remove the param_tuples which we already have in out, thus to avoid calculating them anymore.
            for marginal_index in out.keys():
                if marginal_index in remaining_marginal_indices:
                    remaining_marginal_indices.remove(marginal_index)

            if len(remaining_marginal_indices) > 0:
                weighted_samples = self.weighted_sample(
                    excess_factor * n_samples, observation, batch_size
                )
            else:
                return out
            counter += 1
        warn(
            f"Max iterations {maxiter} reached there were not enough samples produced in {remaining_marginal_indices}."
        )
        return out


if __name__ == "__main__":
    pass
