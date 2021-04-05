from typing import Dict
from warnings import warn

import numpy as np

from swyft.inference import RatioCollection


class PosteriorCollection:
    def __init__(self, ratio_collection, bounded_prior):
        """Marginal container initialization.

        Args:
            re (RatioCollection)
            prior (Prior)
        """
        self._rc = ratio_collection
        self._prior = bounded_prior

    def sample(self, obs0, n_samples=100000):
        """Resturn weighted posterior samples for given observation.

        Args:
            obs0 (dict): Observation of interest.
            prior (Prior): (Constrained) prior used to generate training data.
            n_samples (int): Number of samples to return.

        Note:
            log_priors are not normalized.
        """
        v = self._prior.sample(n_samples)  # prior samples

        # Unmasked original wrongly normalized log_prob densities
        log_probs = self._prior.log_prob(v)
        u = self._prior.ptrans.u(v)

        ratios = self._rc.ratios(obs0, u)  # evaluate lnL for reference observation
        weights = {}
        for k, val in ratios.items():
            weights[k] = np.exp(val)
        return dict(params=v, weights=weights)

    def state_dict(self):
        """Return state_dict."""
        return dict(re=self._re.state_dict(), prior=self._prior.state_dict())

    @classmethod
    def from_state_dict(cls, state_dict):
        """Instantiate RatioEstimatedPosterior based on state_dict."""
        return RatioEstimatedPosterior(
            RatioCollection.from_state_dict(state_dict["re"]),
            Prior.from_state_dict(state_dict["prior"]),
        )

#    def sample_posterior(
#        self, obs: dict, n_samples: int, excess_factor: int = 10, maxiter: int = 100
#    ) -> Dict[str, np.ndarray]:
#        """Samples from each marginal using rejection sampling.
#
#        Args:
#            obs (dict): target observation
#            n_samples (int): number of samples in each marginal to output
#            excess_factor (int, optional): n_samples_to_reject = excess_factor * n_samples . Defaults to 100.
#            maxiter (int, optional): maximum loop attempts to draw n_samples. Defaults to 10.
#
#        Returns:
#            Dict[str, np.ndarray]: keys are marginal tuples, values are samples
#
#        Reference:
#            Section 23.3.3
#            Machine Learning: A Probabilistic Perspective
#            Kevin P. Murphy
#        """
#        draw = self._re.posterior(obs, self._prior, n_samples=excess_factor * n_samples)
#        maximum_log_likelihood_estimates = {
#            k: np.log(np.max(v)) for k, v in draw["weights"].items()
#        }
#        remaining_param_tuples = set(draw["weights"].keys())
#        collector = {k: [] for k in remaining_param_tuples}
#        out = {}
#
#        # Do the rejection sampling.
#        # When a particular key hits the necessary samples, stop calculating on it to reduce cost.
#        # Send that key to out.
#        counter = 0
#        while counter < maxiter:
#            # Calculate chance to keep a sample
#            log_prob_to_keep = {
#                pt: np.log(draw["weights"][pt]) - maximum_log_likelihood_estimates[pt]
#                for pt in remaining_param_tuples
#            }
#
#            # Draw and determine if samples are kept
#            to_keep = {
#                pt: np.less_equal(np.log(np.random.rand(excess_factor * n_samples)), v)
#                for pt, v in log_prob_to_keep.items()
#            }
#
#            # Collect samples for every tuple of parameters, if there are enough, add them to out.
#            for param_tuple in remaining_param_tuples:
#                collector[param_tuple].append(
#                    np.stack(
#                        [
#                            draw["params"][name][to_keep[param_tuple]]
#                            for name in param_tuple
#                        ],
#                        axis=-1,
#                    )
#                )
#                concatenated = np.concatenate(collector[param_tuple])[:n_samples]
#                if len(concatenated) == n_samples:
#                    out[param_tuple] = concatenated
#
#            # Remove the param_tuples which we already have in out, thus not to calculate for them anymore.
#            for param_tuple in out.keys():
#                if param_tuple in remaining_param_tuples:
#                    remaining_param_tuples.remove(param_tuple)
#
#            if len(remaining_param_tuples) > 0:
#                draw = self._re.posterior(
#                    obs, self._prior, n_samples=excess_factor * n_samples
#                )
#            else:
#                return out
#            counter += 1
#        warn(
#            f"Max iterations {maxiter} reached there were not enough samples produced in {remaining_param_tuples}."
#        )
#        return out
