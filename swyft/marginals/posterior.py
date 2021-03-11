from typing import Dict
from warnings import warn

import numpy as np

from swyft.inference import RatioEstimator
from swyft.marginals.prior import Prior


class RatioEstimatedPosterior:
    """Marginal container"""

    def __init__(self, ratio, prior):
        """Marginal container initialization.

        Args:
            re (RatioEstimator)
            prior (Prior)
        """
        self._re = ratio
        self._prior = prior

    @property
    def prior(self):
        """(Constrained) prior of the marginal."""
        return self._prior

    @property
    def ratio(self):
        """Ratio estimator for marginals."""
        return self._re

    def __call__(self, obs, n_samples=100000):
        """Return weighted marginal samples.

        Args:
            obs (dict): Observation.
            n_samples (int): Number of samples.

        Returns:
            dict containing samples.

        Note: Observations must be restricted to constrained prior space to
        lead to valid results.
        """
        # FIXME: Make return of log_priors conditional on constrained prior
        # being factorizable
        return self._re.posterior(obs, self._prior, n_samples=n_samples)

    def state_dict(self):
        """Return state_dict."""
        return dict(re=self._re.state_dict(), prior=self._prior.state_dict())

    @classmethod
    def from_state_dict(cls, state_dict):
        """Instantiate RatioEstimatedPosterior based on state_dict."""
        return RatioEstimatedPosterior(
            RatioEstimator.from_state_dict(state_dict["re"]),
            Prior.from_state_dict(state_dict["prior"]),
        )

    def gen_constr_prior(self, obs, th=-10):
        """Generate new constrained prior based on ratio estimator and target observation.

        Args:
            obs (dict): Observation.
            th (float): Cutoff maximum log likelihood ratio. Default is -10,
                        which correspond roughly to 4 sigma.

        Returns:
            Prior: Constrained prior.
        """
        return self._prior.get_masked(obs, self._re, th=th)

    def sample_posterior(
        self, obs: dict, n_samples: int, excess_factor: int = 10, maxiter: int = 100
    ) -> Dict[str, np.ndarray]:
        """Samples from each marginal using rejection sampling.

        Args:
            obs (dict): target observation
            n_samples (int): number of samples in each marginal to output
            excess_factor (int, optional): n_samples_to_reject = excess_factor * n_samples . Defaults to 100.
            maxiter (int, optional): maximum loop attempts to draw n_samples. Defaults to 10.

        Returns:
            Dict[str, np.ndarray]: keys are marginal tuples, values are samples

        Reference:
            Section 23.3.3
            Machine Learning: A Probabilistic Perspective
            Kevin P. Murphy
        """
        draw = self._re.posterior(obs, self._prior, n_samples=excess_factor * n_samples)
        maximum_log_likelihood_estimates = {
            k: np.log(np.max(v)) for k, v in draw["weights"].items()
        }
        remaining_param_tuples = set(draw["weights"].keys())
        collector = {k: [] for k in remaining_param_tuples}
        out = {}

        # Do the rejection sampling.
        # When a particular key hits the necessary samples, stop calculating on it to reduce cost.
        # Send that key to out.
        counter = 0
        while counter < maxiter:
            # Calculate chance to keep a sample
            log_prob_to_keep = {
                pt: np.log(draw["weights"][pt]) - maximum_log_likelihood_estimates[pt]
                for pt in remaining_param_tuples
            }

            # Draw and determine if samples are kept
            to_keep = {
                pt: np.less_equal(np.log(np.random.rand(excess_factor * n_samples)), v)
                for pt, v in log_prob_to_keep.items()
            }

            # Collect samples for every tuple of parameters, if there are enough, add them to out.
            for param_tuple in remaining_param_tuples:
                collector[param_tuple].append(
                    np.stack(
                        [
                            draw["params"][name][to_keep[param_tuple]]
                            for name in param_tuple
                        ],
                        axis=-1,
                    )
                )
                concatenated = np.concatenate(collector[param_tuple])[:n_samples]
                if len(concatenated) == n_samples:
                    out[param_tuple] = concatenated

            # Remove the param_tuples which we already have in out, thus not to calculate for them anymore.
            for param_tuple in out.keys():
                if param_tuple in remaining_param_tuples:
                    remaining_param_tuples.remove(param_tuple)

            if len(remaining_param_tuples) > 0:
                draw = self._re.posterior(
                    obs, self._prior, n_samples=excess_factor * n_samples
                )
            else:
                return out
            counter += 1
        warn(
            f"Max iterations {maxiter} reached there were not enough samples produced in {remaining_param_tuples}."
        )
        return out
