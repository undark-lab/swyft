import numpy as np

from swyft.inference import RatioCollection
from swyft.marginals.prior import BoundedPrior


class PosteriorCollection:
    def __init__(self, rc, prior):
        """Marginal container initialization.

        Args:
            rc (RatioCollection)
            prior (BoundPrior)
        """
        self._rc = rc
        self._prior = prior 

    def sample(self, obs0, N):
        """Resturn weighted posterior samples for given observation.

        Args:
            obs0 (dict): Observation of interest.
            N (int): Number of samples to return.
        """
        v = self._prior.sample(N)  # prior samples

        # Unmasked original wrongly normalized log_prob densities
        log_probs = self._prior.log_prob(v)
        u = self._prior.ptrans.u(v)

        ratios = self._rc.ratios(obs0, u)  # evaluate lnL for reference observation
        weights = {}
        for k, val in ratios.items():
            weights[k] = np.exp(val)
        return dict(params=v, weights=weights)

    def state_dict(self):
        return dict(rc=self._rc.state_dict(), prior=self._prior.state_dict())

    @classmethod
    def from_state_dict(cls, state_dict):
        return RatioEstimatedPosterior(
            RatioCollection.from_state_dict(state_dict["rc"]),
            BoundedPrior.from_state_dict(state_dict["prior"]),
        )
