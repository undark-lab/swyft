import numpy as np

from swyft.marginals import Prior


class Intensity:
    def __init__(self, prior, mu):
        self.prior = prior
        self.mu = mu

    def sample(self):
        N = np.random.poisson(self.mu)
        return self.prior.sample(N)

    def __call__(self, values):
        """Return the log intensity function on values."""
        return self.prior.log_prob(values) + np.log(self.mu)

    @classmethod
    def from_state_dict(cls, state_dict):
        prior = Prior.from_state_dict(state_dict["prior"])
        mu = state_dict["mu"]
        return cls(prior, mu)

    def state_dict(self):
        prior = self.prior.state_dict()
        return dict(prior=prior, mu=self.mu)
