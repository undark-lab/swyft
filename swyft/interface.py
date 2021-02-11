from warnings import warn

import numpy as np

from .cache import DirectoryCache, MemoryCache
from .estimation import Points, RatioEstimator
from .intensity import Prior
from .network import DefaultHead, DefaultTail
from .utils import all_finite, format_param_list, verbosity
from .types import Dict


class MissingModelError(Exception):
    pass


class Marginals:
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
        return self._prior

    @property
    def ratio(self):
        return self._re

    def __call__(self, obs, n_samples=100000):
        """Return weighted posterior samples.

        Args:
            obs (dict): Observation.
            n_samples (int): Number of samples.

        Returns:
            dict containing samples.

        Note: Observations must be restricted to constrained prior space to
        lead to valid results.
        """
        return self._re.posterior(obs, self._prior, n_samples=n_samples)

    def state_dict(self):
        """Return state_dict."""
        return dict(re=self._re.state_dict(), prior=self._prior.state_dict())

    @classmethod
    def from_state_dict(cls, state_dict):
        """Instantiate posterior based on state_dict."""
        return Marginals(
            RatioEstimator.from_state_dict(state_dict["re"]),
            Prior.from_state_dict(state_dict["prior"]),
        )

    def gen_constr_prior(self, obs, th=-10):
        """Generate constrained prior based on ratio estimator.

        Args:
            obs (dict): Observation.
            th (float): Cutoff maximum log likelihood ratio. Default is -10,
                        which correspond roughly to 4 sigma.

        Returns:
            Prior: Constrained prior.
        """
        return self._prior.get_masked(obs, self._re, th=th)
    
    def sample_posterior(self, obs: dict, n_samples: int, excess_factor: int = 10, maxiter: int = 10) -> Dict[str, np.ndarray]:
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
        maximum_log_likelihood_estimates = {k: np.log(np.max(v)) for k, v in draw['weights'].items()}
        remaining_param_tuples = set(draw['weights'].keys())
        collector = {k: [] for k in remaining_param_tuples}
        out = {}
        
        # Do the rejection sampling.
        # When a particular key hits the necessary samples, stop calculating on it to reduce cost.
        # Send that key to out.
        counter = 0
        while counter < maxiter:
            # Calculate chance to keep a sample
            log_prob_to_keep = {
                pt: np.log(draw['weights'][pt]) - maximum_log_likelihood_estimates[pt] for pt in remaining_param_tuples
            }
            
            # Draw and determine if samples are kept
            to_keep = {
                pt: np.less_equal(
                    np.log(np.random.rand(excess_factor * n_samples)),
                    v
                ) for pt, v in log_prob_to_keep.items()
            }

            # Collect samples for every tuple of parameters, if there are enough, add them to out.
            for param_tuple in remaining_param_tuples:
                collector[param_tuple].append(
                    np.stack([draw['params'][name][to_keep[param_tuple]] for name in param_tuple], axis=-1)
                )
                concatenated = np.concatenate(collector[param_tuple])[:n_samples]
                if len(concatenated) == n_samples:
                    out[param_tuple] = concatenated
            
            # Remove the param_tuples which we already have in out, thus not to calculate for them anymore.
            for param_tuple in out.keys():
                if param_tuple in remaining_param_tuples:
                    remaining_param_tuples.remove(param_tuple)
            
            if len(remaining_param_tuples) > 0:
                draw = self._re.posterior(obs, self._prior, n_samples=excess_factor * n_samples)
            else:
                return out
            counter += 1
        warn(f"Max iterations {maxiter} reached there were not enough samples produced in {remaining_param_tuples}.")
        return out


class NestedRatios:
    """Main SWYFT interface class."""

    def __init__(self, model, prior, obs, noise=None, cache=None, device="cpu"):
        """Initialize swyft.

        Args:
            model (function): Simulator function.
            prior (Prior): Prior model.
            obs (dict): Target observation
            noise (function): Noise model, optional.
            cache (Cache): Storage for simulator results.  If none, create MemoryCache.
            device (str): Device.
        """
        # Not stored
        self._model = model
        self._noise = noise
        if all_finite(obs):
            self._obs = obs
        else:
            raise ValueError("obs must be finite.")
        if cache is None:
            cache = MemoryCache.from_simulator(model, prior)
        self._cache = cache
        self._device = device

        # Stored in state_dict()
        #self._converged = False
        self._base_prior = prior  # Initial prior
        self._posterior = None  # Posterior of a latest round
        self._constr_prior = (
            None  # Constrained prior based on self._posterior and self._obs
        )
        self._R = 0  # Round counter
        self._N = None  # Training data points
        self._history = []

    #def converged(self):
    #    return self._converged

    @property
    def obs(self):
        return self._obs

    @property
    def marginals(self):
        if self._posterior is None:
            if verbosity() >= 1:
                print("NOTE: To generated marginals from NRE, call .run(...).")
        return self._posterior

    def cont(self):
        pass

    def run(
        self,
        Ninit: int = 3000,
        train_args: dict = {},
        head=DefaultHead,
        tail=DefaultTail,
        head_args: dict = {},
        tail_args: dict = {},
        density_factor: float = 2.0,
        volume_conv_th: float = 0.1,
        max_rounds: int = 10,
        Nmax: int = 100000,
        keep_history: bool = False,
    ):
        """Perform 1-dim marginal focus fits.

        Args:
            Ninit (int): Initial number of training points.
            train_args (dict): Training keyword arguments.
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
            density_factor (float > 1): Increase of training point density per round.
            volume_conv_th (float > 0.): Volume convergence threshold.
            max_rounds (int): Maximum number of rounds per invokation of `run`, default 10.
            Nmax (int): Maximum number of training points per round.
            keep_history (bool): append the posterior and number of samples to a list every round
        """

        # TODO: Add optional param_list, and non 1d focus rounds
        param_list = self._cache.params
        D = len(param_list)

        assert density_factor > 1.0
        assert volume_conv_th > 0.0

        for _ in range(max_rounds):
            if verbosity() >= 0:
                print("NRE ROUND %i" % self._R)
            if self._R == 0:  # First round
                self._constr_prior = self._base_prior
                N = Ninit
            else:  # Later rounds
                if self._constr_prior is None:
                    self._constr_prior = self._posterior.gen_constr_prior(
                        self._obs
                    )  # Stochastic!
                v_old = self._posterior.prior.volume()
                v_new = self._constr_prior.volume()
                if np.log(v_old) - np.log(v_new) < volume_conv_th:
                    if verbosity() >= 0:
                        print("--> Posterior volume is converged. <--")
                    #self._converged = True
                    break  # break while loop
                # Increase number of training data points systematically
                density_old = self._N / v_old ** (1 / D)
                density_new = density_factor * density_old
                N = min(max(density_new * v_new ** (1 / D), self._N), Nmax)
                if verbosity() >= 2:
                    print("  new (old) prior volume = %.4g (%.4g)" % (v_new, v_old))

            if verbosity() >= 1:
                print("  number of training samples is N =", N)

            try:
                posterior = self._amortize(
                    self._constr_prior,
                    param_list,
                    head=head,
                    tail=tail,
                    head_args=head_args,
                    tail_args=tail_args,
                    train_args=train_args,
                    N=N,
                )
            except MissingModelError:
                if verbosity() >= 1:
                    print(
                        "! Run `.simulate(model)` on cache object or specify model. !"
                    )
                break

            if keep_history:
                self._history.append(dict(posterior=posterior, N=N))

            # Update object state
            self._posterior = posterior
            self._constr_prior = None  # Reset
            self._N = N
            self._R += 1

        #self._converged = True
        #if verbosity() >= 0:
        #    print("--> Reached maximum number of rounds. <--")

    # NOTE: By convention properties are only quantites that we save in state_dict
    def requires_sim(self):
        return self._cache.requires_sim

    def gen_1d_marginals(
        self,
        params=None,
        N=1000,
        train_args={},
        head=DefaultHead,
        tail=DefaultTail,
        head_args={},
        tail_args={},
    ):
        """Convenience function to generate 1d marginals."""
        param_list = format_param_list(params, all_params=self._cache.params, mode="1d")
        if verbosity() >= 1:
            print("Generating marginals for:", param_list)
        return self.gen_custom_marginals(
            param_list,
            N=N,
            train_args=train_args,
            head=head,
            tail=tail,
            head_args=head_args,
            tail_args=tail_args,
        )

    def gen_2d_marginals(
        self,
        params=None,
        N=1000,
        train_args={},
        head=DefaultHead,
        tail=DefaultTail,
        head_args={},
        tail_args={},
    ):
        """Convenience function to generate 2d marginals."""
        param_list = format_param_list(params, all_params=self._cache.params, mode="2d")
        if verbosity() >= 1:
            print("Generating marginals for:", param_list)
        return self.gen_custom_marginals(
            param_list,
            N=N,
            train_args=train_args,
            head=head,
            tail=tail,
            head_args=head_args,
            tail_args=tail_args,
        )

    def gen_custom_marginals(
        self,
        param_list,
        N=1000,
        train_args={},
        head=DefaultHead,
        tail=DefaultTail,
        head_args={},
        tail_args={},
    ):
        """Perform custom 2-dim posterior estimation.

        Args:
            param_list (list of tuples of strings): List of parameters for which inference is performed.
            N (int): Number of training points.
            train_args (dict): Training keyword arguments.
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
        """
        if self._posterior is None:
            prior = self._base_prior
        else:
            prior = self._posterior.prior
            if verbosity() >= 1:
                print("Using volume:", prior.volume())

        param_list = format_param_list(param_list, all_params=self._cache.params)

        posterior = self._amortize(
            prior=prior,
            N=N,
            param_list=param_list,
            head=head,
            tail=tail,
            head_args=head_args,
            tail_args=tail_args,
            train_args=train_args,
        )
        return posterior

    @property
    def cache(self):
        """Return cache."""
        return self._cache

    @property
    def state_dict(self):
        """Return state dict."""
        return dict(
            base_prior=self._base_prior.state_dict(),
            posterior=self._posterior.state_dict(),
            obs=self._obs,
        )

    @classmethod
    def from_state_dict(cls, state_dict, model, noise=None, cache=None, device="cpu"):
        """Instantiate from state dict."""
        base_prior = Prior.from_state_dict(state_dict["base_prior"])
        posterior = Marginals.from_state_dict(state_dict["posterior"])
        obs = state_dict["obs"]

        nr = NestedRatios(
            model, base_prior, obs, noise=noise, cache=cache, device=device
        )
        nr._posterior = posterior
        return nr

    def _amortize(
        self, prior, param_list, N, train_args, head, tail, head_args, tail_args,
    ):

        self._cache.grow(prior, N)
        if self._cache.requires_sim:
            if self._model is not None:
                self._cache.simulate(self._model)
            else:
                raise MissingModelError("Model not defined.")
        indices = self._cache.sample(prior, N)
        points = Points(indices, self._cache, self._noise)

        if param_list is None:
            param_list = prior.params()

        re = RatioEstimator(
            param_list,
            device=self._device,
            head=head,
            tail=tail,
            tail_args=tail_args,
            head_args=head_args,
        )
        re.train(points, **train_args)

        return Marginals(re, prior)
