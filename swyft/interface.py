from .cache import DirectoryCache, MemoryCache
from .estimation import RatioEstimator, Points
from .network import DefaultHead, DefaultTail
from .utils import format_param_list
from .intensity import Prior

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

    def __call__(self, obs, n_samples = 100000):
        """Return weighted posterior samples.

        Args:
            obs (dict): Observation.
            n_samples (int): Number of samples.

        Returns:
            dict containing samples.

        Note: Observations must be restricted to constrained prior space to
        lead to valid results.
        """
        return self._re.posterior(obs, self._prior, n_samples = n_samples)

    def state_dict(self):
        """Return state_dict."""
        return dict(re = self._re.state_dict(), prior = self._prior.state_dict())

    @classmethod
    def from_state_dict(cls, state_dict):
        """Instantiate posterior based on state_dict."""
        return Marginals(
                RatioEstimator.from_state_dict(state_dict['re']),
                Prior.from_state_dict(state_dict['prior'])
                )

    def gen_constr_prior(self, obs, th = -10):
        """Generate constrained prior based on ratio estimator.

        Args:
            obs (dict): Observation.
            th (float): Cutoff maximum log likelihood ratio. Default is -10,
                        which correspond roughly to 4 sigma.

        Returns:
            Prior: Constrained prior.
        """
        return self._prior.get_masked(obs, self._re, th = th)


class NestedRatios:
    """Main SWYFT interface class."""
    def __init__(self, model, prior, obs, noise = None, cache = None, device = 'cpu'):
        """Initialize swyft.

        Args:
            model (function): Simulator function.
            prior (Prior): Prior model.
            obs (dict): Target observation (can be None for amortized inference).
            noise (function): Noise model, optional.
            cache (Cache): Storage for simulator results.  If none, create MemoryCache.
            device (str): Device.
        """
        # Not stored
        self._model = model
        self._noise = noise
        self._obs = obs
        if cache is None:
            cache = MemoryCache.from_simulator(model, prior, noise = noise)
        self._cache = cache
        self._device = device

        # Stored in state_dict()
        self._prior = prior  # Initial prior
        self._posterior = None  # Available after training

    @property
    def obs(self):
        return self._obs

    @property
    def marginals(self):
        if self._posterior is None:
            print("NOTE: To generated marginals from NRE, call .run(...).")
        return self._posterior

    @property
    def prior(self):
        return self._prior

    def run(self, Ninit = 3000, train_args={}, head=DefaultHead,
            tail=DefaultTail, head_args={}, tail_args={}, f = 2, vr = 0.9,
            max_rounds = 10, Nmax = 100000, keep_history = False, vmax = 0.5):
        """Perform 1-dim marginal focus fits.

        Args:
            Ninit (int): Initial number of training points.
            train_args (dict): Training keyword arguments.
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
            f (float > 1): Increase training point density per round, default 2.
            vr (float < 1): Stopping threshold for fractional volume decrease.
            max_rounds (int): Maximum number of rounds, default 10.
            Nmax (int): Maximum number of training points per round.
        """

        # TODO: Add optional param_list, and non 1d focus rounds
        param_list = self._cache.params

        if self._posterior is not None:
            print("Nothing to do.")
            return

        assert vr < 1.
        assert f > 1.

        prior = self._prior

        N = Ninit
        for r in range(max_rounds):
            print("N =", N)
            posterior = self._amortize(prior, param_list, head = head, tail = tail, head_args = head_args,
                    tail_args = tail_args, train_args = train_args, N = N)

            v_old = prior.volume()
            prior = posterior.gen_constr_prior(self._obs)
            v_new = prior.volume()

            if keep_history:
                self._history.append(dict(
                    posterior=posterior))

            self._posterior = posterior

            print("New prior volume:", v_new)
            print("Constrained prior volume decreased by factor", v_new/v_old)

            print("Done!", v_old, v_new, vmax, vr)
            if (v_new/v_old > vr) and (v_new < vmax):
                print("Done!", v_old, v_new, vmax, vr)
                break  # Break loop if good enough
            else:
                # Increase number of training data points systematically
                N = min(int(N*(1+(f-1)*(v_new/v_old))), Nmax)

    def gen_1d_marginals(self, params = None, N = 1000, train_args={}, head=DefaultHead, tail=DefaultTail, head_args={}, tail_args={}):
        """Convenience function to generate 1d marginals."""
        param_list = format_param_list(params, all_params = self._cache.params, mode = '1d')
        print("Generating marginals for:", param_list)
        return self.gen_custom_marginals(param_list, N = N, train_args=train_args, head=head, tail=tail, head_args=head_args, tail_args=tail_args)

    def gen_2d_marginals(self, params = None, N = 1000, train_args={}, head=DefaultHead, tail=DefaultTail, head_args={}, tail_args={}):
        """Convenience function to generate 2d marginals."""
        param_list = format_param_list(params, all_params = self._cache.params, mode = '2d')
        print("Generating marginals for:", param_list)
        return self.gen_custom_marginals(param_list, N = N, train_args=train_args, head=head, tail=tail, head_args=head_args, tail_args=tail_args)

    def gen_custom_marginals(self, param_list, N = 1000, train_args={}, head=DefaultHead, tail=DefaultTail, head_args={}, tail_args={}):
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
            prior = self._prior
        else:
            prior = self._posterior.prior
            print("Using volume:", prior.volume())

        param_list = format_param_list(param_list, all_params = self._cache.params)

        posterior = self._amortize(
                        prior = prior,
                        N = N,
                        param_list = param_list,
                        head = head,
                        tail = tail,
                        head_args = head_args,
                        tail_args = tail_args,
                        train_args = train_args
                      )
        return posterior
        
    def cache(self):
        """Return cache."""
        return self._cache

    def state_dict(self):
        """Return state dict."""
        return dict(
                prior = self._prior.state_dict(),
                posterior = self._posterior.state_dict(),
                obs = self._obs,
                )

    @classmethod
    def from_state_dict(cls, state_dict, model, noise = None, cache = None, device = 'cpu'):
        """Instantiate from state dict."""
        prior = Prior.from_state_dict(state_dict['prior'])
        posterior = Marginals.from_state_dict(state_dict['posterior'])
        obs = state_dict['obs']

        nr = NestedRatios(model, prior, obs, noise = noise, cache = cache, device = device)
        nr._posterior = posterior
        return nr

    def _amortize(self, prior, param_list = None, N = 3000,
            train_args = {}, head = DefaultHead, tail = DefaultTail, head_args =
            {}, tail_args = {}):

        self._cache.grow(prior, N)
        self._cache.simulate(self._model)
        indices = self._cache.sample(prior, N)
        points = Points(indices, self._cache, self._noise)

        if param_list is None:
            param_list = prior.params()

        re = RatioEstimator(param_list, device=self._device, head = head, tail = tail, tail_args = tail_args, head_args = head_args)
        re.train(points, **train_args)

        return Marginals(re, prior)
