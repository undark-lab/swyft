from .cache import DirectoryCache, MemoryCache
from .estimation import RatioEstimator, Points
from .intensity import Prior
from .network import OnlineNormalizationLayer, DefaultHead, DefaultTail
from .plot import corner
from .utils import Module, corner_params

__all__ = [
    "Prior",
    "Module",
    "DirectoryCache",
    "DefaultHead",
    "DefaultTail",
    "OnlineNormalizationLayer",
    "MemoryCache",
    "RatioEstimator",
    "Points",
    "corner",
]


class SWYFT:
    """Main SWYFT interface class."""
    def __init__(self, model, prior, noise = None, cache = None, obs = None, device = 'cpu'):
        """Initialize swyft.

        Args:
            model (function): Simulator function.
            noise (function): Noise model.
            prior (Prior): Prior model.
            cache (Cache): Storage for simulator results.
            obs (dict): Target observation (can be None for amortized inference).
            device (str): Device.
        """
        # Specific
        self._model = model
        self._prior = prior
        self._noise = noise
        self._obs = obs

        if cache is None:
            cache = MemoryCache.from_simulator(model, prior, noise = noise)
        self._cache = cache

        self._device = device
        
        # Rounds
        self._priors = [prior]
        self._ratios = []
        self._metadata = []

    def _round(self, N = 3000, train_args = {}, params_list  = None, head = DefaultHead, tail = DefaultTail,
             head_args = {}, tail_args = {}):
        print("Round:", len(self._ratios)+1)

        prior = self._priors[-1]  # Using last prior in the list

        # Generate simulations
        self._cache.grow(prior, N)
        self._cache.simulate(self._model)

        # And extract them
        indices = self._cache.sample(prior, N)
        points = Points(indices, self._cache, self._noise)

        # Training!
        re = RatioEstimator(params_list, device=self._device, head = head, tail = tail, tail_args = tail_args, head_args = head_args)
        re.train(points, **train_args)

        # Done!
        new_prior = prior.get_masked(self._obs, re, th = -10)
        self._priors.append(new_prior)
        self._ratios.append(re)
        self.meta.append(dict(N=N))
        
    def infer1d(self, Ninit = 3000, train_args={}, head=DefaultHead, tail=DefaultTail, head_args={}, tail_args={}, f = 1.5, vr = 0.9, max_rounds = 10, Nmax = 30000):
        """Perform 1-dim zoom in inference.

        Args:
            Ninit (int): Number of initial training points.
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
            f (float > 1): Maximum increase of training data each round.
            vr (float < 1): Threshold constrained posterior volume reduction as stopping criterion.
            max_rounds (int): Maximum number of rounds
            Nmax (int): Maximum size of training data per round.
        """

        params_list = self._cache.params
        assert vr < 1.
        assert f >= 1.
        N = Ninit
        for r in range(max_rounds):
            print("N =", N)
            self._round(N, 
                       params_list = params_list,
                       head = head,
                       tail = tail,
                       train_args = train_args,
                       head_args = head_args,
                       tail_args = tail_args
                      )
            volume_ratio = self._priors[-1].volume()/self._priors[-2].volume()
            print("Volume shrinkage:", volume_ratio)
            if volume_ratio > vr:
                break

            N = min(int(N*(1+(f-1)*volume_ratio)), Nmax)
            
    
    def infer2d(self, N = None, params = None, train_args={}, head=DefaultHead, tail=DefaultTail, head_args={}, tail_args={}):
        """Perform one round of 2-dim posterior estimation.
        
        Args:
            params (list of str): List of parameters for which inference is performed.
            ...
        """
        params = self._cache.params if params is None else params
        N = self.meta[-1]["N"] if N is None else N
        print("N =", N)

        params_list = corner_params(self._cache.params)

        self._round(N, 
                       params_list = params_list,
                       head = head,
                       tail = tail,
                       head_args = head_args,
                       tail_args = tail_args,
                       train_args = train_args
                      )
        
    def posteriors(self, n_samples = 100000, nround= -1):
        """Returns weighted posterior samples."""
        re = self._ratios[nround]
        prior = self._priors[nround]
        post = re.posterior(self._obs, prior, n_samples = n_samples)
        return post

    def __len__(self):
        """Return number of rounds."""
        return len(self._ratios)

    def __getitem__(self, i):
        """Get details of specific round."""
        return dict(prior=self._priors[i], ratio=self._ratios[i], meta=self.meta[i])

    def pop(self):
        """Pop last round."""
        self._ratios = self._ratios[:-1]
        self._priors = self._priors[:-1]
        self._priors = self.meta[:-1]

#    def state_dict(self):
#        return dict(
#                priors = [p.state_dict() for p in self._priors],
#                ratios = [r.state_dict() for r in self._ratios],
#                metadata = self._metadata
#                obs = self._obs,
#                )
#
#    @classmethod
#    def from_state_dict(cls, state_dict, model, noise = None, cache = None, device = 'cpu'):
#        SWYFT(model, prior, noise = noise, cache = cache, obs, device = device)
