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
    def __init__(self, model, noise, prior, cache, obs, device = 'cpu'):
        self.cache = cache
        self.model = model
        self.noise = noise
        self.obs = obs
        self.device = device
        
        self.priors = [prior]
        self.ratios = []
        self.meta = []

    def round(self, N = 1000, train_args = {}, params_list  = None, head = DefaultHead, tail = DefaultTail,
             head_args = {}, tail_args = {}):
        print("Round:", len(self.ratios)+1)

        prior = self.priors[-1]  # Using last prior in the list

        # Generate simulations
        self.cache.grow(prior, N)
        self.cache.simulate(self.model)

        # And extract them
        indices = self.cache.sample(prior, N)
        points = Points(indices, self.cache, self.noise)

        # Training!
        re = RatioEstimator(params_list, device=self.device, head = head, tail = tail, tail_args = tail_args, head_args = head_args)
        re.train(points, **train_args)

        # Done!
        new_prior = prior.get_masked(self.obs, re, th = -10)
        self.priors.append(new_prior)
        self.ratios.append(re)
        self.meta.append(dict(N=N))
        
    def infer1d(self, Ninit = 1000, train_args={}, head=DefaultHead, tail=DefaultTail, head_args={}, tail_args={}, f = 1.5, vr = 0.9, max_rounds = 10, Nmax = 30000):
        params_list = self.cache.params
        assert vr < 1.
        assert f >= 1.
        N = Ninit
        for r in range(max_rounds):
            print("N =", N)
            self.round(N, 
                       params_list = params_list,
                       head = head,
                       tail = tail,
                       train_args = train_args,
                       head_args = head_args,
                       tail_args = tail_args
                      )
            volume_ratio = self.priors[-1].volume()/self.priors[-2].volume()
            print("Volume shrinkage:", volume_ratio)
            if volume_ratio > vr:
                break

            N = min(int(N*(1+(f-1)*volume_ratio)), Nmax)
            
    
    def infer2d(self, N = None, params = None, train_args={}, head=DefaultHead, tail=DefaultTail, head_args={}, tail_args={}):
        params = self.cache.params if params is None else params
        N = self.meta[-1]["N"] if N is None else N
        print("N =", N)

        params_list = corner_params(self.cache.params)

        self.round(N, 
                       params_list = params_list,
                       head = head,
                       tail = tail,
                       head_args = head_args,
                       tail_args = tail_args
                      )
        
    def posteriors(self):
        re = self.ratios[-1]
        prior = self.priors[-1]
        post = re.posterior(self.obs, prior, n_samples = 100000)
        return post

    def __len__(self):
        return len(self.ratios)

    def pop(self):
        self.ratios = self.ratios[:-1]
        self.priors = self.priors[:-1]
