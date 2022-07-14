import swyft.lightning as sl
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import powerbox as pb
import swyft

class Gaussian(sl.SwyftModel):
    def __init__(self, sigma=0.1, nbins = 100, bounds = 0.01):
        super().__init__()
        self.sigma = sigma
        self.nbins = nbins
        self.low = -np.ones(2)*bounds
        self.high = np.ones(2)*bounds
    
    def fast(self, S):
        d = S['mu'] + torch.randn_like(S['mu'])*self.sigma
        return sl.SampleStore(data = d)
    
    def blob(self, z):
        x = torch.linspace(-1, 1, self.nbins)
        X, Y = torch.meshgrid(x, x)
        R = ((X-z[0])**2+(Y-z[1])**2)**0.5
        mu = np.exp(-R**2)
        return mu
    
    def chi2(self, z0, z1):
        mu0 = simulator.blob(z0)
        mu1 = simulator.blob(z1)
        chi2 = ((mu1-mu0)**2).sum()/self.sigma**2
        return chi2
        
    def slow(self, S):
        z = S['z']
        mu = self.blob(z)
        return sl.SampleStore(mu=mu.float())

    def prior(self, N, bounds = None):
        draw = np.array([np.random.uniform(low=self.low, high=self.high) for _ in range(N)])
        return sl.SampleStore(z = torch.tensor(draw).float())

class Polynomial(sl.SwyftModel):
    def __init__(self, sigma=0.1, nbins = 1000):
        super().__init__()
        self.sigma = sigma
        self.nbins = nbins
    
    def fast(self, S):
        d = S['mu'] + torch.randn_like(S['mu'])*self.sigma
        return sl.SampleStore(data = d)
        
    def slow(self, S):
        a, b, c = S['z']
        x = np.linspace(-1, 1, self.nbins)
        mu = a+b*x + c*x**2
        return sl.SampleStore(mu=mu.float())

    def prior(self, N, bounds = None):
        low = -np.ones(3)
        high = np.ones(3)
        if bounds is not None:
            low, high = bounds['z'].low, bounds['z'].high
        draw = np.array([np.random.uniform(low=low, high=high) for _ in range(N)])
        return sl.SampleStore(z = torch.tensor(draw).float())

class Model(sl.SwyftModel):
    def __init__(self, bounds = None, seed = None):
        super().__init__()
        self.bounds = bounds
        l = np.linspace(-40, 40, 256)
        b = np.linspace(-40, 40, 256)
        self.L, self.B = np.meshgrid(l, b)
        self.seed = seed
        
    def psc1(self):
        L, B = self.L, self.B
        rho = np.exp(-(L/20)**2 -(B/3)**2)
        rho /= rho.sum()
        p = rho.flatten()
        idx = np.random.choice(len(p), p=p, size = 1000)
        m = np.zeros(len(p))
        Lu = np.random.lognormal(1.0, 2.0, size = len(idx))
        m[idx] += Lu
        m = m.reshape(rho.shape)
        return m
    
    def psc2(self):
        L, B = self.L, self.B
        #rho = np.exp(-(L/20)**2 -(B/3)**2)
        rho = np.ones_like(L)
        rho /= rho.sum()
        p = rho.flatten()
        idx = np.random.choice(len(p), p=p, size = 1000)
        m = np.zeros(len(p))
        Lu = np.random.lognormal(1.0, 2.0, size = len(idx))
        m[idx] += Lu
        m = m.reshape(rho.shape)
        return m
    
    def pi0(self):
        L, B = self.L, self.B
        p = pb.PowerBox(N = 256, dim = 2, pk = lambda k: 0.1*k**-2.3, seed = self.seed)
        x = p.delta_x()
        x = np.exp(x*2)
        x *= np.exp(-(L/40)**2)*(np.exp(-(abs(B)/8)**2) + np.exp(-(abs(B)/2.5)**2) + np.exp(-(abs(B)/1)**2))
        return x
    
    def ICS(self):
        L, B = self.L, self.B
        p = pb.PowerBox(N = 256, dim = 2, pk = lambda k: 0.1*k**-3, seed = self.seed)
        x = p.delta_x()
        x = np.exp(x*2)
        x *= np.exp(-(L/40)**2)*(np.exp(-(abs(B)/12)**2) + np.exp(-(abs(B)/4)**2) + np.exp(-(abs(B)/2)**2))
        return x
    
    def ISO(self):
        x = np.ones_like(self.L)
        return x
        
    def fast(self, pars):
        x = pars['x']
        data = np.random.poisson(x*10)
        #data = torch.poisson(x*10)
        return sl.SampleStore(dict(data = data))
    
    def slow(self, pars):
        z = pars['z'].numpy()
        mu = self.psc1()*z[0]*2
        mu += self.pi0()*z[1]
        mu += self.ICS()*z[2]
        mu += self.ISO()*z[3]
        mu += self.psc2()*z[4]
        x = gaussian_filter(mu, 1)
        x = x.astype(np.float32)
        return sl.SampleStore(dict(x = x))
        
    def prior(self, N, bounds = None):
        bounds = self.bounds
        #print(bounds)
        #z = np.random.rand(N, 5).astype(np.float32)
        if bounds is None:
            low = np.zeros(5)
            high = np.ones(5)
        else:
            low, high = bounds['z'].low, bounds['z'].high
        z = np.random.uniform(low, high, size=(N, 5)).astype(np.float32)
        z = torch.tensor(z)
        return sl.SampleStore(dict(z=z))

def get_simulator(name):
    if name == "Model":
        return Model
    else:
        raise KeyError

