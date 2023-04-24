import numpy as np
from scipy import stats
import swyft
import torch

class SimulatorLinePattern(swyft.Simulator):
    def __init__(self, Npix = 256, bounds = None):
        super().__init__()
        self.transform_samples = swyft.to_numpy32
        g = np.linspace(-1, 1, Npix)
        self.grid = np.meshgrid(g, g)
        self.z_sampler = swyft.RectBoundSampler([
            stats.uniform(-1, 2),
            stats.uniform(-1, 2),
            stats.uniform(-1, 2),
            stats.uniform(-1, 2),
            stats.uniform(0.1, 0.3)
        ], bounds = bounds)
        
    def templates(self, z):
        w = z[4]
        t1 = np.exp(-np.sin((self.grid[0]-z[0]+self.grid[1]-z[1])/0.02)**2/0.2)
        m1 = (self.grid[0]>z[0]-w)*(self.grid[0]<z[0]+w)*(self.grid[1]>z[1]-w)*(self.grid[1]<z[1]+w)
        t2 = np.exp(-np.sin((self.grid[1]-z[3])/0.016)**2/0.4)
        m2 = (self.grid[0]>z[2]-w)*(self.grid[0]<z[2]+w)*(self.grid[1]>z[3]-w)*(self.grid[1]<z[3]+w)
        return t1*m1 + t2*m2
    
    def build(self, graph):
        z = graph.node('z', self.z_sampler)
        mu = graph.node('mu', self.templates, z)
        
        
class SimulatorBlob(swyft.Simulator):
    def __init__(self, bounds = None, Npix = 64):
        super().__init__()
        self.transform_samples = swyft.to_numpy32
        self.bounds = bounds
        self.weights, self.dist, self.Cov = self.setup_cov(Npix = Npix)
        self.Npix = Npix

    def setup_cov(self, Npix = 64):
        N = Npix**2
        l = torch.linspace(-1, 1, Npix)
        L1, L2 = torch.meshgrid(l, l)
        L = torch.stack([L1, L2], dim = -1)
        T = L.unsqueeze(0).unsqueeze(0) - L.unsqueeze(2).unsqueeze(2)
        T = (T**2).sum(-1)**0.5
        T = T.view(N, N)
        Cov = torch.exp(-T/.5)*0.5 + torch.exp(-T/0.5)*.25 + torch.exp(-T/2)*.125
        Cov *= 2
        dist = torch.distributions.MultivariateNormal(torch.zeros(N), Cov)
        R = (L1**2+L2**2)**0.5
        weights = torch.exp(-0.5*(R-0.5)**2/0.1**2)*0 + 1
        return weights, dist, Cov
        
    def sample_GP(self):
        if self.bounds is None:
            return self.dist.sample(torch.Size([1]))[0].numpy().reshape(self.Npix, self.Npix)
        else:
            i = np.random.randint(len(self.bounds))
            return self.bounds[i]
            
    def build(self, graph):
        z = graph.node("z", lambda: self.sample_GP())
        mu = graph.node("mu", lambda z: self.weights*np.exp(z), z)