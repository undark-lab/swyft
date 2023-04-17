import numpy as np
from scipy import stats
import swyft

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