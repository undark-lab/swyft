import torch
import numpy as np

class GEDASampler:
    """Gibbs sampler for MVN based on exact data augmentation (GEDA).
    
    The underlying assumption is that the precision matrix can be split as follows:
    
    Q = Q1 + Q2
    Q1 = G1.T D1 G1  (G1 is arbitrary, D1 is diagonal*)
    Q2 = U2.T D2 U2  (U2 is unitary, D2 is diagonal)
    
    omega is a positive hyper-parameter of the algorithm
    it must obey omega < 1/||Q1|| (1/omega should be larger than the largest singular value of Q1)
    
    * In GEDA D1 just has to be positive definite.
    """
    def __init__(self, omega, G1, D1, G1T, U2, D2, U2T):
        """
        Arguments:
            omega: float
            G1: Callable
            D1: Vector
            G1T: Callable
            U2: Callable
            D2: Vector
            U2T: Callable
        """
        self.omega = omega
        self.N = len(D2)
        self.G1 = G1
        self.D1 = D1
        self.G1T = G1T
        self.U2 = U2
        self.D2 = D2
        self.U2T = U2T
        
        self.device = self.D1.device
        self.U2_dtype = torch.complex128
        self.dtype = torch.float64
        
        self._r2 = torch.zeros(3, self.N, device = self.device, dtype = self.dtype)
        
        # Aux definitions
        self.Q1 = lambda x: self.G1T(self.D1*self.G1(x))
        self.R = lambda x: x/self.omega - self.Q1(x)
        
#    def _gibbs_sample_step(self, theta, u1):
#        mu_u2 = self.G1(u1)
#        u2 = mu_u2+torch.randn(self.N).to(self.device)/self.D1**0.5
#        mu_u1 = theta - self.omega*(self.Q1(theta)-self.G1T(u2/self.D1))
#        u1 = mu_u1 + torch.randn(self.N).to(self.device)*self.omega**0.5
#        
#        mu_U2_theta = (1/self.omega + self.D2)**-1*self.U2((self.R(u1)+0))
#        theta = self.U2T(mu_U2_theta + torch.randn(self.N).to(self.device)/(1/self.omega + self.D2)**0.5)
#        
#        return theta, u1
    
    def _gibbs_sample_step(self, theta, u1):
        torch.randn((3, self.N), out = self._r2)
        mu_u2 = self.G1(u1)
        u2 = mu_u2+self._r2[0, :len(mu_u2)]/self.D1**0.5
        mu_u1 = theta - self.omega*(self.Q1(theta)-self.G1T(u2*self.D1))
        u1 = mu_u1 + self._r2[1]*self.omega**0.5
        mu_U2_theta = (1/self.omega + self.D2)**-1*self.U2((self.R(u1)+0))
        # TODO: This only works for Fourier transform U2 and U2T right now
        r_theta = torch.randn(self.N, dtype = self.U2_dtype, device = self.device)*1.41
        theta = self.U2T(mu_U2_theta + r_theta/(1/self.omega + self.D2)**0.5).real        
        return theta, u1
    
    def sample(self, N, steps = 1000, reset = False, initialize_with_Q2 = False):
        "Generate N samples"
        o_flag = 0 if initialize_with_Q2 else 1
        samples = []
        # Initialize with a random sample from Q2
        for i in range(N):
            if reset or i == 0:
                theta = self.U2T(torch.randn(self.N, dtype = self.U2_dtype, device = self.device)/(o_flag/self.omega + self.D2)**0.5).real # Sample from Q2
                u1 = torch.randn(self.N, device = self.device, dtype = self.dtype)*self.omega**0.5  # Sample u1 assuming theta = u2 = 0
            for _ in range(steps):
                theta, u1 = self._gibbs_sample_step(theta, u1)
            samples.append(theta)
        return samples
