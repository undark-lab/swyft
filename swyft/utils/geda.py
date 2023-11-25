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


class GEDASampler2:
    """Gibbs sampler for MVN based on exact data augmentation (GEDA).
    
    The underlying assumption is that the precision matrix can be split as follows:
    
    Q = Q1 + Q2
    Q1 = G1.T D1 G1  (G1 is arbitrary, D1 is diagonal*)
    Q2 = U2.T D2 U2  (U2 is unitary, D2 is diagonal)
    
    omega is a positive hyper-parameter of the algorithm
    it must obey omega < 1/||Q1|| (1/omega should be larger than the largest singular value of Q1)
    
    * In GEDA D1 just has to be positive definite.

    Motivation for GEDASampler2:

    The GEDA sampler should directly provide noise sampels from the images, without intermediate steps or transformations.

    - $U_2$, $D_2$ and $U_2^T$ are all defined in real space, going from $(N, N, N) \rightarrow (N*N*N,)$ and back
    - The same is true for $G_1$ etc.

    Consequences for GEDA:
        - $\theta$ is in image space (updated according to coupling-strength constrained prior)
        - $u_1$ is in in image space (updated according to coupling strength)
        - $u_2$ is in vector space of $Q_1$ (updated according to likelihood precision)
    """
    def __init__(self, omega, G1, D1, G1T, U2, D2, U2T, out_shape, mu = None):
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
        self.N1 = len(D1)
        self.N2 = len(D2)
        self.G1 = G1
        self.D1 = D1
        self.G1T = G1T
        self.U2 = U2
        self.D2 = D2
        self.U2T = U2T
        self.out_shape = out_shape 
        
        self.device = self.D1.device
        self.U2_dtype = torch.float64 #torch.complex128
        self.dtype = torch.float64
        
        self._r1 = torch.zeros(*self.out_shape, device = self.device, dtype = self.dtype)
        self._r2 = torch.zeros(self.N1, device = self.device, dtype = self.dtype)
        
        # Aux definitions
        self.Q1 = lambda x: self.G1T(self.D1*self.G1(x))
        self.Q2 = lambda x: self.U2T(self.D2*self.U2(x))
        self.R = lambda x: x/self.omega - self.Q1(x)
#        self.Qu = self.Q1(mu) + self.Q2(mu) if mu is not None else 0
        self.Qu = mu if mu is not None else 0
#        self.Qu = self.Q1(mu) if mu is not None else 0
        
    def _gibbs_sample_step(self, theta, u1):
        # u2 lives in Q1 vector space (in the space of D1) (B, NX*NY*NZ)
        mu_u2 = self.G1(u1)
        #print(mu_u2.mean())
        torch.randn(self.N1, out = self._r2)  
        u2 = mu_u2+self._r2/self.D1**0.5
        #print(u2.mean())

        # u1 lives in image space (space of theta)
        mu_u1 = theta - self.omega*(self.Q1(theta)-self.G1T(u2*self.D1))
        #print(mu_u1.mean())
        torch.randn(self.out_shape, out = self._r1)  
        u1 = mu_u1 + self._r1*self.omega**0.5
        #print(u1.mean())

        # mu_U2_theta lives in Q2 vector space (in the space of D2)
        #Qu = 0
        mu_U2_theta = (1/self.omega + self.D2)**-1*self.U2((self.R(u1)+self.Qu))
        #print(mu_U2_theta.mean())
        r_theta = torch.randn(self.N2, dtype = self.U2_dtype, device = self.device)
        #print(r_theta.mean())

        # theta lives in image space
        theta = self.U2T(mu_U2_theta + r_theta/(1/self.omega + self.D2)**0.5)
        #print(theta.mean())

        return theta, u1  # theta and u1 live in image space (B, NX, NY, NZ)
    
    def sample(self, N, steps = 1000, reset = False, initialize_with_Q2 = False):
        "Generate N samples"
        o_flag = 0 if initialize_with_Q2 else 1
        samples = []
        # Initialize with a random sample from Q2
        for i in range(N):
            if reset or i == 0:
                theta = self.U2T(
                        torch.randn(self.N2, dtype = self.U2_dtype, device = self.device)
                        /(o_flag/self.omega + self.D2)**0.5
                        ) # Sample from Q2
                u1 = torch.randn(*self.out_shape, device = self.device, dtype = self.dtype)*self.omega**0.5  # Sample u1 assuming theta = u2 = 0
            for _ in range(steps):
                theta, u1 = self._gibbs_sample_step(theta, u1)
            samples.append(theta)
        return samples
