import torch
    
def get_pk(x, nbins = 50):
    """Extracts the unnormalized power-spectrum based on quadratic input images."""
    N = x.shape[-1]
    freq = torch.fft.fftfreq(N)
    kx, ky = torch.meshgrid(freq, freq)
    k = (kx**2+ky**2)**0.5
    k = k.flatten()
    kedges = torch.linspace(k.min(), k.max()/2**0.5, nbins+1)
    
    fx = torch.fft.ifft2(x).flatten(start_dim=-2)
    PS = []
    for i in range(nbins):
        y = fx[..., (k>=kedges[i])*(k<kedges[i+1])]
        y = torch.abs(y)**2
        PS.append(y.mean(dim=-1).view(-1))
    PS = torch.stack(PS, dim=-1)
    return PS


class PowerSpectrumSampler:
    def __init__(self, N, boxlength = 1.0):
        self.N = N
        d = boxlength/N
        freq = torch.fft.fftfreq(N, d=d)
        kx, ky = torch.meshgrid(freq, freq)
        k = (kx**2+ky**2)**0.5
        self.k = k + k[0,1]  # Offset to avoid singularities

    def sample(self, pk):
        N = self.N
        A = (torch.randn(N,N)+1j*torch.randn(N,N))/2**0.5  # Random complex amplitudes
        phi_k = A*pk(self.k)**0.5
        phi_x = torch.fft.ifft2(phi_k, norm = 'ortho')
        return phi_x.real
    
    def get_prior_Q_factors(self, pk):
        """Return components of prior precision matrix.

        Q = UT * D * U

        Returns:
            UT, D, U: Linear operator, tensor, linear operator
        """
        # Define prior precision matrix function
        D = pk(self.k).view(-1)
        N = self.N
        U = lambda x: torch.fft.fft2(x.view(N, N), norm = 'ortho').view(N*N)
        UT = lambda x: torch.fft.ifft2(x.view(N, N), norm = 'ortho').view(N*N)
        return UT, 2/D, U  # The factor 2 comes from the fact that std results are obtained for pk = lambda k: 2.


class PowerSpectrumSampler2:
    # Using Hartley transforms, why wouldn't we do anything else?
    def __init__(self, shape):
        self.shape = shape
        freqs = []
        for n in self.shape:
            freq = torch.fft.fftfreq(n)
            freqs.append(freq)
        ks = torch.meshgrid(*freqs)
        k = sum([k1**2 for k1 in ks])**0.5
        self.k = k 
        self.hartley_dim = tuple(range(-len(self.shape), 0, 1))

    def hartley(self, x):
        # dim: Which dimensions to perform transformation on
        fx = torch.fft.fftn(x, dim = self.hartley_dim, norm = 'ortho')
        return (fx.real - fx.imag)

    def covariance_decomposition(self, pk):
        """Return components of prior covariance matrix.

        Sigma_prior = UT * D * U

        Returns:
            UT, D, U: Linear operator, tensor, linear operator
        """
        # Define prior precision matrix function
        D = pk(self.k).flatten()
        U = lambda x: self.hartley(x).flatten(-len(self.shape), -1)
        UT = lambda x: self.hartley(x.unflatten(-1, self.shape))
        return UT, D, U

    def sample(self, pk, num_samples = None):
        UT, D, _ = self.covariance_decomposition(pk)
        if num_samples is None:
            r = torch.randn(D.shape)
        else:
            r = torch.randn(num_samples, *D.shape)
        x = UT(r*D**0.5)
        return x
