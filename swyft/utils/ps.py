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
        return UT, 1/D, U
