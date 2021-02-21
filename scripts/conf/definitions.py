import numpy as np
import torch
import swyft

prior = swyft.Prior(
    {
        "ox": ["uniform", 0.0, 10.0],
        "oy": ["uniform", 0.0, 10.0],
        "a": ["uniform", 1.0, 2.0],
        "p1": ["uniform", 0.0, 0.5],
        "p2": ["uniform", 1.0, 2.0],
    }
)

def simulator(a, ox, oy, p1, p2, sigma=0.1):
    """Some examplary image simulator."""
    x = np.linspace(-5, 5, 50, 50)
    X, Y = np.meshgrid(x, x)

    diff = np.cos(X + ox) * np.cos(Y + oy) * a + 2

    p = np.random.randn(*X.shape) * p1 - 0.3
    psc = 10 ** p * p2
    n = np.random.randn(*X.shape) * sigma
    mu = diff * 5 + psc + n
    return mu

def model(params):
    """Model wrapper around simulator code."""
    mu = simulator(
        params["a"], params["ox"], params["oy"], params["p1"], params["p2"]
    )
    return dict(mu=mu)

def noise(obs, params=None, sigma=1.0):
    """Associated noise model."""
    data = {k: v + np.random.randn(*v.shape) * sigma for k, v in obs.items()}
    return data

class CustomHead(swyft.Module):
    def __init__(self, obs_shapes):
        super().__init__(obs_shapes=obs_shapes)

        self.n_features = 10

        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv2 = torch.nn.Conv2d(10, 20, 5)
        self.conv3 = torch.nn.Conv2d(20, 40, 5)
        self.pool = torch.nn.MaxPool2d(2)
        self.l = torch.nn.Linear(160, 10)

    def forward(self, obs):
        x = obs["mu"].unsqueeze(1)
        nbatch = len(x)
        # x = torch.log(0.1+x)

        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(nbatch, -1)
        x = self.l(x)

        return x

par0 = dict(ox=5.0, oy=5.0, a=1.5, p1=0.4, p2=1.1)
obs0 = noise(model(par0))
