import numpy as np

import swyft

prior_FermiV1 = swyft.Prior(
    {
        "ox": ["uniform", 0.0, 10.0],
        "oy": ["uniform", 0.0, 10.0],
        "a": ["uniform", 1.0, 2.0],
        "p1": ["uniform", 0.0, 0.5],
        "p2": ["uniform", 1.0, 2.0],
    }
)


def simulator_FermiV1(a, ox, oy, p1, p2, sigma=0.1):
    x = np.linspace(-5, 5, 50, 50)
    X, Y = np.meshgrid(x, x)

    diff = np.cos(X + ox) * np.cos(Y + oy) * a + 2

    p = np.random.randn(*X.shape) * p1 - 0.3
    psc = 10**p * p2
    n = np.random.randn(*X.shape) * sigma
    mu = diff * 5 + psc + n
    return mu


def model_FermiV1(params):
    mu = simulator_FermiV1(
        params["a"], params["ox"], params["oy"], params["p1"], params["p2"]
    )
    return dict(mu=mu)
