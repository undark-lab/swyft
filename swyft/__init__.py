from .cache import Cache, DirectoryCache, MemoryCache
from .estimation import RatioEstimator, Points
from .intensity import get_unit_intensity, get_constrained_intensity
from .network import OnlineNormalizationLayer
from .plot import cont2d, plot1d, corner
from .train import get_statistics
from .utils import set_device, get_2d_combinations, cred1d

__all__ = [
    "Cache",
    "DirectoryCache",
    "MemoryCache",
    "RatioEstimator",
    "Points",
    "get_unit_intensity",
    "get_constrained_intensity",
    "OnlineNormalizationLayer",
    "cont2d",
    "plot1d",
    "corner",
    "get_statistics",
    "set_device",
    "get_2d_combinations",
    "cred1d",
    "run",
]


def run(
    x0,
    simulator,
    zdim,
    noise=None,
    cache=None,
    n_train=10000,
    n_rounds=3,
    device="cpu",
    max_epochs=10,
    batch_size=16,
    lr_schedule=[1e-3, 1e-4],
    threshold=1e-5,
    early_stopping_patience=1,
):
    """Default training loop. Possible to call just from observation x0 and simulator. Optionally, can tweak training details."""
    if cache is None:
        cache = MemoryCache(zdim=zdim, xshape=x0.shape)
    intensities = []
    res = []
    for i in range(n_rounds):
        if i == 0:
            intensity = get_unit_intensity(expected_n=n_train, dim=zdim)
        else:
            intensity = get_constrained_intensity(
                expected_n=n_train,
                ratio_estimator=res[-1],
                x0=x0,
                threshold=threshold,
            )
        intensities.append(intensity)
        cache.grow(intensities[-1])
        cache.simulate(simulator)
        points = Points(cache, intensities[-1], noise)
        re = RatioEstimator(points, device=device)
        res.append(re)
        res[-1].train(
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr_schedule=lr_schedule,
            early_stopping_patience=early_stopping_patience,
        )
    return points, res[-1]
