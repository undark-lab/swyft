from pathlib import Path
from typing import Sequence

import numpy as np
import scipy
from scipy.integrate import simps

from swyft.types import Array, PathType


def get_obs_shapes(obs):
    return {k: v.shape for k, v in obs.items()}


def depth(seq: Sequence):
    if seq and isinstance(seq, str):
        return 0
    elif seq and isinstance(seq, Sequence):
        return 1 + max(depth(item) for item in seq)
    else:
        return 0


def is_empty(directory: PathType):
    directory = Path(directory)
    if next(directory.iterdir(), None) is None:
        return True
    else:
        return False


def get_stats(z, p):
    # Returns central credible intervals
    zmax = z[p.argmax()]
    c = scipy.integrate.cumtrapz(p, z, initial=0)
    res = np.interp([0.025, 0.16, 0.5, 0.84, 0.975], c, z)
    xmedian = res[2]
    xerr68 = [res[1], res[3]]
    xerr95 = [res[0], res[4]]
    return {
        "mode": zmax,
        "median": xmedian,
        "cred68": xerr68,
        "cred95": xerr95,
        "err68": (xerr68[1] - xerr68[0]) / 2,
        "err95": (xerr95[1] - xerr95[0]) / 2,
    }


def cred1d(re, x0: Array):
    """Calculate credible regions.

    Args:
        re (RatioEstimator): ratio estimators
        x0: true observation
    """
    zdim = re.zdim
    for i in range(zdim):
        z, p = re.posterior(x0, i)
        res = get_stats(z, p)
        print("z%i = %.5f +- %.5f" % (i, res["median"], res["err68"]))


# FIXME: Norm is not informative; need to multiply by constrained prior density
def grid_interpolate_samples(x, y, bins=1000, return_norm=False):
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    x_grid = np.linspace(x[0], x[-1], bins)
    y_grid = np.interp(x_grid, x, y)
    norm = simps(y_grid, x_grid)
    y_grid_normed = y_grid / norm
    if return_norm:
        return x_grid, y_grid_normed, norm
    else:
        return x_grid, y_grid_normed


def get_entropy_1d(x, y, y_true=None, x_true=None, bins=1000):
    """Estimate 1-dim entropy, norm, KL divergence and p-value.

    Args:
        x (Array): x-values
        y (Array): probability density y = p(x)
        y_true (function): functional form of the true probability density for KL calculation
        bins (int): Number of bins to use for interpolation.
    """
    x_int, y_int, norm = grid_interpolate_samples(x, y, bins=bins, return_norm=True)
    entropy = -simps(y_int * np.log(y_int), x_int)
    result = dict(norm=norm, entropy=entropy)
    if y_true is not None:
        y_int_true = y_true(x_int)
        KL = simps(y_int * np.log(y_int / y_int_true), x_int)
        result["KL"] = KL
    if x_true is not None:
        y_sorted = np.sort(y_int)[::-1]  # Sort backwards
        total_mass = y_sorted.sum()
        enclosed_mass = np.cumsum(y_sorted)
        y_at_x_true = np.interp(x_true, x_int, y_int)
        cont_mass = np.interp(
            y_at_x_true, y_sorted[::-1], enclosed_mass[::-1] / total_mass
        )
        result["cont_mass"] = cont_mass
    return result


def sample_diagnostics(samples, true_posteriors={}, true_params={}):
    result = {}
    for params in samples["weights"].keys():
        if len(params) > 1:
            continue
        else:  # 1-dim case
            x = samples["params"][params[0]]
            y = samples["weights"][params]
            if params in true_posteriors.keys():
                y_true = true_posteriors[params]
            else:
                y_true = None
            if params[0] in true_params.keys():
                x_true = true_params[params[0]]
            else:
                x_true = None
            result[params] = get_entropy_1d(x, y, y_true=y_true, x_true=x_true)
    return result


def estimate_coverage(
    marginals,
    dataset,
    nrounds=10,
    nsamples=1000,
    cred_level=[0.68268, 0.95450, 0.99730],
):
    """Estimate coverage of amortized marginals for dataset.

    Args:
        marginals (RatioEstimatedPosterior): Marginals of interest.
        dataset (Dataset): Test dataset within the support of the marginals constrained prior.
        nrounds (int): Noise realizations for each test point.
        nsamples (int): Number of marginal samples used for the calculations.
        cred_level (list): Credible levels.

    NOTE: This algorithm assumes factorized indicator functions of the constrained priors, to accelerate posterior calculations.
    NOTE: Only works for 1-dim marginals right now.
    """
    diags = []
    for _ in range(nrounds):
        for point in dataset:
            samples = marginals(point["obs"], nsamples)
            diag = sample_diagnostics(samples, true_params=point["par"])
            diags.append(diag)
    cont_mass = {key[0]: [v[key]["cont_mass"] for v in diags] for key in diag.keys()}
    params = list(cont_mass.keys())
    cont_fraction = {
        k: [sum(np.array(cont_mass[k]) < c) / len(cont_mass[k]) for c in cred_level]
        for k in params
    }
    return cont_fraction


if __name__ == "__main__":
    pass
