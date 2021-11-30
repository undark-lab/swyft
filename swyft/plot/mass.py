from typing import Dict

import numpy as np
import pylab as plt
from scipy.stats import beta, norm

from swyft.types import Array


def plot_empirical_mass(masses: Dict[str, Array]) -> None:
    """Plot empirical vs nominal mass.

    Args:
        masses: Result from `swyft.Posteriors.empirical_mass()`

    Example::

        >>> masses = posteriors.empirical_mass()
        >>> swyft.plot_empirical_mass(mass[(0,)])  # Plot empirical mass for 1-dim posterior for parameter 0
    """
    plt.plot(1 - masses["nominal"], 1 - masses["empirical"])
    plt.xscale("log")
    plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.plot([0, 1], [0, 1], "k:")
    plt.xlabel("1-Cn, Nominal HDI level")
    plt.xlabel("Nominal HDI credible level")
    plt.ylabel("Empirical HDI credible level")
    cred = [0.683, 0.954, 0.997]
    plt.xticks(1 - np.array(cred), cred)
    plt.yticks(1 - np.array(cred), cred)


# Below is stolen from https://github.com/acole1221/swyft-CMB/blob/main/notebooks/demo-TTTEEE.ipynb


def estimate_empirical_mass(dataset, post, nobs, npost):
    obs0, v0 = dataset[0]
    obs0 = {k: v.numpy() for k, v in obs0.items()}
    # obs0 = noise({k: v.numpy() for k,v in obs0.items()})
    w0 = {
        k: np.exp(v)
        for k, v in post._eval_ratios(obs0, v0.unsqueeze(0).numpy()).items()
    }
    mass = {
        k: dict(nominal=[], empirical=np.linspace(1 / nobs, 1, nobs)) for k in w0.keys()
    }
    for _ in range(nobs):
        j = np.random.randint(len(dataset))
        obs0, v0 = dataset[j]
        obs0 = {k: v.numpy() for k, v in obs0.items()}
        # obs0 = noise({k: v.numpy() for k,v in obs0.items()})
        w0 = {
            k: np.exp(v)
            for k, v in post._eval_ratios(obs0, v0.unsqueeze(0).numpy()).items()
        }
        wS = post.sample(npost, obs0)["weights"]
        for k, v in w0.items():
            f = wS[k][wS[k] >= v].sum() / wS[k].sum()
            mass[k]["nominal"].append(f)
    for k in mass.keys():
        mass[k]["nominal"] = np.asarray(sorted(mass[k]["nominal"]))

    return mass


def empirical_mass(self, nobs: int = 1000, npost: int = 1000):
    """Estimate empirical vs nominal mass.
    Args:
        nobs: Number of mock observations for empirical mass estimate (taken randomly from dataset)
        npost: Number of posterior samples to estimate nominal mass
    Returns:
        Nominal and empirical masses.
    """
    return estimate_empirical_mass(self.dataset, self.posteriors, nobs, npost)


probit = lambda x: norm.ppf(x)
z_from_alpha = lambda alpha: probit(1 - alpha / 2)
alpha_from_z = lambda z: 2 - norm.cdf(z) * 2


def Jefferys_interval(k, n, z=1):
    alpha = alpha_from_z(z=z)
    lower = beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
    upper = beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
    return np.array([np.where(k > 0, lower, 0.0), np.where(k < n, upper, 1.0)]).T


def estimate_hat_z(masses, nbins=50, zmax=4, z_band=1):
    n = len(masses)
    zlist = np.linspace(0, zmax, nbins)
    tlist = 1 - alpha_from_z(zlist)
    k = np.array([sum(masses < t) for t in tlist])
    r_mean = k / n
    r_band = Jefferys_interval(k, n, z=z_band)
    z_mean = z_from_alpha(1 - r_mean)
    z_band = z_from_alpha(1 - r_band)
    return dict(z=zlist, mean=z_mean, upper=z_band[:, 1], lower=z_band[:, 0])


def plot_band(m, zmax=3.5):
    plt.figure(figsize=(4, 4))
    hat_z = estimate_hat_z(m, zmax=zmax)
    z = hat_z["z"]
    plt.plot(z, hat_z["mean"], "k")
    upper = hat_z["upper"]
    upper = np.where(upper == np.inf, 100.0, upper)
    plt.fill_between(z, hat_z["lower"], upper, color="0.8")
    plt.plot([0, 4], [0, 4], "--", color="darkgreen")
    for t in range(1, int(zmax) + 1):
        l = np.interp(t, z, hat_z["mean"])
        if l != np.inf:
            plt.plot([0, t], [l, l], ":", color="r")
            c = 1 - alpha_from_z(l)
            plt.text(0.1, l + 0.05, ("%.2f" % (c * 100)) + "%")
            plt.plot([t, t], [0, l], ":", color="r")
        else:
            plt.plot([t, t], [0, 10.0], ":", color="r")
        c = 1 - alpha_from_z(t)
        plt.text(t, 0.3, ("%.2f" % (c * 100)) + "%", rotation=-90)
    plt.xlim([0, zmax])
    plt.ylim([0, zmax + 0.5])
    plt.ylabel("Empirical coverage, $\hat z$")
    plt.xlabel("Confidence level, $z$")
    phi = 40
    plt.text(
        zmax / 2,
        zmax / 2 + 0.4,
        "Conservative",
        ha="center",
        va="center",
        rotation=phi,
        color="darkgreen",
    )
    plt.text(
        zmax / 2 + 0.4,
        zmax / 2,
        "Overconfident",
        ha="center",
        va="center",
        rotation=phi,
        color="darkgreen",
    )


if __name__ == "__main__":
    pass
