from typing import Dict

import numpy as np
import pylab as plt

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


if __name__ == "__main__":
    pass
