from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats

from swyft.plot.histogram import split_corner_axes
from swyft.types import Array, MarginalToArray

# from swyft.utils.marginals import get_d_dim_marginal_indices


def get_z_score(alpha: Union[float, np.ndarray]) -> np.ndarray:
    """Recover the z_score given by `z = normal_ppd(1 - alpha / 2)`.

    Args:
        alpha: significance level

    Returns:
        z_score
    """
    return stats.norm.ppf(1 - alpha / 2)


def get_alpha(z_score: Union[float, np.ndarray]) -> np.ndarray:
    """Recover the alpha (significance level) given by `alpha = 2 * (1 - normal_cdf(z_score))`.

    Args:
        z_score: z_score aka `z`

    Returns:
        alpha: significance level
    """
    return 2 * (1 - stats.norm.cdf(z_score))


def get_jefferys_interval(
    n_success: Union[int, np.ndarray],
    n_trials: int,
    alpha: Union[float, np.ndarray] = 0.05,
) -> np.ndarray:
    """Estimate the lower and upper jefferys confidence intervals.

    Args:
        n_success: number of successes. shape=(n,)
        n_trials: number of trials
        alpha: significance level. Defaults to 0.05. shape=(n,)

    Returns:
        shape=(n, 2), lower confidence interval, upper confidence interval

    Citation:
        Brown, Lawrence D., T. Tony Cai, and Anirban DasGupta. "Interval estimation for a binomial proportion." Statistical science 16.2 (2001): 101-133.
    """
    lower = stats.beta.ppf(alpha / 2, n_success + 0.5, n_trials - n_success + 0.5)
    upper = stats.beta.ppf(1 - alpha / 2, n_success + 0.5, n_trials - n_success + 0.5)
    return np.stack(
        [
            np.where(n_success > 0, lower, 0.0),
            np.where(n_success < n_trials, upper, 1.0),
        ],
        axis=-1,
    )


def get_empirical_z_score(
    empirical_mass: Array,
    max_z_score: float,
    n_bins: int = 50,
    interval_z_score: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the empirical z-score and corresponding Jeffery's interval given mass estimates.

    Args:
        empirical_mass: empirical mass containing the parameter
        max_z_score: max limit to compute nominal z-score
        n_bins: number of nominal z-scores to compute. Defaults to 50.
        band_z_score: z-score of Jeffery's interval to plot. 1 implies one standard deviation of normal. Defaults to 1.0.

    Returns:
        nominal z-scores, mean empirical z-score, interval empirical z-score
    """
    empirical_mass = np.asarray(empirical_mass)
    n, *_ = empirical_mass.shape

    # create x axis z scores / confidences
    nominal_z_scores = np.linspace(0.0, max_z_score, n_bins)
    confidences = 1 - get_alpha(nominal_z_scores)

    # determine what counts as a success for us
    n_not_containing_truth = np.sum(empirical_mass[..., None] < confidences, axis=0)

    # compute the properties of the mean and Jeffery's interval
    mean = n_not_containing_truth / n
    interval = get_jefferys_interval(
        n_not_containing_truth, n, alpha=get_alpha(interval_z_score)
    )
    z_mean = get_z_score(1 - mean)
    z_interval = get_z_score(1 - interval)
    return nominal_z_scores, z_mean, z_interval


def plot_empirical_z_score(
    axes: Axes,
    nominal_z_scores: np.ndarray,
    z_mean: np.ndarray,
    z_interval: np.ndarray,
    mean_color: str = "black",
    interval_color: str = "0.8",
    diagonal_color: str = "darkgreen",
    sigma_color: str = "red",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xlabel: Optional[str] = r"Empirical coverage [$z_p$]",
    ylabel: Optional[str] = r"Nominal credibility [$z_p$]",
    diagonal_text: bool = False,
) -> Axes:
    """target a particular matplotlib Axes and produce an empirical coverage test plot with Jeffrey's interval

    Args:
        axes: matplotlib axes
        nominal_z_scores: sorted array of nominal z-scores
        z_mean: empirical mean of z-score estimate using a binominal distribution
        z_interval: jeffrey's interval of z-score estimate
        mean_color: color of the mean line.
        interval_color: color of the interval, floats are grey.
        diagonal_color: color of the diagonal, nominal z-score.
        sigma_color: color of the vertical and horizontal sigma lines.
        xlim: force xlim
        ylim: force ylim
        xlabel: set xlabel
        ylabel: set ylabel
        diagonal_text: turns on semantic description of above / below diagonal

    Returns:
        the matplotlib axes given
    """
    lower = z_interval[:, 0]
    upper = z_interval[:, 1]
    assert np.all(lower <= upper), "the lower interval must be <= the upper interval."
    upper = np.where(upper == np.inf, 100.0, upper)

    # empirical lines & interval
    axes.plot(nominal_z_scores, z_mean, color=mean_color)
    axes.fill_between(nominal_z_scores, lower, upper, color=interval_color)

    # diagonal line
    max_z_score = np.max(nominal_z_scores)
    axes.plot([0, max_z_score], [0, max_z_score], "--", color=diagonal_color)

    # horizontal and vertical lines, vertical are the "truth", horizontal are empirical
    for i_sigma in range(1, int(max_z_score) + 1):
        empirical_i_sigma = np.interp(i_sigma, nominal_z_scores, z_mean)
        if empirical_i_sigma != np.inf:  # when the vertical line intersects z_mean
            # Horizontal line
            axes.plot(
                [0, i_sigma],
                [empirical_i_sigma, empirical_i_sigma],
                ":",
                color=sigma_color,
            )
            # horizontal text
            c = 1 - get_alpha(empirical_i_sigma)
            axes.text(0.1, empirical_i_sigma + 0.05, ("%.2f" % (c * 100)) + "%")
            # vertical line
            axes.plot(
                [i_sigma, i_sigma], [0, empirical_i_sigma], ":", color=sigma_color
            )
            # vertical text
            c = 1 - get_alpha(i_sigma)
            axes.text(i_sigma, 0.3, ("%.2f" % (c * 100)) + "%", rotation=-90)
        else:  # when the vertical line fails to intersect z_mean
            # # horizontal line
            # axes.plot([i_sigma, i_sigma], [0, 10.0], ":", color=sigma_color)
            # # horizontal text
            # c = 1 - get_alpha(i_sigma)
            # axes.text(i_sigma, 0.3, ("%.2f" % (c * 100)) + "%", rotation=-90)
            pass

    # set labels
    axes.set_ylabel(xlabel)
    axes.set_xlabel(ylabel)

    # Add the semantic meaning of being above / below diagonal
    if diagonal_text:
        raise NotImplementedError("must add rotation description")
        phi = None
        axes.text(
            max_z_score / 2,
            max_z_score / 2 + 0.4,
            "Conservative",
            ha="center",
            va="center",
            rotation=phi,
            color="darkgreen",
        )
        axes.text(
            max_z_score / 2 + 0.4,
            max_z_score / 2,
            "Overconfident",
            ha="center",
            va="center",
            rotation=phi,
            color="darkgreen",
        )

    # set limits
    if xlim is None:
        axes.set_xlim([0, max_z_score])
    else:
        axes.set_xlim(xlim)

    if ylim is None:
        axes.set_ylim([0, max_z_score + np.round(0.15 * max_z_score, 1)])
    else:
        axes.set_ylim(ylim)
    return axes


def empirical_z_score_corner(
    empirical_mass_1d: MarginalToArray,
    empirical_mass_2d: MarginalToArray,
    max_z_score: float = 3.5,
    labels: Sequence[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    space_between_axes: float = 0.1,
) -> Tuple[Figure, Axes]:
    """create a corner plot with each subplot containing a p-p test of expected coverage probability

    Args:
        empirical_mass_1d: empirical mass dict from marginal_indices to array
        empirical_mass_2d: empirical mass dict from marginal_indices to array
        max_z_score: maximum nominal z-score to plot. Defaults to 3.5.
        labels: for labeling the x and y axes. Defaults to None.
        figsize: set figsize like in `plt.subplots`. Defaults to None.
        space_between_axes: changes the `wspace` and `hspace` between subplots. see `plt.subplots_adjust`. Defaults to 0.1.

    Returns:
        matplotlib figure, np array of matplotlib axes
    """
    # compute the dimension, necessary inds, assert all is there
    d = len(empirical_mass_1d)
    upper_inds = get_d_dim_marginal_indices(d, 2)
    assert len(empirical_mass_2d) == len(upper_inds)

    # create the plots, adjust them, remove unnecessary upper corner
    fig, axes = plt.subplots(
        nrows=d, ncols=d, sharex="col", sharey="row", figsize=figsize
    )

    _, diag, upper = split_corner_axes(axes)
    lb = 0.125
    tr = 0.9
    fig.subplots_adjust(
        left=lb,
        bottom=lb,
        right=tr,
        top=tr,
        wspace=space_between_axes,
        hspace=space_between_axes,
    )

    for ax in upper:
        ax.axis("off")

    # plot the 1d mass
    for i, (k, ax) in enumerate(zip(empirical_mass_1d.keys(), diag)):
        nominal_z_scores, z_mean, z_interval = get_empirical_z_score(
            empirical_mass_1d[k], max_z_score
        )
        plot_empirical_z_score(
            ax,
            nominal_z_scores,
            z_mean,
            z_interval,
            xlabel=None,
            ylabel=None,
        )

    # plot the 2d mass
    # for i, (k, ax) in enumerate(zip(empirical_mass_2d.keys(), upper)):
    for k, i in zip(empirical_mass_2d.keys(), upper_inds):
        a, b = i  # plot array index, upper index (lower index is transpose)
        ax = axes[b, a]  # targets the lower left corner

        nominal_z_scores, z_mean, z_interval = get_empirical_z_score(
            empirical_mass_2d[k], max_z_score
        )
        plot_empirical_z_score(
            ax,
            nominal_z_scores,
            z_mean,
            z_interval,
            xlabel=None,
            ylabel=None,
        )

    # bottom row
    for i, ax in enumerate(axes[-1, :]):
        if labels is not None:
            ax.set_xlabel(labels[i])

    # left column
    for i, ax in enumerate(axes[1:, 0], 1):
        if labels is not None:
            ax.set_ylabel(labels[i])

    fig.align_labels()
    return fig, axes


if __name__ == "__main__":
    pass
