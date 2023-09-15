import numpy as np
import pylab as plt
import swyft

from scipy.integrate import simps
from swyft.lightning.utils import get_pdf
from swyft.plot.mass import plot_empirical_z_score, get_alpha

from typing import (
    Sequence,
    Union,
)


#####################
# Auxiliary functions
#####################


def _grid_interpolate_samples(x, y, bins=1000, return_norm=False):
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


def _get_HDI_thresholds(x, cred_level=[0.68268, 0.95450, 0.99730]):
    x = x.flatten()
    x = np.sort(x)[::-1]  # Sort backwards
    total_mass = x.sum()
    enclosed_mass = np.cumsum(x)
    idx = [np.argmax(enclosed_mass >= total_mass * f) for f in cred_level]
    levels = np.array(x[idx])
    return levels


def _contour1d(z, v, levels, ax=plt, linestyles=None, color=None, **kwargs):
    y0 = -1.0 * v.max()
    y1 = 5.0 * v.max()
    ax.fill_between(z, y0, y1, where=v >= levels[0], color=color, alpha=0.1)
    ax.fill_between(z, y0, y1, where=v >= levels[1], color=color, alpha=0.1)
    ax.fill_between(z, y0, y1, where=v >= levels[2], color=color, alpha=0.1)
    # if not isinstance(colors, list):
    #    colors = [colors]*len(levels)
    # for i, l in enumerate(levels):
    #    zero_crossings = np.where(np.diff(np.sign(v-l*1.001)))[0]
    #    for c in z[zero_crossings]:
    #        ax.axvline(c, ls=linestyles[i], color = colors[i], **kwargs)


def _plot_2d(
    lrs_coll,
    parname1,
    parname2,
    ax=None,
    bins=100,
    color="k",
    cmap="gray_r",
    smooth=0.0,
    cred_level=[0.68268, 0.95450, 0.99730],
    truth=None,
    smooth_prior=False,
):
    """Plot 2-dimensional posterior.

    Args:
        lrs_coll: Collection of swyft.LogRatioSamples objects
        parname1: Name of parameter 1
        parname2: Name of parameter 2
        ax: Optional figure axis argument
        bins: Number of bins used for histograms.
        color: Contour colors
        cmap: Density colors
        smooth: Applied smoothing factor
        cred_level: Credible levels for contours
        truth: Dictionary with parameters names as keys and true values
        smooth_prior: Smooth and histogram prior instead of posterior (default False)
    """
    counts, xy = get_pdf(
        lrs_coll,
        [parname1, parname2],
        bins=bins,
        smooth=smooth,
        smooth_prior=smooth_prior,
    )
    xbins = xy[:, 0]
    ybins = xy[:, 1]

    if ax is None:
        ax = plt.gca()

    #    # FIXME: use interpolation when grid_interpolate == True
    #    x = samples[:,0].numpy()
    #    y = samples[:,1].numpy()
    #    w = weights.numpy()
    #    counts, xbins, ybins, _ = ax.hist2d(x, y, weights=w, bins=bins, cmap=cmap)
    #    if smooth is not None:
    #        counts = gaussian_filter(counts, smooth)

    levels = sorted(_get_HDI_thresholds(counts, cred_level=cred_level))
    ax.contour(
        counts.T,
        extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
        levels=levels,
        linestyles=[":", "--", "-"],
        colors=color,
    )
    ax.imshow(
        counts.T,
        extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
        cmap=cmap,
        origin="lower",
        aspect="auto",
    )
    ax.set_xlim([xbins.min(), xbins.max()])
    ax.set_ylim([ybins.min(), ybins.max()])

    if truth is not None:
        if parname1 in truth.keys():
            ax.axvline(truth[parname1], color="k", lw=1.0, zorder=10, ls=(1, (5, 1)))
        if parname2 in truth.keys():
            ax.axhline(truth[parname2], color="k", lw=1.0, zorder=10, ls=(1, (5, 1)))
        if parname1 in truth.keys() and parname2 in truth.keys():
            ax.scatter(
                [truth[parname1]],
                [truth[parname2]],
                c="k",
                marker=".",
                s=100,
            )


#    xm = (xbins[:-1] + xbins[1:]) / 2
#    ym = (ybins[:-1] + ybins[1:]) / 2
#
#    cx = counts.sum(axis=1)
#    cy = counts.sum(axis=0)
#
#    mean = (sum(xm * cx) / sum(cx), sum(ym * cy) / sum(cy))
#
#    return dict(mean=mean, mode=None, HDI1=None, HDI2=None, HDI3=None, entropy=None)


def _plot_1d(
    lrs_coll,
    parname,
    ax=None,
    bins=100,
    color="k",
    contours=True,
    smooth=0.0,
    cred_level=[0.68268, 0.95450, 0.99730],
    truth=None,
    smooth_prior=False,
):
    """Plot 1-dimensional posteriors.

    Args:
        lrs_coll: Collection of swyft.LogRatioSamples objects
        parname: Name of parameter
        ax: Optional figure axis argument
        bins: Number of bins used for histograms.
        color: Contour colors
        contours: Indicate contours
        smooth: Applied smoothing factor
        cred_level: Credible levels for contours
        truth: Dictionary with parameters names as keys and true values
        smooth_prior: Smooth and histogram prior instead of posterior (default False)
    """

    v, zm = get_pdf(
        lrs_coll, parname, bins=bins, smooth=smooth, smooth_prior=smooth_prior
    )
    zm = zm[:, 0]

    if ax is None:
        ax = plt.gca()

    levels = sorted(_get_HDI_thresholds(v, cred_level=cred_level))
    if contours:
        _contour1d(zm, v, levels, ax=ax, color=color)
    ax.plot(zm, v, color=color)
    ax.set_xlim([zm.min(), zm.max()])
    ax.set_ylim([-v.max() * 0.05, v.max() * 1.1])

    if truth is not None:
        ax.axvline(truth[parname], color="k", lw=1.0, zorder=10, ls=(1, (5, 1)))


#####
# API
#####


def plot_corner(
    lrs_coll,
    parnames,
    bins=100,
    figsize=None,
    color="k",
    labels=None,
    label_args={},
    contours_1d: bool = True,
    fig=None,
    smooth=0.0,
    cred_level=[0.68268, 0.95450, 0.99730],
    truth=None,
    smooth_prior=False,
    #    plot_diagonal=True  # TODO: Implement supression of diagonals
) -> None:
    """Make a beautiful corner plot.

    Args:
        lrs_coll: Collection of swyft.LogRatioSamples objects
        parnames: List of parameters of interest
        bins: Number of bins used for histograms.
        figsize: Size of figure
        color: Color
        labels: Optional custom labels, either list or dict.
        label_args: Custom label arguments
        contours_1d: Plot 1-dim contours
        fig: Figure instance
        smooth: histogram smoothing
        cred_level: Credible levels for contours
        truth: Dictionary with parameters names as keys and true values
        smooth_prior: Smooth and histogram prior instead of posterior (default False)
    """
    K = len(parnames)
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=figsize)
    else:
        axes = np.array(fig.get_axes()).reshape((K, K))
    #    lb = 0.125
    #    tr = 0.9
    #    whspace = 0.1
    #    fig.subplots_adjust(
    #        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    #    )
    #
    #    diagnostics = {}

    if labels is None:
        labels = parnames
    elif isinstance(labels, list):
        assert len(labels) == len(
            parnames
        ), "Length of labels list must correspond to number of parameters."
    elif isinstance(labels, dict):
        labels = [labels.get(k, k) for k in parnames]
    else:
        raise ValueError("labels must be None, list or dict")

    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            # Switch off upper left triangle
            if i < j:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                continue

            # Formatting labels
            if j > 0 or i == 0:
                ax.set_yticklabels([])
                # ax.set_yticks([])
            if i < K - 1:
                ax.set_xticklabels([])
                # ax.set_xticks([])
            if i == K - 1:
                ax.set_xlabel(labels[j], **label_args)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], **label_args)

            # Set limits
            # ax.set_xlim(x_lims[j])
            # if i != j:
            #    ax.set_ylim(y_lims[i])

            # 2-dim plots
            if j < i:
                try:
                    ret = _plot_2d(
                        lrs_coll,
                        parnames[j],
                        parnames[i],
                        ax=ax,
                        color=color,
                        bins=bins,
                        smooth=smooth,
                        cred_level=cred_level,
                        truth=truth,
                        smooth_prior=smooth_prior,
                    )
                except swyft.SwyftParameterError:
                    pass

            if j == i:
                try:
                    ret = _plot_1d(
                        lrs_coll,
                        parnames[i],
                        ax=ax,
                        color=color,
                        bins=bins,
                        contours=contours_1d,
                        smooth=smooth,
                        truth=truth,
                        smooth_prior=smooth_prior,
                    )
                except swyft.SwyftParameterError:
                    pass

    # Tight things up
    fig.tight_layout()

    return fig


def plot_zz(
    coverage_samples,
    params: Union[str, Sequence[str]],
    z_max: float = 3.5,
    bins: int = 50,
    ax=None,
):
    """Make a zz plot.

    Args:
        coverage_samples: Collection of CoverageSamples object
        params: Parameters of interest
        z_max: Maximum value of z.
        bins: Number of discretization bins.
        ax: Optional axes instance.
    """
    cov = swyft.estimate_coverage(coverage_samples, params, z_max=z_max, bins=bins)
    ax = ax if ax else plt.gca()
    plot_empirical_z_score(ax, cov[:, 0], cov[:, 1], cov[:, 2:])


def plot_pp(
    coverage_samples,
    params: Union[str, Sequence[str]],
    z_max: float = 3.5,
    bins: int = 50,
    ax=None,
):
    """Make a pp plot.

    Args:
        coverage_samples: Collection of CoverageSamples object
        params: Parameters of interest
        z_max: Maximum value of z.
        bins: Number of discretization bins.
        ax: Optional axes instance.
    """
    cov = swyft.estimate_coverage(coverage_samples, params, z_max=z_max, bins=bins)
    alphas = 1 - get_alpha(cov)
    ax = ax if ax else plt.gca()
    ax.fill_between(alphas[:, 0], alphas[:, 2], alphas[:, 3], color="0.8")
    ax.plot(alphas[:, 0], alphas[:, 1], "k")
    plt.plot([0, 1], [0, 1], "g--")
    plt.xlabel("Nominal credibility [$1-p$]")
    plt.ylabel("Empirical coverage [$1-p$]")


def plot_posterior(
    lrs_coll,
    parnames=None,
    bins=100,
    figsize=None,
    color="k",
    labels=None,
    label_args={},
    ncol=None,
    subplots_kwargs={},
    fig=None,
    contours=True,
    smooth=1.0,
    cred_level=[0.68268, 0.95450, 0.99730],
    truth=None,
    smooth_prior=False,
) -> None:
    """Make beautiful 1-dim posteriors.

    Args:
        lrs_coll: Collection of swyft.LogRatioSamples objects
        parnames: (Optional) List of parameters of interest
        bins: Number of bins used for histograms.
        figsize: Optional size of figure
        color: Color
        labels: (Optional) Custom labels
        label_args: (Pptional) Custom label arguments
        ncol: (Optional) Number of panel columns
        subplots_kwargs: Optional arguments for subplots generation.
        fig: Optional figure instance
        contours: Plot 1-dim contours
        smooth: Gaussian smothing scale
        cred_level: Credible levels for contours
        truth: (Optional) Dictionary with parameters names as keys and true values
        smooth_prior: Smooth and histogram prior instead of posterior (default False)
    """

    # parnames should be single str or list of strings
    if isinstance(parnames, str):
        parnames = [parnames]

    # labels can be None (defaulting to parnames), or list of names of dictionary mapping parnames on labels
    if labels is None:
        labels = parnames
    elif isinstance(labels, list):
        assert len(labels) == len(
            parnames
        ), "Length of labels list must correspond to number of parameters."
    elif isinstance(labels, dict):
        labels = [labels.get(k, k) for k in parnames]
    else:
        raise ValueError("labels must be None, list or dict")

    # If ncol is None, default to (max) 4 panels per row
    if ncol is None:
        ncol = min(len(parnames), 4)

    K = len(parnames)
    nrow = (K - 1) // ncol + 1

    if fig is None:
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, **subplots_kwargs)
    else:
        axes = fig.get_axes()

    # Ensure axes has always the same shape
    if isinstance(axes, np.ndarray):
        axes = axes.reshape(-1)
    elif isinstance(axes, list):
        axes = np.array(axes)
    else:
        axes = np.array([axes])
        ncol = nrow = 1

    for k in range(ncol * nrow):
        ax = axes[k]
        if k >= K:
            ax.set_visible(False)
            continue
        _plot_1d(
            lrs_coll,
            parnames[k],
            ax=ax,
            bins=bins,
            color=color,
            contours=contours,
            smooth=smooth,
            cred_level=cred_level,
            truth=truth,
            smooth_prior=smooth_prior,
        )
        ax.set_xlabel(labels[k], **label_args)
        ax.set_yticks([])
        # ax.tick_params(axis='x', which='minor', bottom = True)
        ax.minorticks_on()

    # Tight things up
    fig.tight_layout()

    return fig


def plot_pair(
    lrs_coll,
    parnames=None,
    bins=100,
    figsize=None,
    color="k",
    labels=None,
    label_args={},
    ncol=None,
    subplots_kwargs={},
    fig=None,
    smooth=1.0,
    cred_level=[0.68268, 0.95450, 0.99730],
    truth=None,
    smooth_prior=False,
) -> None:
    """Make beautiful 2-dim posteriors.

    Args:
        lrs_coll: Collection of swyft.LogRatioSamples objects
        parnames: (Optional) List of parameter pairs of interest
        bins: Number of bins used for histograms.
        figsize: Optional size of figure
        color: Color
        labels: (Optional) Custom labels
        label_args: (Pptional) Custom label arguments
        ncol: (Optional) Number of panel columns
        subplots_kwargs: Optional arguments for subplots generation.
        fig: Optional figure instance
        smooth: Gaussian smothing scale
        cred_level: Credible levels for contours
        truth: (Optional) Dictionary with parameters names as keys and true values
        smooth_prior: Smooth and histogram prior instead of posterior (default False)
    """

    # parnames should be single str or list of strings
    if isinstance(parnames[0], str):
        parnames = [parnames]

    # labels can be None (defaulting to parnames), or list of names of dictionary mapping parnames on labels
    if labels is None:
        labels = parnames
    elif isinstance(labels, dict):
        labels = [
            [l[i].get(k[i], k[i]) for i in [0, 1]] for l, k in zip(labels, parnames)
        ]
    else:
        raise ValueError("labels must be None or dict")

    # If ncol is None, default to (max) 4 panels per row
    if ncol is None:
        ncol = min(len(parnames), 4)

    K = len(parnames)
    nrow = (K - 1) // ncol + 1

    if fig is None:
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, **subplots_kwargs)
    else:
        axes = fig.get_axes()

    # Ensure axes has always the same shape
    if isinstance(axes, np.ndarray):
        axes = axes.reshape(-1)
    else:
        axes = np.array([axes])
        ncol = nrow = 1

    for k in range(ncol * nrow):
        ax = axes[k]
        if k >= K:
            ax.set_visible(False)
            continue
        _plot_2d(
            lrs_coll,
            parnames[k][0],
            parnames[k][1],
            ax=ax,
            bins=bins,
            color=color,
            smooth=smooth,
            cred_level=cred_level,
            truth=truth,
            smooth_prior=smooth_prior
        )
        ax.set_xlabel(labels[k][0], **label_args)
        ax.set_ylabel(labels[k][1], **label_args)
        # ax.set_yticks([])
        # ax.tick_params(axis='x', which='minor', bottom = True)
        ax.minorticks_on()

    # Tight things up
    fig.tight_layout()
