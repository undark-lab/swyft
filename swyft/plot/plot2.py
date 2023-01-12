import numpy as np
import pylab as plt
from scipy.integrate import simps
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import swyft
import swyft.lightning.utils

from typing import (
    Callable,
    Dict,
    Hashable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    Any,
)


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


def get_HDI_thresholds(x, cred_level=[0.68268, 0.95450, 0.99730]):
    x = x.flatten()
    x = np.sort(x)[::-1]  # Sort backwards
    total_mass = x.sum()
    enclosed_mass = np.cumsum(x)
    idx = [np.argmax(enclosed_mass >= total_mass * f) for f in cred_level]
    levels = np.array(x[idx])
    return levels


def plot_2d(
    logratios,
    parname1,
    parname2,
    ax=None,
    bins=100,
    color="k",
    cmap="gray_r",
    smooth=0.0,
):
    """Plot 2-dimensional posteriors."""
    counts, xy = swyft.lightning.utils.get_pdf(
        logratios, [parname1, parname2], bins=bins, smooth=smooth
    )
    xbins = xy[:, 0]
    ybins = xy[:, 1]
    #    if not isinstance(logratios, list):
    #        logratios = [logratios,]
    #
    #    samples = None
    #    for s in logratios:
    #        weighted_samples = s.get_matching_weighted_samples(parname1, parname2)
    #        if weighted_samples is not None:
    #            samples, weights = weighted_samples
    #    if samples is None:
    #        return

    if ax is None:
        ax = plt.gca()

    #    # FIXME: use interpolation when grid_interpolate == True
    #    x = samples[:,0].numpy()
    #    y = samples[:,1].numpy()
    #    w = weights.numpy()
    #    counts, xbins, ybins, _ = ax.hist2d(x, y, weights=w, bins=bins, cmap=cmap)
    #    if smooth is not None:
    #        counts = gaussian_filter(counts, smooth)

    levels = sorted(get_HDI_thresholds(counts))
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


#    xm = (xbins[:-1] + xbins[1:]) / 2
#    ym = (ybins[:-1] + ybins[1:]) / 2
#
#    cx = counts.sum(axis=1)
#    cy = counts.sum(axis=0)
#
#    mean = (sum(xm * cx) / sum(cx), sum(ym * cy) / sum(cy))
#
#    return dict(mean=mean, mode=None, HDI1=None, HDI2=None, HDI3=None, entropy=None)


def plot_1d(
    logratios,
    parname,
    weights_key=None,
    ax=None,
    grid_interpolate=False,
    bins=100,
    color="k",
    contours=True,
    smooth=0.0,
):
    """Plot 1-dimensional posteriors."""
    #    samples, weights, = swyft.get_weighted_samples(logratios, parname)

    #    if not isinstance(logratios, list):
    #        logratios = [logratios,]
    #
    #    samples = None
    #    for s in logratios:
    #        weighted_samples = s.get_matching_weighted_samples(parname)
    #        if weighted_samples is not None:
    #            samples, weights = weighted_samples
    #    if samples is None:
    #        return

    v, zm = swyft.lightning.utils.get_pdf(logratios, parname, bins=bins, smooth=smooth)
    zm = zm[:, 0]

    #    x = samples[:,0].numpy()
    #    w = weights.numpy()
    #
    #    v, e = np.histogram(x, weights=w, bins=bins, density=True)
    #    zm = (e[1:] + e[:-1]) / 2
    #    if smooth is not None:
    #        v = gaussian_filter1d(v, smooth)

    if ax is None:
        ax = plt.gca()

    levels = sorted(get_HDI_thresholds(v))
    if contours:
        contour1d(zm, v, levels, ax=ax, color=color)
    ax.plot(zm, v, color=color)
    ax.set_xlim([zm.min(), zm.max()])
    ax.set_ylim([-v.max() * 0.05, v.max() * 1.1])


#    # Diagnostics
#    mean = sum(w * x) / sum(w)
#    mode = zm[v == v.max()][0]
#    int2 = zm[v > levels[2]].min(), zm[v > levels[2]].max()
#    int1 = zm[v > levels[1]].min(), zm[v > levels[1]].max()
#    int0 = zm[v > levels[0]].min(), zm[v > levels[0]].max()
#    entropy = -simps(v * np.log(v), zm)
#    return dict(
#        mean=mean, mode=mode, HDI1=int2, HDI2=int1, HDI3=int0, entropy=entropy
#    )


def plot_posterior(
    samples,
    pois,
    weights_key=None,
    ax=None,
    grid_interpolate=False,
    bins=100,
    color="k",
    contours=True,
    **kwargs
):
    if ax is None:
        ax = plt.gca()

    if isinstance(pois, int):
        pois = (pois,)

    w = None

    # FIXME: Clean up ad hoc code
    if weights_key is None:
        weights_key = tuple(sorted(pois))
    try:
        w = samples["weights"][tuple(weights_key)]
    except KeyError:
        if len(weights_key) == 1:
            for k in samples["weights"].keys():
                if weights_key[0] in k:
                    weights_key = k
                    break
            w = samples["weights"][tuple(weights_key)]
        elif len(weights_key) == 2:
            for k in samples["weights"].keys():
                if set(weights_key).issubset(k):
                    weights_key = k
                    w = samples["weights"][k]
    if w is None:
        return

    if len(pois) == 1:
        x = samples["v"][:, pois[0]]

        if grid_interpolate:
            # Grid interpolate samples
            log_prior = samples["log_priors"][pois[0]]
            w_eff = np.exp(np.log(w) + log_prior)  # p(z|x) = r(x, z) p(z)
            zm, v = grid_interpolate_samples(x, w_eff)
        else:
            v, e = np.histogram(x, weights=w, bins=bins, density=True)
            zm = (e[1:] + e[:-1]) / 2

        levels = sorted(get_HDI_thresholds(v))
        if contours:
            contour1d(zm, v, levels, ax=ax, color=color)
        ax.plot(zm, v, color=color, **kwargs)
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-v.max() * 0.05, v.max() * 1.1])

        # Diagnostics
        mean = sum(w * x) / sum(w)
        mode = zm[v == v.max()][0]
        int2 = zm[v > levels[2]].min(), zm[v > levels[2]].max()
        int1 = zm[v > levels[1]].min(), zm[v > levels[1]].max()
        int0 = zm[v > levels[0]].min(), zm[v > levels[0]].max()
        entropy = -simps(v * np.log(v), zm)
        return dict(
            mean=mean, mode=mode, HDI1=int2, HDI2=int1, HDI3=int0, entropy=entropy
        )
    elif len(pois) == 2:
        # FIXME: use interpolation when grid_interpolate == True
        x = samples["v"][:, pois[0]]
        y = samples["v"][:, pois[1]]
        counts, xbins, ybins, _ = ax.hist2d(x, y, weights=w, bins=bins, cmap="gray_r")
        levels = sorted(get_HDI_thresholds(counts))
        try:
            ax.contour(
                counts.T,
                extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
                levels=levels,
                linestyles=[":", "--", "-"],
                colors=color,
            )
        except ValueError:
            print("WARNING: 2-dim contours not well-defined.")
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])

        xm = (xbins[:-1] + xbins[1:]) / 2
        ym = (ybins[:-1] + ybins[1:]) / 2

        cx = counts.sum(axis=1)
        cy = counts.sum(axis=0)

        mean = (sum(xm * cx) / sum(cx), sum(ym * cy) / sum(cy))

        return dict(mean=mean, mode=None, HDI1=None, HDI2=None, HDI3=None, entropy=None)


def plot_1d_old(
    samples,
    pois,
    truth=None,
    bins=100,
    figsize=(15, 10),
    color="k",
    labels=None,
    label_args={},
    ncol=None,
    subplots_kwargs={},
    fig=None,
    contours=True,
) -> None:
    """Make beautiful 1-dim posteriors.

    Args:
        samples: Samples from `swyft.Posteriors.sample`
        pois: List of parameters of interest
        truth: Ground truth vector
        bins: Number of bins used for histograms.
        figsize: Size of figure
        color: Color
        labels: Custom labels (default is parameter names)
        label_args: Custom label arguments
        ncol: Number of panel columns
        subplot_kwargs: Subplot kwargs
    """

    grid_interpolate = False
    diags = {}

    if ncol is None:
        ncol = len(pois)
    K = len(pois)
    nrow = (K - 1) // ncol + 1

    if fig is None:
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, **subplots_kwargs)
    else:
        axes = fig.get_axes()
    lb = 0.125
    tr = 0.9
    whspace = 0.15
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    if labels is None:
        labels = [samples["parameter_names"][pois[i]] for i in range(K)]

    for k in range(K):
        if nrow == 1 and ncol > 1:
            ax = axes[k]
        elif nrow == 1 and ncol == 1:
            ax = axes
        else:
            i, j = k % ncol, k // ncol
            ax = axes[j, i]
        ret = plot_posterior(
            samples,
            pois[k],
            ax=ax,
            grid_interpolate=grid_interpolate,
            color=color,
            bins=bins,
            contours=contours,
        )
        ax.set_xlabel(labels[k], **label_args)
        if truth is not None:
            ax.axvline(truth[pois[k]], ls=":", color="r")
        diags[(pois[k],)] = ret
    return fig, diags


def corner(
    logratios,
    parnames,
    bins=100,
    truth=None,
    figsize=(10, 10),
    color="k",
    labels=None,
    label_args={},
    contours_1d: bool = True,
    fig=None,
    labeler=None,
    smooth=0.0,
) -> None:
    """Make a beautiful corner plot.

    Args:
        samples: Samples from `swyft.Posteriors.sample`
        pois: List of parameters of interest
        truth: Ground truth vector
        bins: Number of bins used for histograms.
        figsize: Size of figure
        color: Color
        labels: Custom labels (default is parameter names)
        label_args: Custom label arguments
        contours_1d: Plot 1-dim contours
        fig: Figure instance
    """
    K = len(parnames)
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=figsize)
    else:
        axes = np.array(fig.get_axes()).reshape((K, K))
    lb = 0.125
    tr = 0.9
    whspace = 0.1
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    diagnostics = {}

    if labeler is not None:
        labels = [labeler.get(k, k) for k in parnames]
    else:
        labels = parnames

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
                    ret = plot_2d(
                        logratios,
                        parnames[j],
                        parnames[i],
                        ax=ax,
                        color=color,
                        bins=bins,
                        smooth=smooth,
                    )
                except swyft.SwyftParameterError:
                    pass
            #                if truth is not None:
            #                    ax.axvline(truth[parnames[j]], color="r")
            #                    ax.axhline(truth[parnames[i]], color="r")
            #                diagnostics[(pois[j], pois[i])] = ret
            if j == i:
                try:
                    ret = plot_1d(
                        logratios,
                        parnames[i],
                        ax=ax,
                        color=color,
                        bins=bins,
                        contours=contours_1d,
                        smooth=smooth,
                    )
                except swyft.SwyftParameterError:
                    pass

    #                if truth is not None:
    #                    ax.axvline(truth[pois[i]], ls=":", color="r")
    #                diagnostics[(pois[i],)] = ret
    return fig


def contour1d(z, v, levels, ax=plt, linestyles=None, color=None, **kwargs):
    y0 = -1.0 * v.max()
    y1 = 5.0 * v.max()
    ax.fill_between(z, y0, y1, where=v > levels[0], color=color, alpha=0.1)
    ax.fill_between(z, y0, y1, where=v > levels[1], color=color, alpha=0.1)
    ax.fill_between(z, y0, y1, where=v > levels[2], color=color, alpha=0.1)
    # if not isinstance(colors, list):
    #    colors = [colors]*len(levels)
    # for i, l in enumerate(levels):
    #    zero_crossings = np.where(np.diff(np.sign(v-l*1.001)))[0]
    #    for c in z[zero_crossings]:
    #        ax.axvline(c, ls=linestyles[i], color = colors[i], **kwargs)


if __name__ == "__main__":
    pass


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
    """
    cov = swyft.estimate_coverage(coverage_samples, params, z_max=z_max, bins=bins)
    ax = ax if ax else plt.gca()
    swyft.plot.mass.plot_empirical_z_score(ax, cov[:, 0], cov[:, 1], cov[:, 2:])


def plot_pp(
    coverage_samples,
    params: Union[str, Sequence[str]],
    z_max: float = 3.5,
    bins: int = 50,
    ax=None,
):
    """Make a pp plot."""
    cov = swyft.estimate_coverage(coverage_samples, params, z_max=z_max, bins=bins)
    alphas = 1 - swyft.plot.mass.get_alpha(cov)
    ax = ax if ax else plt.gca()
    ax.fill_between(alphas[:, 0], alphas[:, 2], alphas[:, 3], color="0.8")
    ax.plot(alphas[:, 0], alphas[:, 1], "k")
    plt.plot([0, 1], [0, 1], "g--")
    plt.xlabel("Nominal credibility [$1-p$]")
    plt.ylabel("Empirical coverage [$1-p$]")
    # swyft.plot.mass.plot_empirical_z_score(ax, cov[:,0], cov[:,1], cov[:,2:])


# def plot_scores(mass):
#    s = mass.get_z_scores()
#    for j, k in enumerate(s.keys()):
#        for i in [0, 1, 2]:
#            y = s[k][i,1]
#            yerr = np.array([s[k][i,2]-y, y-s[k][i,0]]).reshape(2, 1)
#            plt.errorbar(j, [y], yerr = yerr, marker='.', color='k')
#    labels = [list(v) for v in s.keys()]
#    plt.xticks(range(len(labels)), labels)
#    plt.axhline(1., color='k', ls=':')
#    plt.axhline(2., color='k', ls=':')
#    plt.axhline(3., color='k', ls=':')
