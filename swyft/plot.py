import numpy as np
import pylab as plt
from scipy.interpolate import griddata

from .types import Array, Sequence, Tuple
from .utils import grid_interpolate_samples, verbosity

# def get_contour_levels(x, cred_level=[0.68268, 0.95450, 0.99730]):
#    x = np.sort(x)[::-1]  # Sort backwards
#    total_mass = x.sum()
#    enclosed_mass = np.cumsum(x)
#    idx = [np.argmax(enclosed_mass >= total_mass * f) for f in cred_level]
#    levels = np.array(x[idx])
#    return levels


def cont2d(
    ax, labels, post, cmap: str = "gray_r",
):
    """Create a 2d contour plot.

    Args:
        ax (matplotlib.axes.Axes): matplotlib axes
        re: train ratio estimator
        x0: true observation
        z0: true parameters
        i: combination index
        j: combination index
        cmap: color map
        max_n_points: number of points to train on
    """
    #    z, p = re.posterior(x0, [i, j], max_n_points=max_n_points)
    #    levels = get_contour_levels(p)

    #    if z0 is not None:
    #        ax.axvline(z0[i], color="r", ls=":")
    #        ax.axhline(z0[j], color="r", ls=":")

    #    N = 100 * 1j
    #    extent = [z[:, 0].min(), z[:, 0].max(), z[:, 1].min(), z[:, 1].max()]
    #    xs, ys = np.mgrid[
    #        z[:, 0].min() : z[:, 0].max() : N, z[:, 1].min() : z[:, 1].max() : N
    #    ]
    #    resampled = griddata(z, p, (xs, ys))
    #    ax.imshow(resampled.T, extent=extent, origin="lower", cmap=cmap, aspect="auto")
    #    ax.tricontour(z[:, 0], z[:, 1], -p, levels=-levels, colors="k", linestyles=["-"])
    w = post[labels]["weight"]
    w = post[labels]["weight"]
    plt.scatter()


def hist1d(ax, re, x0, z0, i, max_n_points=1000):
    if z0 is not None:
        ax.axvline(z0[i], color="r", ls=":")
    z, p = re.posterior(x0, i, max_n_points=max_n_points)
    ax.plot(z, p, "k")


def plot1d(
    post,
    params,
    figsize=(15, 5),
    color="k",
    labels=None,
    ncol=None,
    truth=None,
    bins=100,
    grid_interpolate=False,
    label_args={},
) -> None:

    if ncol is None:
        ncol = len(params)
    K = len(params)
    nrow = (K - 1) // ncol + 1

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    lb = 0.125
    tr = 0.9
    whspace = 0.15
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    if labels is None:
        labels = [params[i] for i in range(K)]

    for k in range(K):
        if nrow == 1:
            ax = axes[k]
        else:
            i, j = k % ncol, k // ncol
            ax = axes[j, i]
        plot_posterior(
            post,
            params[k],
            ax=ax,
            grid_interpolate=grid_interpolate,
            color=color,
            bins=bins,
        )
        ax.set_xlabel(labels[k], **label_args)
        if truth is not None:
            ax.axvline(truth[params[k]], ls=":", color="r")


def corner(
    post,
    params,
    figsize=(10, 10),
    color="k",
    labels=None,
    label_args={},
    truth=None,
    bins=100,
) -> None:
    K = len(params)
    fig, axes = plt.subplots(K, K, figsize=figsize)
    lb = 0.125
    tr = 0.9
    whspace = 0.1
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    if labels is None:
        labels = [params[i] for i in range(K)]
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
                plot_posterior(
                    post, [params[j], params[i]], ax=ax, color=color, bins=bins
                )
                if truth is not None:
                    ax.axvline(truth[params[j]], color="r")
                    ax.axhline(truth[params[i]], color="r")
            if j == i:
                plot_posterior(post, params[i], ax=ax, color=color, bins=bins)
                if truth is not None:
                    ax.axvline(truth[params[i]], ls=":", color="r")


def get_contour_levels(x, cred_level=[0.68268, 0.95450, 0.99730]):
    x = x.flatten()
    x = np.sort(x)[::-1]  # Sort backwards
    total_mass = x.sum()
    enclosed_mass = np.cumsum(x)
    idx = [np.argmax(enclosed_mass >= total_mass * f) for f in cred_level]
    levels = np.array(x[idx])
    return levels


def contour1d(z, v, levels, ax=plt, linestyles=None, color=None, **kwargs):
    y0 = -0.05 * v.max()
    y1 = 1.1 * v.max()
    ax.fill_between(z, y0, y1, where=v > levels[0], color=color, alpha=0.1)
    ax.fill_between(z, y0, y1, where=v > levels[1], color=color, alpha=0.1)
    ax.fill_between(z, y0, y1, where=v > levels[2], color=color, alpha=0.1)
    # if not isinstance(colors, list):
    #    colors = [colors]*len(levels)
    # for i, l in enumerate(levels):
    #    zero_crossings = np.where(np.diff(np.sign(v-l*1.001)))[0]
    #    for c in z[zero_crossings]:
    #        ax.axvline(c, ls=linestyles[i], color = colors[i], **kwargs)


def plot_posterior(
    post,
    params,
    weights_key=None,
    ax=plt,
    grid_interpolate=False,
    bins=100,
    color="k",
    **kwargs
):
    if isinstance(params, str):
        params = (params,)

    if weights_key is None:
        weights_key = tuple(sorted(params))
    try:
        w = post["weights"][tuple(weights_key)]
    except KeyError:
        w = None

    if len(params) == 1:
        x = post["params"][params[0]]

        if grid_interpolate:
            zm, v = grid_interpolate_samples(x, w)
        else:
            v, e = np.histogram(x, weights=w, bins=bins, density=True)
            zm = (e[1:] + e[:-1]) / 2

        levels = sorted(get_contour_levels(v))
        contour1d(zm, v, levels, ax=ax, color=color)
        ax.plot(zm, v, color=color, **kwargs)
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-v.max() * 0.05, v.max() * 1.1])
    elif len(params) == 2:
        x = post["params"][params[0]]
        y = post["params"][params[1]]
        counts, xbins, ybins, _ = ax.hist2d(x, y, weights=w, bins=bins, cmap="gray_r")
        levels = sorted(get_contour_levels(counts))
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


if __name__ == "__main__":
    pass
