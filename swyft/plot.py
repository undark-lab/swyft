import numpy as np
import pylab as plt
from scipy.interpolate import griddata


def get_contour_levels(x, cred_level=[0.68268, 0.95450, 0.99730]):
    x = np.sort(x)[::-1]  # Sort backwards
    total_mass = x.sum()
    enclosed_mass = np.cumsum(x)
    idx = [np.argmax(enclosed_mass >= total_mass * f) for f in cred_level]
    levels = np.array(x[idx])
    return levels


def cont2d(ax, re, x0, z0, i, j, cmap="gray_r", max_n_points=1000):
    z, p = re.posterior(x0, [i, j], max_n_points=max_n_points)
    levels = get_contour_levels(p)

    if z0 is not None:
        ax.axvline(z0[i], color="r", ls=":")
        ax.axhline(z0[j], color="r", ls=":")

    N = 100 * 1j
    extent = [z[:, 0].min(), z[:, 0].max(), z[:, 1].min(), z[:, 1].max()]
    xs, ys = np.mgrid[
        z[:, 0].min() : z[:, 0].max() : N, z[:, 1].min() : z[:, 1].max() : N
    ]
    resampled = griddata(z, p, (xs, ys))
    ax.imshow(resampled.T, extent=extent, origin="lower", cmap=cmap, aspect="auto")
    ax.tricontour(z[:, 0], z[:, 1], -p, levels=-levels, colors="k", linestyles=["-"])


def hist1d(ax, re, x0, z0, i, max_n_points=1000):
    if z0 is not None:
        ax.axvline(z0[i], color="r", ls=":")
    z, p = re.posterior(x0, i, max_n_points=max_n_points)
    ax.plot(z, p, "k")


def plot1d(
    re1d,
    x0,
    dims=(15, 5),
    ncol=None,
    params=None,
    labels=None,
    z0=None,
    cmap="Greys",
    max_n_points=1000,
):
    # TODO: Rewrite
    if params is None:
        params = range(re1d.zdim)

    if ncol is None:
        ncol = re1d.zdim

    K = len(params)
    nrow = (K - 1) // ncol + 1

    fig, axes = plt.subplots(nrow, ncol, figsize=dims)
    lb = 0.125
    tr = 0.9
    whspace = 0.15
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    if labels is None:
        labels = ["z%i" % params[i] for i in range(K)]
    for k in range(K):
        if nrow == 1:
            ax = axes[k]
        else:
            i, j = k % ncol, k // ncol
            ax = axes[j, i]
        hist1d(ax, re1d, x0, z0, params[k], max_n_points=max_n_points)
        ax.set_xlabel(labels[k])


def corner(
    re1d,
    re2d,
    x0,
    dim=10,
    params=None,
    labels=None,
    z0=None,
    cmap="Greys",
    max_n_points=1000,
):
    # TODO: Rewrite
    if params is None:
        params = range(re1d.zdim)

    K = len(params)
    fig, axes = plt.subplots(K, K, figsize=(dim, dim))
    lb = 0.125
    tr = 0.9
    whspace = 0.1
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    if labels is None:
        labels = ["z%i" % params[i] for i in range(K)]
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
                ax.set_xlabel(labels[j])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])

            # Set limits
            # ax.set_xlim(x_lims[j])
            # if i != j:
            #    ax.set_ylim(y_lims[i])

            # 2-dim plots
            if j < i:
                cont2d(
                    ax,
                    re2d,
                    x0,
                    z0,
                    params[j],
                    params[i],
                    cmap=cmap,
                    max_n_points=max_n_points,
                )

            if j == i:
                hist1d(ax, re1d, x0, z0, params[i], max_n_points=max_n_points)


if __name__ == "__main__":
    pass
