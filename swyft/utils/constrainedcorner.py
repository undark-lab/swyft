from itertools import combinations

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from swyft.utils import filter_marginals_by_dim, split_corner_axes


def get_upper_inds(d):
    return list(combinations(range(d), 2))


def _set_weight_keyword(df):
    if "weights" in df.columns:
        return "weights"
    else:
        return None


def corner(
    marginal_dfs,
    figsize=None,
    bins=50,
    kde=False,
    xlim=None,
    ylim_lower=None,
    truth=None,
    levels=3,
    labels=None,
    ticks=True,
    ticklabels=True,
    ticklabelsize="x-small",
    tickswhich="both",
    labelrotation=45,
    space_between_axes=0.1,
):
    marginals_1d = filter_marginals_by_dim(marginal_dfs, 1)
    marginals_2d = filter_marginals_by_dim(marginal_dfs, 2)
    d = len(marginals_1d)
    upper_inds = get_upper_inds(d)
    assert len(marginals_2d) == len(upper_inds)

    fig, axes = plt.subplots(nrows=d, ncols=d, sharex="col", figsize=figsize)
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

    color = "k"

    for ax in upper:
        ax.axis("off")

    for i, ax in enumerate(diag):
        df = marginals_1d[(i,)]
        sns.histplot(
            marginals_1d[(i,)],
            x=i,
            weights=_set_weight_keyword(df),
            bins=bins,
            ax=ax,
            element="step",
            fill=False,
            color=color,
        )
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            direction="out",
            labelbottom=False,
            labelleft=False,
        )
        if truth is not None:
            ax.axvline(truth[i], color="r")

    for i in tqdm(upper_inds):
        x, y = i
        ax = axes[y, x]  # targets the lower left corner
        df = marginals_2d[i]

        sns.histplot(
            data=df,
            x=x,
            y=y,
            weights=_set_weight_keyword(df),
            bins=bins,
            ax=ax,
            color=color,
            pthresh=0.01,
        )
        if kde:
            sns.kdeplot(
                data=df,
                x=x,
                y=y,
                weights=_set_weight_keyword(df),
                ax=ax,
                # palette="muted",
                color=color,
                levels=levels,
            )

        if truth is not None:
            ax.axvline(truth[i[0]], color="r")
            ax.axhline(truth[i[1]], color="r")
            ax.scatter(*truth[i, ...], color="r")

        if ylim_lower is not None:
            ax.set_ylim(*ylim_lower)
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            # direction="out",
            labelbottom=False,
            labelleft=False,
        )

    # clear all
    for ax in axes.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
        if xlim is not None:
            ax.set_xlim(*xlim)

    # bottom row
    for i, ax in enumerate(axes[-1, :]):
        ax.tick_params(
            axis="x",
            which=tickswhich,
            bottom=ticks,
            direction="out",
            labelbottom=ticks and ticklabels,
            labelrotation=labelrotation,
            labelsize=ticklabelsize,
        )
        if labels is not None:
            ax.set_xlabel(labels[i])

    # left column
    for i, ax in enumerate(axes[1:, 0], 1):
        ax.tick_params(
            axis="y",
            which=tickswhich,
            left=ticks,
            direction="out",
            labelleft=ticks and ticklabels,
            labelrotation=labelrotation,
            labelsize=ticklabelsize,
        )
        if labels is not None:
            ax.set_ylabel(labels[i])

    fig.align_labels()
    return fig, axes


def diagonal_constraint(axes, bounds, alpha=0.25):
    _, diag, _ = split_corner_axes(axes)
    for i, ax in enumerate(diag):
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], bounds[i, 0], alpha=alpha)
        ax.axvspan(bounds[i, 1], xlim[1], alpha=alpha)
    return axes


def mask_outside_polygon(poly_verts, ax, facecolor=None, edgecolor=None, alpha=0.25):
    """
    Plots a mask on the specified axis ("ax", defaults to plt.gca()) such that
    all areas outside of the polygon specified by "poly_verts" are masked.

    "poly_verts" must be a list of tuples of the verticies in the polygon in
    counter-clockwise order.

    Returns the matplotlib.patches.PathPatch instance plotted on the figure.
    """

    # Get current plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Verticies of the plot boundaries in clockwise order
    bound_verts = [
        (xlim[0], ylim[0]),
        (xlim[0], ylim[1]),
        (xlim[1], ylim[1]),
        (xlim[1], ylim[0]),
        (xlim[0], ylim[0]),
    ]

    # A series of codes (1 and 2) to tell matplotlib whether to draw a line or
    # move the "pen" (So that there's no connecting line)
    bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
    poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]

    # Plot the masking patch
    path = mpath.Path(bound_verts + poly_verts, bound_codes + poly_codes)
    patch = mpatches.PathPatch(
        path, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha
    )
    patch = ax.add_patch(patch)

    # Reset the plot limits to their original extents
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return patch


def construct_vertices(bounds, x, y):
    xbound = bounds[x, :]
    ybound = bounds[y, :]
    return [
        (xbound[0], ybound[0]),
        (xbound[0], ybound[1]),
        (xbound[1], ybound[1]),
        (xbound[1], ybound[0]),
        (xbound[0], ybound[0]),
    ][
        ::-1
    ]  # counter clockwise


def lower_constraint(axes, bounds, alpha=0.25):
    d, _ = axes.shape
    upper_inds = get_upper_inds(d)
    for i in upper_inds:
        x, y = i
        ax = axes[y, x]  # targets the lower left corner
        inside_verts = construct_vertices(bounds, x, y)
        mask_outside_polygon(inside_verts, ax)


if __name__ == "__main__":
    import numpy as np

    from swyft.utils import get_df_dict_from_weighted_marginals

    n = 1000
    weighted_marginals = {
        "params": 0.1 * np.random.rand(n, 3) + 0.45,
        "weights": {
            k: np.random.rand(n) for k in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
        },
    }
    dfs = get_df_dict_from_weighted_marginals(weighted_marginals)

    bounds = np.asarray([[0.1, 0.9], [0.2, 0.8], [0.3, 0.9]])
    boundss = [bounds, [1.2, 0.8] * bounds]

    fig, axes = corner(dfs)
    for bounds in boundss:
        diagonal_constraint(axes, bounds)
    for bounds in boundss:
        lower_constraint(axes, bounds)
