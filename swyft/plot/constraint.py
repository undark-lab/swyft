import matplotlib.patches as mpatches
import matplotlib.path as mpath

from swyft.plot.histogram import split_corner_axes
from swyft.utils.marginals import get_d_dim_marginal_indices


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
    upper_inds = get_d_dim_marginal_indices(d, 2)
    for i in upper_inds:
        x, y = i
        ax = axes[y, x]  # targets the lower left corner
        inside_verts = construct_vertices(bounds, x, y)
        mask_outside_polygon(inside_verts, ax, alpha=alpha)
