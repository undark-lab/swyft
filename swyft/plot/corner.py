from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from pandas import DataFrame
from tqdm import tqdm

from swyft.types import Array, LimitType
from swyft.utils.marginals import get_d_dim_marginal_indices


def split_corner_axes(axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    diag = np.diag(axes)
    lower = axes[np.tril(axes, -1).nonzero()]
    upper = axes[np.triu(axes, 1).nonzero()]
    return lower, diag, upper


def _set_weight_keyword(df: DataFrame) -> Optional[str]:
    if "weight" in df.columns:
        return "weight"
    elif "weights" in df.columns:
        return "weights"
    else:
        return None


def corner(
    marginal_df_1d: Dict[Tuple[int], DataFrame],
    marginal_df_2d: Dict[Tuple[int], DataFrame],
    figsize: Optional[Tuple[float, float]] = None,
    bins: int = 50,
    kde: Optional[bool] = False,
    xlim: LimitType = None,
    ylim_lower: LimitType = None,
    truth: Array = None,
    levels: int = 3,
    labels: Sequence[str] = None,
    ticks: bool = True,
    ticklabels: bool = True,
    ticklabelsize: str = "x-small",
    tickswhich: str = "both",
    labelrotation: float = 45.0,
    space_between_axes: float = 0.1,
) -> Tuple[Figure, np.ndarray]:
    """create a corner plot from a dictionary of marginal DataFrames

    Args:
        marginal_dfs: a dictionary map from marginal_index to DataFrame (with a weight column). the DataFrame requires columns tilted by the corresponding marginal_index integer.
        figsize: choose the figsize. like matplotlib.
        bins: number of bins for the histogram
        kde: do a kernel density estimate to produce isocontour lines? (may be expensive)
        xlim: set the xlim. either a single tuple for the same value on all plots, or a sequence of tuples for every column in the plot.
        ylim_lower: set the ylim for the 2d histograms on the lower triangle. either a single tuple for the same value on all plots, or a sequence of tuples for every row in the plot. first row is not considered.
        truth: array denoting the true parameter which generated the observation.
        levels: number of isocontour lines to plot. only functions when `kde=True`.
        labels: the string labels for the parameters.
        ticks: whether to show ticks on the bottom row and leftmost column
        ticklabels: whether to show the value of the ticks bottom row and leftmost column. only functions when `ticks=True`.
        ticklabelsize: set size of tick labels. see `plt.tick_params`
        tickswhich: whether to affect major or minor ticks. see `plt.tick_params`
        labelrotation: tick label rotation. see `plt.tick_params`
        space_between_axes: changes the `wspace` and `hspace` between subplots. see `plt.subplots_adjust`

    Returns:
        matplotlib figure, np array of matplotlib axes
    """
    marginals_1d = marginal_df_1d
    marginals_2d = marginal_df_2d

    d = len(marginals_1d)
    upper_inds = get_d_dim_marginal_indices(d, 2)
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

    for i, (k, ax) in enumerate(zip(marginals_1d.keys(), diag)):
        df = marginals_1d[k]
        sns.histplot(
            df,
            x=k[0],
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

    for k, i in tqdm(zip(marginals_2d.keys(), upper_inds)):
        a, b = i  # plot array index, upper index (lower index is transpose)
        x, y = k  # marginal index
        ax = axes[b, a]  # targets the lower left corner
        df = marginals_2d[k]

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
            ax.axvline(truth[a], color="r")
            ax.axhline(truth[b], color="r")
            ax.scatter(*truth[i, ...], color="r")

        if ylim_lower is None:
            pass
        elif isinstance(ylim_lower[0], (int, float)):
            ax.set_ylim(*ylim_lower)
        elif isinstance(ylim_lower[0], (tuple, list)):
            ax.set_ylim(*ylim_lower[b])
        else:
            raise NotImplementedError(
                "ylim should be a tuple or a list of tuples. Rows are different, columns have the same ylim."
            )

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
    for i, axrow in enumerate(axes):
        for j, ax in enumerate(axrow):
            ax.set_xlabel("")
            ax.set_ylabel("")
            if xlim is None:
                pass
            elif isinstance(xlim[0], (int, float)):
                ax.set_xlim(*xlim)
            elif isinstance(xlim[0], (tuple, list)):
                ax.set_xlim(*xlim[j])
            else:
                raise NotImplementedError("xlim should be a tuple or a list of tuples.")

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


if __name__ == "__main__":
    pass
