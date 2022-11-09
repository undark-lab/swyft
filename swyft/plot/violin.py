from typing import Optional, Sequence

from matplotlib.axes import Axes

from swyft.types import MarginalToArray
from swyft.utils.marginals import filter_marginals_by_dim

##import pandas as pd
#
# def create_violin_df_from_marginal_dict(
#    marginals: MarginalToArray, method: Optional[str] = None
# ) -> pd.DataFrame:
#    """map from a marginal sample dict to the df format for violin plots
#
#    Args:
#        marginals: marginal dictionary
#        method: name of method used to estimate posterior
#
#    Returns:
#        violin dataframe
#    """
#    marginals_1d = filter_marginals_by_dim(marginals, 1)
#    rows = []
#    for key, value in marginals_1d.items():
#        data = {}
#        data["Marginal"] = [key[0]] * len(value)
#        data["Parameter"] = value.flatten()
#        data["Method"] = [method] * len(value)
#        df = pd.DataFrame.from_dict(data)
#        rows.append(df)
#    return pd.concat(rows, ignore_index=True)


def violin(
    marginals: MarginalToArray,
    axes: Axes = None,
    palette: str = "muted",
    labels: Optional[Sequence[str]] = None,
) -> Axes:
    """create a seaborn violin plot

    Args:
        marginals: marginals from the estimator, must be samples (NOT weighted samples)
        axes: matplotlib axes
        palette: seaborn palette
        labels: the string labels for the parameters.

    Returns:
        Axes
    """
    import seaborn as sns

    ax = sns.violinplot(
        x="Marginal",
        y="Parameter",
        data=create_violin_df_from_marginal_dict(marginals),
        palette=palette,
        split=True,
        scale="width",
        inner="quartile",
        ax=axes,
    )
    if labels is not None:
        ax.set_xticklabels(labels)
    return ax
