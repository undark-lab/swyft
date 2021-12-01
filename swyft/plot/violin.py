import pandas as pd
import seaborn as sns

from swyft.utils.marginals import filter_marginals_by_dim


def create_violin_df_from_marginal_dict(marginals, method: str):
    marginals_1d = filter_marginals_by_dim(marginals, 1)
    rows = []
    for key, value in marginals_1d.items():
        data = {}
        data["Marginal"] = [key[0]] * len(value)
        data["Parameter"] = value.flatten()
        data["Method"] = [method] * len(value)
        df = pd.DataFrame.from_dict(data)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def violin_plot(
    reference_marginals, estimated_marginals, method: str, ax=None, palette="muted"
):
    data = [
        create_violin_df_from_marginal_dict(reference_marginals, "Reference"),
        # create_violin_df_from_marginal_dict(estimated_marginals, method),
    ]
    data = pd.concat(data, ignore_index=True)
    sns.set_theme(style="whitegrid")
    ax = sns.violinplot(
        x="Marginal",
        y="Parameter",
        # hue="Method",
        data=data,
        palette=palette,
        split=True,
        scale="width",
        inner="quartile",
        ax=ax,
    )
    return ax
