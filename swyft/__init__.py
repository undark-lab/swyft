from swyft.bounds import Bound
from swyft.inference import MarginalPosterior, MarginalRatioEstimator
from swyft.networks import get_marginal_classifier
from swyft.plot import corner, empirical_z_score_corner, hist1d, violin
from swyft.prior import (
    Prior,
    PriorTruncator,
    get_diagonal_normal_prior,
    get_uniform_prior,
)
from swyft.store import DaskSimulator, Dataset, Simulator, Store


def zen():
    print("  Cursed by the dimensionality of your nuisance space?")
    print("  Wasted by Markov chains that reject your simulations?")
    print("     Exhausted from messing with simplistic models,")
    print("because your inference algorithm cannot handle the truth?")
    print("         Try swyft for some pain relief.")


__all__ = [
    "Bound",
    "corner",
    "DaskSimulator",
    "Dataset",
    "empirical_z_score_corner",
    "get_marginal_classifier",
    "hist1d",
    "MarginalPosterior",
    "MarginalRatioEstimator",
    "OnlineDictStandardizingLayer",
    "OnlineStandardizingLayer",
    "Prior",
    "PriorTruncator",
    "Simulator",
    "Store",
    "get_diagonal_normal_prior",
    "get_uniform_prior",
    "violin",
]
