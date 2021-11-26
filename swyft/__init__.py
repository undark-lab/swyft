from swyft.bounds import Bound
from swyft.inference import (
    MarginalPosterior,
    MarginalRatioEstimator,
    Posteriors,
    TrainOptions,
)
from swyft.networks import DefaultHead, DefaultTail, Module, OnlineStandardizingLayer
from swyft.plot import plot_empirical_mass
from swyft.prior import (
    Prior,
    PriorTruncator,
    get_diagonal_normal_prior,
    get_uniform_prior,
)
from swyft.store import DaskSimulator, Dataset, DirectoryStore, MemoryStore, Simulator


def zen():
    print("  Cursed by the dimensionality of your nuisance space?")
    print("  Wasted by Markov chains that reject your simulations?")
    print("     Exhausted from messing with simplistic models,")
    print("because your inference algorithm cannot handle the truth?")
    print("         Try swyft for some pain relief.")


__all__ = [
    "TrainOptions",
    "Posteriors",
    "Prior",
    "PriorTruncator",
    "Bound",
    "MarginalRatioEstimator",
    "MarginalPosterior",
    "Module",
    "DirectoryStore",
    "MemoryStore",
    "DefaultHead",
    "DefaultTail",
    "OnlineStandardizingLayer",
    "Dataset",
    "Simulator",
    "DaskSimulator",
    "plot_empirical_mass",
    "get_uniform_prior",
    "get_diagonal_normal_prior",
]
