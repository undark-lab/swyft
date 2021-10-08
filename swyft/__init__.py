from swyft.bounds import Bound
from swyft.inference import Posteriors, TrainOptions
from swyft.networks import DefaultHead, DefaultTail, Module, OnlineStandardizingLayer
from swyft.prior import (
    Prior,
    PriorTruncator,
    get_diagonal_normal_prior,
    get_uniform_prior,
)
from swyft.store import DaskSimulator, Dataset, DirectoryStore, MemoryStore, Simulator
from swyft.utils import plot_1d, plot_corner, plot_empirical_mass


def zen():
    print("  Cursed by the dimensionality of your nuisance space?")
    print("  Wasted by Markov chains that reject your simulations?")
    print("     Exhausted from messing with simplistic models,")
    print("because your inference algorithm cannot handle the truth?")
    print("         Try swyft for some pain relief.")


__all__ = [
    "TrainOptions",
    "PriorTruncator",
    "Posteriors",
    "Prior",
    "Bound",
    "Module",
    "DirectoryStore",
    "MemoryStore",
    "DefaultHead",
    "DefaultTail",
    "OnlineStandardizingLayer",
    "Dataset",
    "Simulator",
    "DaskSimulator",
    "plot_corner",
    "plot_1d",
    "plot_empirical_mass",
    "get_uniform_prior",
    "get_diagonal_normal_prior",
]
