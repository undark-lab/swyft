from swyft.bounds import Bound, Prior, TruncatedPrior
from swyft.inference import Posteriors #, Microscope, Task
from swyft.networks import DefaultHead, DefaultTail, Module, OnlineNormalizationLayer
from swyft.store import (
    Dataset,
    DirectoryStore,
    ExactDataset,
    MemoryStore,
    Simulator,
    DaskSimulator,
)
from swyft.utils import corner, plot1d


def zen():
    print("  Cursed by the dimensionality of your nuisance space?")
    print("  Wasted by Markov chains that reject your simulations?")
    print("     Exhausted from messing with simplistic models,")
    print("because your inference algorithm cannot handle the truth?")
    print("         Try swyft for some pain relief.")


__all__ = [
    "TruncatedPrior",
    #"Microscope",
    #"Task",
    "Posteriors",
    "Prior",
    "Bound",
    "Module",
    "DirectoryStore",
    "MemoryStore",
    "DefaultHead",
    "DefaultTail",
    "OnlineNormalizationLayer",
    "Dataset",
    "ExactDataset",
    "Simulator",
    "DaskSimulator",
    "corner",
    "plot1d",
]
