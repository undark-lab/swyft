from .store import DirectoryStore, MemoryStore, Dataset, Simulator
from .bounds import Prior, Bound
from .inference import Posteriors, Microscope, Task
from .networks import OnlineNormalizationLayer, Module, DefaultHead, DefaultTail
from .utils import corner, plot1d

def zen():
    print("  Cursed by the dimensionality of your nuisance space?")
    print("  Wasted by Markov chains that reject your simulations?")
    print("     Exhausted from messing with simplistic models,")
    print("because your inference algorithm cannot handle the truth?")
    print("         Try swyft for some pain relief.")


__all__ = [
    "Microscope",
    "Task",
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
    "Simulator",
    "corner",
    "plot1d",
]
