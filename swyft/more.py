# pylint: disable=no-member
from typing import Optional, Union, Callable, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class Warehouse(object):
    # The idea is to store everything in lists which are identified by rounds and the associated estimators (or model params)
    # Does it take in a simulator? It should be connected with a simulator I believe.
    def __init__(self):
        super().__init__()
    
    @property
    def x(self) -> Sequence[Tensor]:
        raise NotImplementedError

    @property
    def z(self) -> Sequence[Tensor]:
        raise NotImplementedError
    
    @property
    def rounds(self) -> Sequence[int]:
        raise NotImplementedError
    
    @property
    def likelihood_estimators(self) -> Sequence[nn.Module]:
        raise NotImplementedError
    
    @property
    def masks(self) -> Sequence[Tensor]:
        # Setup lazy evaluation of mask for parameter on current round.
        # Saving this implies that we need a Warehouse to be assosciated to a particular x0, maybe that isn't a good thing.
        raise NotImplementedError

    def get_dataset(
        self,
        masking_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        percent_train: float = 1.0,
        loader: bool = True
    ) -> Union[Dataset, Tuple[Dataset, Dataset]]:
        raise NotImplementedError

    def get_dataloader(
        self,
    ) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
        raise NotImplementedError
        


def masking_fn(likelihood_estimator: nn.Module, x0: Tensor, threshold: float) -> Callable[[Tensor,], Tensor]:
    # Return a function which classifies parameters as above or below the threshold
    raise NotImplementedError


def sample(
    n_samples = int,
    n_dim = int,
    masking_fn: Optional[Callable[[Tensor,], Tensor]] = None,
    existing_samples: Optional[Tensor] = None
) -> Tensor:
    # Start by looking at existing samples, when masking function is there use it, after that sample the hypercube with masking function
    raise NotImplementedError

def train(
    network: nn.Module, 
    train_loader: DataLoader,
    validation_loader: DataLoader,
    early_stopping_patience: int,
    max_epochs: Optional[int] = None,
    lr: float = 1e-3,
    combinations: Optional[Sequence[Sequence[int]]] = None,
    device: Union[torch.device, str] = None,
    non_blocking: bool = True,
) -> Tuple[Sequence[float], Sequence[float], dict]:
    # Given loaders and a network it returns the training stats and the best params
    # When looping over legs, consider that multiple dimension posteriors are probably lower weight than single dimension ones.
    raise NotImplementedError


if __name__ == "__main__":
    pass
