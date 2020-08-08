# pylint: disable=no-member
from typing import Optional, Union, Callable, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


def simulate(simulator: Callable[[Tensor,], Tensor], z: Tensor) -> Tensor:
    """Simulate given parameters z. Simulator must return shape (1, N)"""
    return torch.cat([simulator(zz) for zz in z], dim=0)  # TODO make a batched version of this.


class Warehouse(object):
    # The idea is to store everything in lists which are identified by rounds and the associated estimators (or model params)
    # Does it take in a simulator? It should be connected with a simulator I believe.
    def __init__(
        self,
        x: Optional[Tensor] = None, 
        z: Optional[Tensor] = None, 
        rounds: Optional[Tensor] = None, 
        likelihood_estimator: Optional[nn.Module] = None
    ,):
        super().__init__()
    
    @property
    def x(self) -> Sequence[Tensor]:
        raise NotImplementedError

    @property
    def z(self) -> Sequence[Tensor]:
        raise NotImplementedError
    
    @property
    def rounds(self) -> Sequence[Tensor]:
        # A sequence which shows which round a particular sample was initially drawn in. Has the same samples per round as x or z
        raise NotImplementedError
    
    @property
    def likelihood_estimators(self) -> Sequence[nn.Module]:
        raise NotImplementedError

    def get_dataset(
        self,
        subset_percents: Iterable[float] = (1.0,),
    ) -> Union[Dataset, Sequence[Dataset]]:
        raise NotImplementedError

    def get_dataloader(
        self,
        percent_train: Iterable[float] = (1.0,),
    ) -> Union[DataLoader, Sequence[Dataset]]:
        raise NotImplementedError

    def append(
        self,
        x: Optional[Tensor] = None, 
        z: Optional[Tensor] = None, 
        rounds: Optional[Tensor] = None, 
        likelihood_estimator: Optional[nn.Module] = None,
    ) -> None:
        # Should allow the addition of new data and estimators to the warehouse. 
        # Perhaps some saftey checks so that the user doesn't add two sets of zs before an x or something.
        raise NotImplementedError


def get_masking_fn(likelihood_estimator: nn.Module, x0: Tensor, threshold: float) -> Callable[[Tensor,], Tensor]:
    # Return a function which classifies parameters as above or below the threshold, i.e. returns a boolean tensor
    raise NotImplementedError


def apply_mask(
    masking_fn: Callable[[Tensor,], Tensor], 
    x: Tensor,
    z: Tensor, 
    rounds: Optional[Tensor] = None
) -> Union[Tuple[Tensor, Tensor,], Tuple[Tensor, Tensor, Tensor]]:
    # Actually mask the parameters (and rounds if given)
    raise NotImplementedError


def sample(
    n_samples = int,
    n_dim = int,
    masking_fn: Optional[Callable[[Tensor,], Tensor]] = None,
    existing_z: Optional[Tensor] = None,
    existing_rounds: Optional[Tensor] = None,
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
