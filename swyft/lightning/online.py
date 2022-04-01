from typing import Iterator, Optional

import pytorch_lightning as pl
from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data import DataLoader

from . import SwyftModelForward


def truncate_iterator(it: Iterator, n) -> Iterator:
    """
    Truncates an iterator.

    Args:
        n: maximum length of truncated Iterator

    Return:
        ``it``, truncated to return at most ``n`` values.
    """
    for _ in range(n):
        try:
            yield next(it)
        except StopIteration:
            break


class SwyftOnlineDataModule(pl.LightningDataModule):
    """
    A data module that directly wraps the simulator to generate data online. Works
    with batched simulators.

    Notes:
        This has many subtleties when the simulator uses CUDA and ``num_workers``
        is greater than 0.
    """

    def __init__(
        self,
        simulator: SwyftModelForward,
        n_train: int,
        n_val: int,
        n_test: int,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()

        # Tell the dataloaders whether the simulator is batched. After this block,
        # self.batch_size contains the batch size used by the dataloaders and
        # self._dl_batch_size contains the argument passed to the dataloaders.
        if simulator.batch_size is not None:
            if batch_size is not None and batch_size != simulator.batch_size:
                raise ValueError(
                    "conflicting values for simulator and explicit batch_size"
                )
            self.batch_size = simulator.batch_size
            self._dl_batch_size = None
        elif batch_size is not None and batch_size >= 2:
            self.batch_size = self._dl_batch_size = batch_size
        else:
            raise ValueError("batch_size must be >= 2")

        if n_train % self.batch_size != 0:
            raise ValueError("n_train must be a multiple of batch_size")
        if n_val % self.batch_size != 0:
            raise ValueError("n_val must be a multiple of batch_size")
        if n_test % self.batch_size != 0:
            raise ValueError("n_test must be a multiple of batch_size")

        self.save_hyperparameters(ignore=("simulator", "batch_size"))
        self.simulator = simulator

    def setup(self, _: Optional[str] = None):
        it_train = truncate_iterator(self.simulator, self.hparams.n_train)  # type: ignore
        self.dp_train = IterableWrapper(it_train)

        it_val = truncate_iterator(self.simulator, self.hparams.n_val)  # type: ignore
        self.dp_val = IterableWrapper(it_val)

        it_test = truncate_iterator(self.simulator, self.hparams.n_test)  # type: ignore
        self.dp_test = IterableWrapper(it_test)

    def train_dataloader(self):
        return DataLoader(self.dp_train, self._dl_batch_size, num_workers=self.hparams.num_workers)  # type: ignore

    def val_dataloader(self):
        return DataLoader(self.dp_val, self._dl_batch_size, num_workers=self.hparams.num_workers)  # type: ignore

    def test_dataloader(self):
        return DataLoader(self.dp_test, self._dl_batch_size, num_workers=self.hparams.num_workers)  # type: ignore

    def samples(self, n):  # TODO: deprecate
        return self.simulator.sample(n)
