from typing import (
    Callable,
    Dict,
    Hashable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import numpy as np
import torch
import pytorch_lightning as pl
import swyft


class Sample(dict):
    """In Swyft, a 'sample' is a dictionary
    with string-type keys and tensor/array-type values."""

    def __repr__(self):
        return "Sample(" + super().__repr__() + ")"


class SwyftDataModule(pl.LightningDataModule):
    """DataModule to handle simulated data.

    Args:
        data: Simulation data
        lenghts: List of number of samples used for [training, validation, testing].
        fractions: Fraction of samples used for [training, validation, testing].
        batch_size: Minibatch size.
        num_workers: Number of workers for dataloader.
        shuffle: Shuffle training data.

    Returns:
        pytorch_lightning.LightningDataModule
    """

    def __init__(
        self,
        data,
        lengths: Union[Sequence[int], None] = None,
        fractions: Union[Sequence[float], None] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = False,
    ):
        super().__init__()
        self.data = data
        if lengths is not None and fractions is None:
            self.lengths = lengths
        elif lengths is None and fractions is not None:
            self.lengths = self._get_lengths(fractions, len(data))
        else:
            raise ValueError("Either lenghts or fraction must be set, but not both.")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    @staticmethod
    def _get_lengths(fractions, N):
        fractions = np.array(fractions)
        fractions /= sum(fractions)
        mu = N * fractions
        n = np.floor(mu)
        n[0] += N - sum(n)
        return [int(v) for v in n]

    def setup(self, stage: str):
        if isinstance(self.data, Samples):
            dataset = self.data.get_dataset()
            splits = torch.utils.data.random_split(dataset, self.lengths)
            self.dataset_train, self.dataset_val, self.dataset_test = splits
        elif isinstance(self.data, swyft.ZarrStore):
            idxr1 = (0, self.lengths[1])
            idxr2 = (self.lengths[1], self.lengths[1] + self.lengths[2])
            idxr3 = (self.lengths[1] + self.lengths[2], len(self.data))
            self.dataset_train = self.data.get_dataset(
                idx_range=idxr1, on_after_load_sample=None
            )
            self.dataset_val = self.data.get_dataset(
                idx_range=idxr2, on_after_load_sample=None
            )
            self.dataset_test = self.data.get_dataset(
                idx_range=idxr3, on_after_load_sample=None
            )
        else:
            raise ValueError

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return dataloader


class Samples(dict):
    """Handles memory-based samples in Swyft.  Samples are stored as dictionary
    of arrays/tensors with number of samples as first dimension. This class
    provides a few convenience methods for accessing the samples."""

    def __len__(self):
        """Number of samples."""
        n = [len(v) for v in self.values()]
        assert all([x == n[0] for x in n]), "Inconsistent lengths in Samples"
        return n[0]

    def __repr__(self):
        return "Samples(" + super().__repr__() + ")"

    def __getitem__(self, i):
        """For integers, return 'rows', for string returns 'columns'."""
        if isinstance(i, int):
            return {k: v[i] for k, v in self.items()}
        elif isinstance(i, slice):
            return Samples({k: v[i] for k, v in self.items()})
        else:
            return super().__getitem__(i)

    def get_dataset(self, on_after_load_sample=None):
        """Generator function for SamplesDataset object.

        Args:
            on_after_load_sample: Callable, that is applied to individual samples on the fly.

        Returns:
            SamplesDataset
        """
        return SamplesDataset(self, on_after_load_sample=on_after_load_sample)

    def get_dataloader(
        self,
        batch_size=1,
        shuffle=False,
        on_after_load_sample=None,
        repeat=None,
        num_workers=0,
    ):
        """(Deprecated) Generator function to directly generate a dataloader object.

        Args:
            batch_size: batch_size for dataloader
            shuffle: shuffle for dataloader
            on_after_load_sample: see `get_dataset`
            repeat: If not None, Wrap dataset in RepeatDatasetWrapper
        """
        print("WARNING: Deprecated")
        dataset = self.get_dataset(on_after_load_sample=on_after_load_sample)
        if repeat is not None:
            dataset = RepeatDatasetWrapper(dataset, repeat=repeat)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )


#    def to_numpy(self, single_precision = True):
#        return to_numpy(self, single_precision = single_precision)


class SamplesDataset(torch.utils.data.Dataset):
    """Simple torch dataset based on Samples."""

    def __init__(self, sample_store, on_after_load_sample=None):
        self._dataset = sample_store
        self._on_after_load_sample = on_after_load_sample

    def __len__(self):
        return len(self._dataset[list(self._dataset.keys())[0]])

    def __getitem__(self, i):
        d = {k: v[i] for k, v in self._dataset.items()}
        if self._on_after_load_sample is not None:
            d = self._on_after_load_sample(d)
        return d


class RepeatDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, repeat):
        self._dataset = dataset
        self._repeat = repeat

    def __len__(self):
        return len(self._dataset) * self._repeat

    def __getitem__(self, i):
        return self._dataset[i // self._repeat]
