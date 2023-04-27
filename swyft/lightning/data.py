import math
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
from torch.utils.data import random_split
import pytorch_lightning as pl
import zarr
import fasteners
import swyft
from swyft.lightning.simulator import Samples, Sample


######################
# Datasets and loaders
######################


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
        on_after_load_sample: Optional[callable] = None,
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
        self.on_after_load_sample = on_after_load_sample

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
            dataset = self.data.get_dataset(
                on_after_load_sample=self.on_after_load_sample
            )
            splits = torch.utils.data.random_split(dataset, self.lengths)
            self.dataset_train, self.dataset_val, self.dataset_test = splits
        elif isinstance(self.data, swyft.ZarrStore):
            idxr1 = (0, self.lengths[0])
            idxr2 = (self.lengths[0], self.lengths[0] + self.lengths[1])
            idxr3 = (self.lengths[0] + self.lengths[1], len(self.data))
            self.dataset_train = self.data.get_dataset(
                idx_range=idxr1, on_after_load_sample=self.on_after_load_sample
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


###########
# ZarrStore
###########


class ZarrStore:
    r"""Storing training data in zarr archive."""

    def __init__(self, file_path, sync_path=None):
        if sync_path is None:
            sync_path = file_path + ".sync"
        synchronizer = zarr.ProcessSynchronizer(sync_path) if sync_path else None
        self.store = zarr.DirectoryStore(file_path)
        self.root = zarr.group(store=self.store, synchronizer=synchronizer)
        self.lock = fasteners.InterProcessLock(file_path + ".lock.file")

    def reset_length(self, N, clubber=False):
        """Resize store.  N >= current store length."""
        if N < len(self) and not clubber:
            raise ValueError(
                """New length shorter than current store length.
                You can use clubber = True if you know what your are doing."""
            )
        for k in self.data.keys():
            shape = self.data[k].shape
            self.data[k].resize(N, *shape[1:])
        self.root["meta/sim_status"].resize(
            N,
        )

    def init(self, N, chunk_size, shapes=None, dtypes=None):
        if len(self) > 0:
            print("WARNING: Already initialized.")
            return self
        self._init_shapes(shapes, dtypes, N, chunk_size)
        return self

    def __len__(self):
        if "data" not in self.root.keys():
            return 0
        keys = self.root["data"].keys()
        ns = [len(self.root["data"][k]) for k in keys]
        N = ns[0]
        assert all([n == N for n in ns])
        return N

    def keys(self):
        return list(self.data.keys())

    def __getitem__(self, i):
        if isinstance(i, int):
            return Sample({k: self.data[k][i] for k in self.keys()})
        elif isinstance(i, slice):
            return Samples({k: self.data[k][i] for k in self.keys()})
        elif isinstance(i, str):
            return self.data[i]
        else:
            raise ValueError

    # TODO: Remove consistency checks
    def _init_shapes(self, shapes, dtypes, N, chunk_size):
        """Initializes shapes, or checks consistency."""
        for k in shapes.keys():
            s = shapes[k]
            dtype = dtypes[k]
            try:
                self.root.zeros(
                    "data/" + k, shape=(N, *s), chunks=(chunk_size, *s), dtype=dtype
                )
            except zarr.errors.ContainsArrayError:
                assert self.root["data/" + k].shape == (
                    N,
                    *s,
                ), "Inconsistent array sizes"
                assert self.root["data/" + k].chunks == (
                    chunk_size,
                    *s,
                ), "Inconsistent chunk sizes"
                assert self.root["data/" + k].dtype == dtype, "Inconsistent dtype"
        try:
            self.root.zeros(
                "meta/sim_status", shape=(N,), chunks=(chunk_size,), dtype="i4"
            )
        except zarr.errors.ContainsArrayError:
            assert self.root["meta/sim_status"].shape == (
                N,
            ), "Inconsistent array sizes"
        try:
            assert self.chunk_size == chunk_size, "Inconsistent chunk size"
        except KeyError:
            self.data.attrs["chunk_size"] = chunk_size

    @property
    def chunk_size(self):
        return self.data.attrs["chunk_size"]

    @property
    def data(self):
        return self.root["data"]

    def numpy(self):
        return {k: v[:] for k, v in self.root["data"].items()}

    def get_sample_store(self):
        return Samples(self.numpy())

    @property
    def meta(self):
        return {k: v for k, v in self.root["meta"].items()}

    @property
    def sims_required(self):
        return sum(self.root["meta"]["sim_status"][:] == 0)

    def simulate(self, sampler, max_sims=None, batch_size=10):
        total_sims = 0
        if isinstance(sampler, swyft.Simulator):
            sampler = sampler.sample
        while self.sims_required > 0:
            if max_sims is not None and total_sims >= max_sims:
                break
            num_sims = self._simulate_batch(sampler, batch_size)
            total_sims += num_sims

    def _simulate_batch(self, sample_fn, batch_size):
        # Run simulator
        num_sims = min(batch_size, self.sims_required)
        if num_sims == 0:
            return num_sims

        samples = sample_fn(num_sims)

        # Reserve slots
        with self.lock:
            sim_status = self.root["meta"]["sim_status"]
            data = self.root["data"]

            idx = np.arange(len(sim_status))[sim_status[:] == 0][:num_sims]
            index_slices = _get_index_slices(idx)

            for i_slice, j_slice in index_slices:
                sim_status[j_slice[0] : j_slice[1]] = 1
                for k, v in data.items():
                    data[k][j_slice[0] : j_slice[1]] = samples[k][
                        i_slice[0] : i_slice[1]
                    ]

        return num_sims

    def get_dataset(self, idx_range=None, on_after_load_sample=None):
        return ZarrStoreIterableDataset(
            self, idx_range=idx_range, on_after_load_sample=on_after_load_sample
        )

    def get_dataloader(
        self,
        num_workers=0,
        batch_size=1,
        pin_memory=False,
        drop_last=True,
        idx_range=None,
        on_after_load_sample=None,
    ):
        ds = self.get_dataset(
            idx_range=idx_range, on_after_load_sample=on_after_load_sample
        )
        dl = torch.utils.data.DataLoader(
            ds,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
        return dl


def _get_index_slices(idx):
    """Returns list of enumerated consecutive indices"""
    idx = np.array(idx)
    pointer = 0
    residual_idx = idx
    slices = []
    while len(residual_idx) > 0:
        mask = residual_idx - residual_idx[0] - np.arange(len(residual_idx)) == 0
        slc1 = [residual_idx[mask][0], residual_idx[mask][-1] + 1]
        slc2 = [pointer, pointer + sum(mask)]
        pointer += sum(mask)
        slices.append([slc2, slc1])
        residual_idx = residual_idx[~mask]
    return slices


class ZarrStoreIterableDataset(torch.utils.data.dataloader.IterableDataset):
    def __init__(
        self, zarr_store: ZarrStore, idx_range=None, on_after_load_sample=None
    ):
        self.zs = zarr_store
        if idx_range is None:
            self.n_samples = len(self.zs)
            self.offset = 0
        else:
            self.offset = idx_range[0]
            self.n_samples = idx_range[1] - idx_range[0]
        self.chunk_size = self.zs.chunk_size
        self.n_chunks = int(math.ceil(self.n_samples / float(self.chunk_size)))
        self.on_after_load_sample = on_after_load_sample

    @staticmethod
    def get_idx(n_chunks, worker_info):
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            n_chunks_per_worker = int(math.ceil(n_chunks / float(num_workers)))
            idx = [
                worker_id * n_chunks_per_worker,
                min((worker_id + 1) * n_chunks_per_worker, n_chunks),
            ]
            idx = np.random.permutation(range(*idx))
        else:
            idx = np.random.permutation(n_chunks)
        return idx

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        idx = self.get_idx(self.n_chunks, worker_info)
        offset = self.offset
        for i0 in idx:
            # Read in chunks
            data_chunk = {}
            for k in self.zs.data.keys():
                data_chunk[k] = self.zs.data[k][
                    offset + i0 * self.chunk_size : offset + (i0 + 1) * self.chunk_size
                ]
            n = len(data_chunk[k])

            # Return separate samples
            for i in np.random.permutation(n):
                out = {k: v[i] for k, v in data_chunk.items()}
                if self.on_after_load_sample:
                    out = self.on_after_load_sample(out)
                yield out


# def get_ntrain_nvalid(
#    validation_amount: Union[float, int], len_dataset: int
# ) -> Tuple[int, int]:
#    """Divide a dataset into a training and validation set.
#
#    Args:
#        validation_amount: percentage or number of elements in the validation set
#        len_dataset: total length of the dataset
#
#    Raises:
#        TypeError: When the validation_amount is neither a float or int.
#
#    Returns:
#        (n_train, n_valid)
#    """
#    assert validation_amount > 0
#    if isinstance(validation_amount, float):
#        percent_validation = validation_amount
#        percent_train = 1.0 - percent_validation
#        n_valid, n_train = split_length_by_percentage(
#            len_dataset, (percent_validation, percent_train)
#        )
#        if n_valid % 2 != 0:
#            n_valid += 1
#            n_train -= 1
#    elif isinstance(validation_amount, int):
#        n_valid = validation_amount
#        n_train = len_dataset - n_valid
#        assert n_train > 0
#
#        if n_valid % 2 != 0:
#            n_valid += 1
#            n_train -= 1
#    else:
#        raise TypeError("validation_amount must be int or float")
#    return n_train, n_valid


## TODO: Deprecate
# class SwyftDataModule_deprecated(pl.LightningDataModule):
#    def __init__(
#        self,
#        on_after_load_sample=None,
#        store=None,
#        batch_size: int = 32,
#        validation_percentage=0.2,
#        manual_seed=None,
#        train_multiply=10,
#        num_workers=0,
#    ):
#        super().__init__()
#        self.store = store
#        self.on_after_load_sample = on_after_load_sample
#        self.batch_size = batch_size
#        self.num_workers = num_workers
#        self.validation_percentage = validation_percentage
#        self.train_multiply = train_multiply
#        print(
#            "Deprecation warning: Use dataloaders directly rathe than this data module for transparency."
#        )
#
#    def setup(self, stage):
#        self.dataset = SamplesDataset(
#            self.store, on_after_load_sample=self.on_after_load_sample
#        )  # , x_keys = ['data'], z_keys=['z'])
#        n_train, n_valid = get_ntrain_nvalid(
#            self.validation_percentage, len(self.dataset)
#        )
#        self.dataset_train, self.dataset_valid = random_split(
#            self.dataset,
#            [n_train, n_valid],
#            generator=torch.Generator().manual_seed(42),
#        )
#        self.dataset_test = SamplesDataset(
#            self.store
#        )  # , x_keys = ['data'], z_keys=['z'])
#
#    def train_dataloader(self):
#        return torch.utils.data.DataLoader(
#            self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers
#        )
#
#    def val_dataloader(self):
#        return torch.utils.data.DataLoader(
#            self.dataset_valid, batch_size=self.batch_size, num_workers=self.num_workers
#        )
#
#    # # TODO: Deprecate
#    # def predict_dataloader(self):
#    #     return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, num_workers = self.num_workers)
#
#    def test_dataloader(self):
#        return torch.utils.data.DataLoader(
#            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
#        )
#
#    def samples(self, N, random=False):
#        dataloader = torch.utils.data.DataLoader(
#            self.dataset_train, batch_size=N, num_workers=0, shuffle=random
#        )
#        examples = next(iter(dataloader))
#        return Samples(examples)
