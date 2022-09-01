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

class Sample(dict):
    def __repr__(self):
        return "Sample("+super().__repr__()+")"


class Samples(dict):
    """Handles storing samples in memory.  Samples are stored as dictionary of arrays/tensors with num of samples as first dimension."""
    def __len__(self):
        n = [len(v) for v in self.values()] 
        assert all([x == n[0] for x in n]), "Inconsistent lengths in Samples"
        return n[0]

    def __repr__(self):
        return "Samples("+super().__repr__()+")"
    
    def __getitem__(self, i):
        """For integers, return 'rows', for string returns 'columns'."""
        if isinstance(i, int):
            return {k: v[i] for k, v in self.items()}
        elif isinstance(i, slice):
            return Samples({k: v[i] for k, v in self.items()})
        else:
            return super().__getitem__(i)
        
    def get_dataset(self, on_after_load_sample = None):
        """Generator function for SamplesDataset object.

        Args:
            on_after_load_sample: Callable, that is applied to individual samples on the fly.

        Returns:
            SamplesDataset
        """
        return SamplesDataset(self, on_after_load_sample = on_after_load_sample)
    
    def get_dataloader(self, batch_size = 1, shuffle = False, on_after_load_sample = None, repeat = None):
        """Generator function to directly generate a dataloader object.

        Args:
            batch_size: batch_size for dataloader
            shuffle: shuffle for dataloader
            on_after_load_sample: see `get_dataset`
            repeat: If not None, Wrap dataset in RepeatDatasetWrapper
        """
        dataset = self.get_dataset(on_after_load_sample = on_after_load_sample)
        if repeat is not None:
            dataset = RepeatDatasetWrapper(dataset, repeat = repeat)
        return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    
    def to_numpy(self, single_precision = True):
        return to_numpy(self, single_precision = single_precision)
        

class SamplesDataset(torch.utils.data.Dataset):
    """Simple torch dataset based on Samples."""
    def __init__(self, sample_store, on_after_load_sample = None):
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
        return len(self._dataset)*self._repeat

    def __getitem__(self, i):
        return self._dataset[i//self._repeat]
