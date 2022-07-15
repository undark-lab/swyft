# pylint: disable=no-member
from itertools import product

import numpy as np
import pytest
import torch

from swyft.utils.array import (
    array_to_tensor,
    np_bool_types,
    np_complex_types,
    np_float_types,
    np_int_types,
    torch_complex_types,
    torch_float_types,
    torch_int_types,
)


class TestArrayToTensor:
    shapes = [(0,), (0, 0), (1, 0), (0, 1), (10, 10), (15, 3, 2, 1, 6)]

    @pytest.mark.parametrize(
        "shape, dtype",
        product(shapes, np_bool_types + np_int_types + np_float_types + [None]),
    )
    def test_shape(self, shape, dtype):
        array = np.empty(shape, dtype)
        tensor = array_to_tensor(array)
        assert tensor.shape == array.shape

    @pytest.mark.parametrize(
        "shape, dtypes",
        product(
            shapes,
            zip(
                np_int_types + np_int_types,
                torch_int_types + [None] * len(np_int_types),
            ),
        ),
    )
    def test_np_to_tensor_int(self, shape, dtypes):
        np_dtype, torch_dtype = dtypes
        array = np.empty(shape, np_dtype)
        tensor = array_to_tensor(array, dtype=torch_dtype)
        if torch_dtype is None:
            assert tensor.dtype == torch.int64
        else:
            assert tensor.dtype == torch_dtype

    @pytest.mark.parametrize("shape, dtype", product(shapes, np_bool_types))
    def test_np_to_tensor_bool(self, shape, dtype):
        array = np.empty(shape, dtype)
        tensor = array_to_tensor(array)
        assert tensor.dtype == torch.bool

    @pytest.mark.parametrize(
        "shape, np_dtype, torch_dtype",
        product(shapes, np_float_types, torch_float_types + [None]),
    )
    def test_np_to_tensor_float(self, shape, np_dtype, torch_dtype):
        array = np.empty(shape, np_dtype)
        tensor = array_to_tensor(array, dtype=torch_dtype)
        if torch_dtype is None:
            assert tensor.dtype == torch.float32
        else:
            assert tensor.dtype == torch_dtype


if __name__ == "__main__":
    pass
