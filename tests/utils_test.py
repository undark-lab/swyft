# pylint: disable=no-member
from itertools import product

import pytest

import torch
import numpy as np
from swyft.utils import array_to_tensor


class TestArrayToTensor:
    shapes = [(0,), (0, 0,), (1, 0,), (0, 1,), (10,10), (15,3,2,1,6)]
    np_bool_type = [np.bool]
    np_int_types = [np.int8, np.int16, np.int32, np.int64]
    np_float_types = [np.float32, np.float64]
    torch_bool_type = [torch.bool]
    torch_int_types = [torch.int8, torch.int16, torch.int32, torch.int64]
    torch_float_types = [torch.float32, torch.float64]

    @pytest.mark.parametrize("shape, dtype", product(shapes, np_bool_type + np_int_types + np_float_types))
    def test_shape(self, shape, dtype):
        array = np.empty(shape, dtype)
        tensor = array_to_tensor(array)
        assert tensor.shape == array.shape
    
    @pytest.mark.parametrize("shape, dtypes", product(shapes, zip(np_int_types, torch_int_types)))
    def test_np_to_tensor_int(self, shape, dtypes):
        np_dtype, torch_dtype = dtypes
        array = np.empty(shape, np_dtype)
        tensor = array_to_tensor(array)
        assert tensor.dtype == torch_dtype
    
    @pytest.mark.parametrize("shape, dtype", product(shapes, np_bool_type))
    def test_np_to_tensor_bool(self, shape, dtype):
        array = np.empty(shape, dtype)
        tensor = array_to_tensor(array)
        assert tensor.dtype == torch.bool

    # @pytest.mark.parametrize("shape, np_dtype, tensor_dtype", product(shapes, np_float_types, torch_float_types))
    # def test_np_to_tensor_float(self, shape, np_dtype, tensor_dtype):
    #     torch.set_default_dtype(tensor_dtype)
    #     array = np.empty(shape, np_dtype)
    #     tensor = array_to_tensor(array)
    #     assert tensor.dtype == tensor_dtype

if __name__ == "__main__":
    pass
