import tempfile
from itertools import product
from pathlib import Path
from typing import Optional

import pytest
import torch

import swyft.inference.marginalratioestimator as mre
import swyft.networks.classifier as classifier
from swyft.types import MarginalIndex
from swyft.utils.marginals import tupleize_marginal_indices


class TestSplitLengthByPercentage:
    def test_sum_to_1(self):
        """The percentage vector should sum to one."""
        with pytest.raises(AssertionError):
            mre.split_length_by_percentage(10, [0.2, 0.3])

    def test_extra_in_first_position(self):
        """The rounding error goes into the first index."""
        out = mre.split_length_by_percentage(10, [0.61, 0.39])
        assert out == [7, 3]


class TestNTrainNValid:
    dataset_lens = [9, 25, 100]
    validation_amounts = [1, 2, 0.23, 0.5]

    @pytest.mark.parametrize(
        "validation_amount, len_dataset", product(validation_amounts, dataset_lens)
    )
    def test_nvalid_divisible_by_2(self, validation_amount, len_dataset):
        """nvalid should be divisible by 2."""
        _, nvalid = mre.get_ntrain_nvalid(validation_amount, len_dataset)
        assert nvalid % 2 == 0

    def test_wrong_type(self):
        """A non-int or non-float for validation_amount should raise a TypeError."""
        with pytest.raises(TypeError):
            mre.get_ntrain_nvalid(complex(5), 100)

    @pytest.mark.parametrize(
        "validation_amount, len_dataset", product(validation_amounts, dataset_lens)
    )
    def test_nvalid_divisible_by_2(self, validation_amount, len_dataset):
        """nvalid should be divisible by 2."""
        _, nvalid = mre.get_ntrain_nvalid(validation_amount, len_dataset)
        assert nvalid % 2 == 0


def test_double_observation():
    b, o = 2, 3
    a = torch.arange(b * o).reshape(b, o)
    truth = torch.Tensor([[0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5]])
    assert torch.all(mre.double_observation(a) == truth)


class TestDoubleParameters:
    def test_assertion(self):
        with pytest.raises(AssertionError):
            mre.double_parameters(torch.rand(5, 2))

    def test_shape(self):
        with pytest.raises(ValueError):
            mre.double_parameters(torch.rand(6, 2, 2))

    def test_correcness(self):
        b, p = 2, 3
        a = torch.arange(b * p).reshape(b, p)
        truth = torch.Tensor([[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]])
        assert torch.all(mre.double_parameters(a) == truth)


class TestSaveLoadMarginalRatioEstimator:
    @classmethod
    def setup_class(cls):
        cls.directory = tempfile.TemporaryDirectory()

    @classmethod
    def teardown_class(cls):
        cls.directory.cleanup()

    def test_save_load_marginal_ratio_estimator(self):
        # Preliminary
        observation_key = "x"
        n_parameters = 2
        marginal_indices = list(range(n_parameters))
        device = "cpu"

        network = classifier.get_marginal_classifier(
            observation_key=observation_key,
            marginal_indices=marginal_indices,
            observation_shapes={observation_key: (10,)},
            n_parameters=n_parameters,
            hidden_features=32,
            num_blocks=2,
        )

        marginal_ratio_estimator = mre.MarginalRatioEstimator(
            marginal_indices=marginal_indices,
            network=network,
            device=device,
        )

        # Saving
        path = Path(self.directory.name) / f"marginal_ratio_estimator"
        marginal_ratio_estimator.save(path)

        # Loading
        marginal_ratio_estimator_loaded = mre.MarginalRatioEstimator.load(
            network=network,
            device=device,
            filename=path,
        )

        # Testing each loaded piece
        for weights, weights_loaded in zip(
            marginal_ratio_estimator.network.parameters(),
            marginal_ratio_estimator_loaded.network.parameters(),
        ):
            assert torch.all(weights == weights_loaded)

        for buffers, buffers_loaded in zip(
            marginal_ratio_estimator.network.buffers(),
            marginal_ratio_estimator_loaded.network.buffers(),
        ):
            assert torch.all(buffers == buffers_loaded)

        assert marginal_ratio_estimator.device == marginal_ratio_estimator_loaded.device

        # TODO: need to add a test for scheduler and optimizer loading


class TestMarginalRatioEstimator:
    @classmethod
    def setup_class(cls):
        cls.observation_key = "x"
        cls.observation_shapes = {cls.observation_key: (10,)}
        cls.n_parameters = 2
        cls.device = "cpu"

    @classmethod
    def get_marginal_ratio_estimator(cls, marginal_indices):
        network = classifier.get_marginal_classifier(
            observation_key=cls.observation_key,
            marginal_indices=marginal_indices,
            observation_shapes=cls.observation_shapes,
            n_parameters=cls.n_parameters,
            hidden_features=16,
            num_blocks=1,
        )
        marginal_ratio_estimator = mre.MarginalRatioEstimator(
            marginal_indices=marginal_indices,
            network=network,
            device=cls.device,
        )
        return marginal_ratio_estimator

    @pytest.mark.skip
    def test_train(self):
        raise NotImplementedError("Need to test this function.")

    @pytest.mark.parametrize(
        "marginal_indices, batch_size",
        product(
            [[0, 1], [(0, 1)]],  # With these, n_parameters >= 2
            [None, 10],
        ),
    )
    def test_log_ratio_shape(
        self, marginal_indices: MarginalIndex, batch_size: Optional[int]
    ):
        """The log_ratio function should return as many weights as there were n_batches of parameters provided."""
        n_batch = 100
        marginal_indices = tupleize_marginal_indices(marginal_indices)
        marginal_ratio_estimator = self.get_marginal_ratio_estimator(marginal_indices)
        fabricated_observation = {
            key: torch.rand(*shape) for key, shape in self.observation_shapes.items()
        }
        fabricated_v = torch.randn(n_batch, self.n_parameters)
        log_ratio = marginal_ratio_estimator.log_ratio(
            observation=fabricated_observation,
            v=fabricated_v,
            batch_size=batch_size,
        )
        assert set(log_ratio.keys()) == set(marginal_indices)
        for _, value in log_ratio.items():
            assert value.shape == (n_batch,)


if __name__ == "__main__":
    pass
