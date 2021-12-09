import tempfile
from itertools import product
from pathlib import Path
from typing import Dict, Hashable, Optional

import numpy as np
import pytest
import torch

import swyft.inference.marginalratioestimator as mre
import swyft.networks.classifier as classifier
from swyft.bounds import Bound, RectangleBound
from swyft.inference.marginalposterior import MarginalPosterior
from swyft.prior import (
    Prior,
    PriorTruncator,
    get_diagonal_normal_prior,
    get_uniform_prior,
)
from swyft.store.dataset import SimpleDataset
from swyft.types import MarginalIndex
from swyft.utils import tupleize_marginal_indices


class AllOneNetwork(torch.nn.Module, classifier.HeadTailClassifier):
    def __init__(self, marginal_indices: MarginalIndex) -> None:
        super().__init__()
        self.marginal_indices = tupleize_marginal_indices(marginal_indices)
        self.block_shape = classifier.ParameterTransform.get_marginal_block_shape(
            self.marginal_indices
        )
        self.m, _ = self.block_shape
        self.register_buffer("out", torch.ones(1, self.m))

    def forward(self, _, y: torch.Tensor) -> torch.Tensor:
        b, *_ = y.shape
        return self.out.expand(b, self.m)

    def head(self, _: Dict[Hashable, torch.Tensor]) -> torch.Tensor:
        return self.out

    def tail(self, features: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        return self(features, parameters)


class TestMarginalPosterior:
    @classmethod
    def setup_class(cls):
        cls.observation_key = "x"
        cls.observation_shapes = {cls.observation_key: (10,)}
        cls.n_parameters = 2
        cls.device = "cpu"

    @classmethod
    def get_marginal_ratio_estimator(cls, marginal_indices: MarginalIndex):
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

    def test_init_prior(self):
        n_parameters = 2
        prior = get_uniform_prior([-0.5] * n_parameters, [5.0] * n_parameters)

        marginal_indices = list(range(n_parameters))
        mre = self.get_marginal_ratio_estimator(marginal_indices)

        mp = MarginalPosterior(mre, prior)
        assert isinstance(mp.prior, Prior)

    def test_init_prior_truncator(self):
        bound = [[0.1, 0.9], [0.2, 0.8]]
        bound = np.asarray(bound)
        rectangle_bound = RectangleBound(bound)

        n_parameters = len(bound)
        prior = get_uniform_prior([-0.5] * n_parameters, [5.0] * n_parameters)

        marginal_indices = list(range(n_parameters))
        mre = self.get_marginal_ratio_estimator(marginal_indices)

        mp = MarginalPosterior(mre, prior, rectangle_bound)
        assert isinstance(mp.prior_truncator, PriorTruncator)

    @pytest.mark.parametrize("marginal_indices", [[0], [0, 1], [(0, 1)]])
    def test_truncate(self, marginal_indices: MarginalIndex):
        prior = get_uniform_prior([-2.5] * self.n_parameters, [5.0] * self.n_parameters)
        marginal_indices = list(range(self.n_parameters))
        network = AllOneNetwork(marginal_indices)
        marginal_ratio_estimator = mre.MarginalRatioEstimator(
            marginal_indices=marginal_indices,
            network=network,
            device=self.device,
        )
        mp = MarginalPosterior(marginal_ratio_estimator, prior)

        n_samples = 1_000
        fabricated_observation = {
            key: torch.rand(*shape) for key, shape in self.observation_shapes.items()
        }
        bound = mp.truncate(
            n_samples,
            fabricated_observation,
            threshold=-50.0,
        )  # TODO make this test the actual behavior of the function.
        assert isinstance(bound, Bound)

    @pytest.mark.parametrize("marginal_indices", [[0], [0, 1], [(0, 1)]])
    def test_empirical_mass(self, marginal_indices):
        prior = get_uniform_prior([-2.5] * self.n_parameters, [5.0] * self.n_parameters)
        marginal_indices = list(range(self.n_parameters))
        network = AllOneNetwork(marginal_indices)
        marginal_ratio_estimator = mre.MarginalRatioEstimator(
            marginal_indices=marginal_indices,
            network=network,
            device=self.device,
        )
        mp = MarginalPosterior(marginal_ratio_estimator, prior)

        n_observations = 100
        us = torch.randn(n_observations, self.n_parameters)
        vs = torch.randn(n_observations, self.n_parameters)
        observations = {
            key: torch.rand(n_observations, *shape)
            for key, shape in self.observation_shapes.items()
        }
        dataset = SimpleDataset(observations, us, vs)

        empirical_mass, _ = mp.empirical_mass(
            n_observations=n_observations,
            n_posterior_samples=1000,
            dataset=dataset,
        )  # TODO make this test the actual behavior of the function.
        assert isinstance(empirical_mass, dict)

    @pytest.mark.skip
    def test_log_prob_value_with_fake_mre(self):
        raise NotImplementedError(
            "This test could be constructed using a conjugate distribution and a 'decoy' marginal ratio estimator."
        )

    @pytest.mark.parametrize(
        "marginal_indices, batch_size",
        product(
            [[0], [0, 1], [(0, 1)]],
            [None, 10],
        ),
    )
    def test_log_prob_shape(
        self, marginal_indices: MarginalIndex, batch_size: Optional[int]
    ):
        n_batch = 100
        marginal_indices = tupleize_marginal_indices(marginal_indices)
        marginal_ratio_estimator = self.get_marginal_ratio_estimator(marginal_indices)

        prior = get_diagonal_normal_prior(
            loc=[0] * self.n_parameters, scale=[1] * self.n_parameters
        )

        marginal_posterior = MarginalPosterior(marginal_ratio_estimator, prior)

        fabricated_observation = {
            key: torch.rand(*shape) for key, shape in self.observation_shapes.items()
        }
        fabricated_v = torch.randn(n_batch, self.n_parameters)
        log_prob = marginal_posterior.log_prob(
            observation=fabricated_observation,
            v=fabricated_v,
            batch_size=batch_size,
        )
        assert set(log_prob.keys()) == set(marginal_indices)
        for _, value in log_prob.items():
            assert value.shape == (n_batch,)

    @pytest.mark.parametrize(
        "n_samples, marginal_indices, batch_size",
        product(
            [101, 1000],
            [[0], [0, 1], [(0, 1)]],
            [None, 10],
        ),
    )
    def test_weighted_sample_shape(
        self, n_samples: int, marginal_indices: MarginalIndex, batch_size: Optional[int]
    ):
        marginal_indices = tupleize_marginal_indices(marginal_indices)
        n_marginal_parameters = len(marginal_indices[0])
        marginal_ratio_estimator = self.get_marginal_ratio_estimator(marginal_indices)

        prior = get_diagonal_normal_prior(
            loc=[0] * self.n_parameters, scale=[1] * self.n_parameters
        )

        marginal_posterior = MarginalPosterior(marginal_ratio_estimator, prior)

        fabricated_observation = {
            key: torch.rand(*shape) for key, shape in self.observation_shapes.items()
        }
        weighted_samples = marginal_posterior.weighted_sample(
            n_samples=n_samples,
            observation=fabricated_observation,
            batch_size=batch_size,
        )

        assert set(weighted_samples.marginal_indices) == set(marginal_indices)
        assert weighted_samples.v.shape == (
            n_samples,
            self.n_parameters,
        ), "size of the parameters v does not match"
        for marginal_index in marginal_indices:
            logweight, marginal = weighted_samples.get_logweight_marginal(
                marginal_index
            )
            assert logweight.shape == (n_samples,)
            assert marginal.shape == (
                n_samples,
                n_marginal_parameters,
            ), "size of the marginal parameters does not match"

    @pytest.mark.parametrize(
        "n_samples, marginal_indices, batch_size",
        product(
            [5, 10],
            [[0], [0, 1], [(0, 1)]],
            [None, 10],
        ),
    )
    def test_sample_shape(
        self, n_samples: int, marginal_indices: MarginalIndex, batch_size: Optional[int]
    ):
        np.random.seed(0)
        torch.manual_seed(0)
        marginal_indices = tupleize_marginal_indices(marginal_indices)
        n_marginal_parameters = len(marginal_indices[0])
        marginal_ratio_estimator = self.get_marginal_ratio_estimator(marginal_indices)

        prior = get_diagonal_normal_prior(
            loc=[0] * self.n_parameters, scale=[1] * self.n_parameters
        )

        marginal_posterior = MarginalPosterior(marginal_ratio_estimator, prior)

        fabricated_observation = {
            key: torch.rand(*shape) for key, shape in self.observation_shapes.items()
        }
        samples = marginal_posterior.sample(
            n_samples=n_samples,
            observation=fabricated_observation,
            batch_size=batch_size,
        )
        assert set(samples.keys()) == set(marginal_indices)
        for _, value in samples.items():
            assert value.shape == (n_samples, n_marginal_parameters)


if __name__ == "__main__":
    pass
