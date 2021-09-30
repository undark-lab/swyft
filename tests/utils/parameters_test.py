from typing import Tuple

import pytest

from swyft.types import MarginalIndex, StrictMarginalIndex
from swyft.utils.parameters import tupleize_marginals
from swyft.utils.utils import depth


class TupleizeMarginals:
    marginal_indices = [
        0,
        [0, 1, 2],
        [0, [1, 2]],
        [0, 1, [2, 3]],
        [0, 1, [2, 3], [3, 4]],
        [[0, 1], [2, 3]],
        (0, (1, 2), 2),
    ]

    @pytest.mark.parametrize(
        "marginal_index",
        marginal_indices,
    )
    def test_depth(mi: MarginalIndex) -> None:
        assert depth(tupleize_marginals(mi)) == 2

    @pytest.mark.parametrize(
        "marginal_index",
        marginal_indices,
    )
    def test_tuple(mi: MarginalIndex) -> StrictMarginalIndex:
        mi = tupleize_marginals(mi)
        assert isinstance(mi, Tuple)

    @pytest.mark.parametrize(
        "marginal_index",
        marginal_indices,
    )
    def test_nested_tuple(mi: MarginalIndex) -> StrictMarginalIndex:
        mi = tupleize_marginals(mi)
        assert all(isinstance(i, Tuple) for i in mi)

    def test_sorting():
        raise NotImplementedError("Make sure the sorting is good.")


if __name__ == "__main__":
    pass
