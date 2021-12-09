from typing import Tuple

import pytest

from swyft.types import MarginalIndex, StrictMarginalIndex
from swyft.utils.marginals import tupleize_marginal_indices
from swyft.utils.misc import depth


class TupleizeMarginals:
    marginal_indices = [
        0,
        [0, 1, 2],
        [0, [1, 2]],
        [0, 1, [2, 3]],
        [0, 1, [2, 3], [3, 4]],
        [[0, 1], [2, 3]],
        (0, (1, 2), 2),
        (0, 1),
    ]
    truth = [
        ((0,),),
        ((0,), (1,), (2,)),
        ((0,), (1, 2)),
        ((0,), (1,), (2, 3)),
        ((0,), (1,), (2, 3), (3, 4)),
        ((0, 1), (2, 3)),
        ((0,), (1, 2), (2,)),
        ((0, 1),),
    ]
    str_to_truth = {str(mi): t for mi, t in zip(marginal_indices, truth)}

    @pytest.mark.parametrize(
        "marginal_index",
        marginal_indices,
    )
    def test_compare_to_truth(self, mi: MarginalIndex) -> None:
        assert tupleize_marginal_indices(mi) == self.str_to_truth[str(mi)]

    @pytest.mark.parametrize(
        "marginal_index",
        marginal_indices,
    )
    def test_depth(mi: MarginalIndex) -> None:
        assert depth(tupleize_marginal_indices(mi)) == 2

    @pytest.mark.parametrize(
        "marginal_index",
        marginal_indices,
    )
    def test_tuple(mi: MarginalIndex) -> StrictMarginalIndex:
        mi = tupleize_marginal_indices(mi)
        assert isinstance(mi, Tuple)

    @pytest.mark.parametrize(
        "marginal_index",
        marginal_indices,
    )
    def test_nested_tuple(mi: MarginalIndex) -> StrictMarginalIndex:
        mi = tupleize_marginal_indices(mi)
        assert all(isinstance(i, Tuple) for i in mi)

    def test_sorting():
        raise NotImplementedError("Make sure the sorting is good.")


if __name__ == "__main__":
    pass
