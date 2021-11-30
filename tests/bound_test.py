import numpy as np
from swyft.prior import get_uniform_prior
from swyft.bounds import UnitCubeBound, RectangleBound, BallsBound
from swyft import PriorTruncator

# Define a prior with 3 parameters
low = np.zeros(3)
high = np.array([0.7, 1.0, 0.5])
prior = get_uniform_prior(low, high)


class TestBoundSeed:
    def test_UnitCubeBound_seed(self):
        bound1 = UnitCubeBound(3)
        bound1.set_seed(1234)
        pdf1 = PriorTruncator(prior, bound1)
        bound2 = UnitCubeBound(3)
        bound2.set_seed(1234)
        pdf2 = PriorTruncator(prior, bound2)
        assert np.all(pdf1.sample(10) == pdf2.sample(10))

    def test_RectangleBound_seed(self):
        bound1 = RectangleBound(np.stack((low, high), axis=1))
        bound1.set_seed(1234)
        pdf1 = PriorTruncator(prior, bound1)
        bound2 = RectangleBound(np.stack((low, high), axis=1))
        bound2.set_seed(1234)
        pdf2 = PriorTruncator(prior, bound2)
        assert np.all(pdf1.sample(10) == pdf2.sample(10))

