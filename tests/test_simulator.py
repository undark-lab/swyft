import numpy as np
from scipy import stats
import swyft


class Simulator(swyft.Simulator):
    def __init__(self):
        super().__init__()
        self.transform_samples = swyft.to_numpy32
        self.x = np.linspace(-1, 1, 10)

    def build(self, graph):
        z = graph.node("z", lambda: np.random.rand(2) * 2 - 1)
        f = graph.node("f", lambda z: z[0] + z[1] * self.x, z)
        x = graph.node("x", lambda f: f + np.random.randn(10) * 0.1, f)


def test_simulator():
    sim = Simulator()
    samples = sim.sample(N=10)
