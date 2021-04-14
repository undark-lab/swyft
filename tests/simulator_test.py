import pytest
import numpy as np
from dask.distributed import LocalCluster
from swyft.utils.simulator import Simulator

# linear model
def model(params):
    p = np.linspace(-1, 1, 10)  # Nbin = 10
    mu = params["a"] + p * params["b"]
    return dict(x=mu)


def model_none_dict(params):
    mu = None
    return dict(x=mu)


def model_nan_array(params):
    mu = np.linspace(-1, 1, 10)  # Nbin = 10
    mu[0] = np.nan
    return dict(x=mu)


def model_inf_array(params):
    mu = np.linspace(-1, 1, 10)  # Nbin = 10
    mu[0] = np.inf
    return dict(x=mu)


class TestSimulator:
    z = [{"a": 0, "b": 1}, {"a": 0, "b": -1}, {"a": 1, "b": 1}]

    @pytest.mark.parametrize("params", [z])
    def test_run_process(self, params):
        simulator = Simulator(model)
        results = simulator.run(params)
        assert all([r[1] == 0 for r in results])
        assert all(
            [all(model(param)["x"] == r[0]["x"]) for param, r in zip(params, results)]
        )

    @pytest.mark.parametrize("params", [z])
    def test_run_localcluster(self, params):
        simulator = Simulator(model)
        cluster = LocalCluster()
        simulator.set_dask_cluster(cluster)
        results = simulator.run(params)
        assert all([r[1] == 0 for r in results])
        assert all(
            [all(model(param)["x"] == r[0]["x"]) for param, r in zip(params, results)]
        )

    @pytest.mark.parametrize("params", [z])
    def test_run_process_nonevalue(self, params):
        simulator = Simulator(model_none_dict)
        results = simulator.run(params)
        assert all([r[1] == 1 for r in results])

    @pytest.mark.parametrize("params", [z])
    def test_run_process_nanarray(self, params):
        simulator = Simulator(model_nan_array)
        results = simulator.run(params)
        assert all([r[1] == 2 for r in results])

    @pytest.mark.parametrize("params", [z])
    def test_run_process_infarray(self, params):
        simulator = Simulator(model_inf_array)
        results = simulator.run(params)
        assert all([r[1] == 2 for r in results])


if __name__ == "__main__":
    pass
