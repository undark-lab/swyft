import logging
import typing
from warnings import warn

import numpy as np

from swyft.cache import MemoryCache
from swyft.inference import DefaultHead, DefaultTail, RatioEstimator
from swyft.ip3 import Points
from swyft.ip3.exceptions import NoPointsError
from swyft.marginals import Prior, RatioEstimatedPosterior
from swyft.utils import all_finite, format_param_list
from swyft.utils.simulator import Simulator

logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class MissingModelError(Exception):
    pass


class NestedRatios:
    """Main SWYFT interface class."""

    def __init__(
        self,
        simulator,
        prior,
        obs,
        noise=None,
        cache=None,
        device="cpu",
        Ninit=3000,
        Nmax=100000,
        density_factor=2.0,
        log_volume_convergence_threshold=0.1,
    ):
        """Initialize swyft.

        Args:
            simulator (function): A callablefunction, or an instance of swyft.utils.simulator.Simulator.
            prior (Prior): Prior model.
            obs (dict): Target observation
            noise (function): Noise model, optional.
            cache (Cache): Storage for simulator results.  If none, create MemoryCache.
            Ninit (int): Initial number of training points.
            Nmax (int): Maximum number of training points per round.
            density_factor (float > 1): Increase of training point density per round.
            log_volume_convergence_threshold (float > 0.): Convergence threshold measured as difference between log prior volume to log contrainted prior volume.
            device (str): Device.
        """
        assert density_factor > 1.0
        assert log_volume_convergence_threshold > 0.0
        if not all_finite(obs):
            raise ValueError("obs must be finite.")

        # Not stored
        self._simulator = (
            simulator
            if isinstance(simulator, Simulator)
            else Simulator.from_model(simulator, prior)
        )
        self._noise = noise
        self._cache = cache or MemoryCache(
            self._simulator.params, self._simulator.obs_shapes
        )
        self._device = device

        # Stored in state_dict()
        self._obs = obs
        self._config = dict(
            Ninit=Ninit,
            Nmax=Nmax,
            density_factor=density_factor,
            log_volume_convergence_threshold=log_volume_convergence_threshold,
        )
        self.converged = False
        self.failed_to_converge = False
        self._base_prior = prior  # Initial prior
        self._history = []

    @property
    def num_elapsed_rounds(self):
        """Number of rounds."""
        return len(self._history)

    @property
    def obs(self):
        """The target observation."""
        return self._obs

    @property
    def marginals(self):
        """Marginals from the last round."""
        if self._history is []:
            logging.warning("To generated marginals from NRE, call .run(...).")
        return self._history[-1]["marginals"]

    @property
    def prior(self):
        """Original (unconstrained) prior."""
        return self._base_prior

    def run(
        self,
        train_args: dict = {},
        head=DefaultHead,
        tail=DefaultTail,
        head_args: dict = {},
        tail_args: dict = {},
        max_rounds: int = 10,
        keep_history=False,
    ):
        """Perform 1-dim marginal focus fits.

        Args:
            train_args (dict): Training keyword arguments.
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
            max_rounds (int): Maximum number of rounds per invokation of `run`, default 10.
        """

        param_list = self._cache.params
        r = 0

        while (not self.converged) and (r < max_rounds):
            logging.info("NRE round: R = %i" % (self.num_elapsed_rounds + 1))

            ##################################################
            # Step 1 - get prior and number of training points
            ##################################################
            prior_R, N_R = self._get_prior_R()

            ####################################
            # Step 2 - Update cache and simulate
            ####################################
            indices = self._cache.sample(prior_R, N_R)

            # Fail gracefully if no points are drawn.
            if len(indices) == 0:
                raise NoPointsError("No points were sampled from the cache.")

            self._cache.simulate(self._simulator, indices)
            points = Points(indices, self._cache, self._noise)

            ################
            # Step 3 - Infer
            ################
            try:
                marginals_R = self._train(
                    prior_R,
                    param_list,
                    head=head,
                    tail=tail,
                    head_args=head_args,
                    tail_args=tail_args,
                    train_args=train_args,
                    points=points,
                )
            except NoPointsError:
                self.failed_to_converge = True
                break

            constr_prior_R = marginals_R.gen_constr_prior(self._obs)

            # Update object history
            self._history.append(
                dict(
                    marginals=marginals_R,
                    constr_prior=constr_prior_R,
                    N=N_R,
                )
            )
            r += 1

            # Drop previous marginals
            if (not keep_history) and (self.num_elapsed_rounds > 1):
                self._history[-2]["marginals"] = None
                self._history[-2]["constr_prior"] = None

            # Check convergence
            logging.debug(
                "constr_prior_R : prior_R volume = %.4g : %.4g"
                % (constr_prior_R.volume(), prior_R.volume())
            )
            if (
                np.log(prior_R.volume()) - np.log(constr_prior_R.volume())
                < self._config["log_volume_convergence_threshold"]
            ):
                logging.info("Volume converged.")
                self.converged = True

        if r >= max_rounds:
            self.failed_to_converge = True

    # NOTE: By convention properties are only quantites that we save in state_dict
    def requires_sim(self):
        """Does cache require simulation runs?"""
        return self._cache.requires_sim

    def gen_1d_marginals(
        self,
        params=None,
        N=1000,
        train_args={},
        head=DefaultHead,
        tail=DefaultTail,
        head_args={},
        tail_args={},
    ):
        """Convenience function to generate 1d marginals."""
        param_list = format_param_list(params, all_params=self._cache.params, mode="1d")
        logging.info(f"Generating marginals for: {param_list}")
        return self.gen_custom_marginals(
            param_list,
            N=N,
            train_args=train_args,
            head=head,
            tail=tail,
            head_args=head_args,
            tail_args=tail_args,
        )

    def gen_2d_marginals(
        self,
        params=None,
        N=1000,
        train_args={},
        head=DefaultHead,
        tail=DefaultTail,
        head_args={},
        tail_args={},
    ):
        """Convenience function to generate 2d marginals."""
        param_list = format_param_list(params, all_params=self._cache.params, mode="2d")
        logging.info(f"Generating marginals for: {param_list}")
        return self.gen_custom_marginals(
            param_list,
            N=N,
            train_args=train_args,
            head=head,
            tail=tail,
            head_args=head_args,
            tail_args=tail_args,
        )

    def gen_custom_marginals(
        self,
        param_list,
        N=1000,
        train_args={},
        head=DefaultHead,
        tail=DefaultTail,
        head_args={},
        tail_args={},
    ):
        """Perform custom marginal estimation, based on the most recent constrained prior.

        Args:
            param_list (list of tuples of strings): List of parameters for which inference is performed.
            N (int): Number of training points.
            train_args (dict): Training keyword arguments.
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
        """
        if self.num_elapsed_rounds == 0:
            prior = self._base_prior
        else:
            prior = self._history[-1]["constr_prior"]
            logging.debug("Constrained prior volume = %.4f" % prior.volume())

        param_list = format_param_list(param_list, all_params=self._cache.params)

        ####################################
        # Step 1 - Update cache and simulate
        ####################################
        indices = self._cache.sample(prior, N)

        # Fail gracefully if no points are drawn.
        if len(indices) == 0:
            raise NoPointsError("No points were sampled from the cache.")

        self._cache.simulate(self._simulator, indices)
        points = Points(indices, self._cache, self._noise)

        #################
        # Step 2 - Train!
        #################
        marginals = self._train(
            prior=prior,
            points=points,
            param_list=param_list,
            head=head,
            tail=tail,
            head_args=head_args,
            tail_args=tail_args,
            train_args=train_args,
        )
        return marginals

    @property
    def cache(self):
        """Return simulation cache."""
        return self._cache

    @staticmethod
    def _history_state_dict(history):
        state_dict = [
            {
                "marginals": v["marginals"].state_dict(),
                "constr_prior": v["constr_prior"].state_dict(),
                "N": v["N"],
            }
            for v in history
        ]
        return state_dict

    @staticmethod
    def _history_from_state_dict(history_state_dict):
        history = [
            {
                "marginals": RatioEstimatedPosterior.from_state_dict(v["marginals"]),
                "constr_prior": Prior.from_state_dict(v["constr_prior"]),
                "N": v["N"],
            }
            for v in history_state_dict
        ]
        return history

    def state_dict(self):
        """Return `state_dict`."""
        return dict(
            base_prior=self._base_prior.state_dict(),
            obs=self._obs,
            history=self._history_state_dict(self._history),
            converged=self.converged,
            config=self._config,
        )

    @classmethod
    def from_state_dict(cls, state_dict, model, noise=None, cache=None, device="cpu"):
        """Instantiate NestedRatios from saved `state_dict`."""
        base_prior = Prior.from_state_dict(state_dict["base_prior"])
        obs = state_dict["obs"]
        config = state_dict["config"]

        nr = cls(
            model, base_prior, obs, noise=noise, cache=cache, device=device, **config
        )

        nr.converged = state_dict["converged"]
        nr._history = cls._history_from_state_dict(state_dict["history"])
        return nr

    def _get_prior_R(self):
        # Method does not change internal states

        param_list = self._cache.params
        D = len(param_list)
        if self.num_elapsed_rounds == 0:  # First round
            prior_R = self._base_prior
            N_R = self._config["Ninit"]
        else:  # Subsequent rounds
            prior_R = self._history[-1]["constr_prior"]

            # Derive new number of training points
            prior_Rm1 = self._history[-1]["marginals"].prior
            v_R = prior_R.volume()
            v_Rm1 = prior_Rm1.volume()
            N_Rm1 = self._history[-1]["N"]
            density_Rm1 = N_Rm1 / v_Rm1 ** (1 / D)
            density_R = self._config["density_factor"] * density_Rm1
            N_R = min(max(density_R * v_R ** (1 / D), N_Rm1), self._config["Nmax"])

        logging.info("Number of training samples is N_R = %i" % N_R)
        return prior_R, N_R

    def _train(
        self,
        prior,
        param_list,
        points,
        train_args,
        head,
        tail,
        head_args,
        tail_args,
    ):
        """Perform amortized inference on constrained priors."""
        logging.info("Starting neural network training.")

        if param_list is None:
            param_list = prior.params()

        re = RatioEstimator(
            param_list,
            device=self._device,
            head=head,
            tail=tail,
            tail_args=tail_args,
            head_args=head_args,
        )
        re.train(points, **train_args)

        return RatioEstimatedPosterior(re, prior)
