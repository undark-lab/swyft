import logging
from warnings import warn

import numpy as np

from swyft.store import MemoryStore
from swyft.inference import DefaultHead, DefaultTail, RatioCollection, JoinedRatioCollection
from swyft.ip3 import Dataset
from swyft.ip3.exceptions import NoPointsError
from swyft.marginals import PosteriorCollection
from swyft.marginals.prior import Prior
from swyft.marginals.bounds import Bound
from swyft.utils import all_finite, format_param_list
from swyft.posteriors import Posteriors

logging.basicConfig(level=logging.DEBUG, format="%(message)s")


class RoundStatus:
    Initialized = 0
    Simulated = 1
    Done = 2

class MissingModelError(Exception):
    pass


class NestedInference:
    """Main SWYFT interface class."""

    def __init__(
        self,
        partitions,
        prior,
        obs,
        store=None,
        simhook=None,
        device="cpu",
        Ninit=3000,
        Nmax=100000,

        head=DefaultHead,
        tail=DefaultTail,

        density_factor=2.0,
        train_args: dict = {},
        head_args: dict = {},
        tail_args: dict = {},
        log_volume_convergence_threshold=0.1,
    ):
        """Initialize swyft.

        Args:
            prior (Prior): Prior model.
            obs (dict): Target observation
            noise (function): Noise model, optional.
            store (Store): Storage for simulator results.  If none, create MemoryStore.
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
        self._simhook = simhook
        self._store = store
        self._device = device
        self._status = 0
        self._partitions = format_param_list(partitions)

        # Stored in state_dict()
        self._obs = obs
        self._config = dict(
            Ninit=Ninit,
            Nmax=Nmax,
            train_args=train_args,
            head_args=head_args,
            tail_args=tail_args,
        )
        self._initial_prior = prior  # Initial prior

        self._datasets = []
        self._posteriors = []
        self._priors = []

    def status(self):
        pass

#    @property
#    def num_elapsed_rounds(self):
#        """Number of rounds."""
#        return len(self._history)
#
    @property
    def obs(self):
        """The target observation."""
        return self._obs

    @property
    def posteriors(self):
        return self._posteriors

    @property
    def datasets(self):
        return self._datasets

    @property
    def prior(self):
        """Original (unconstrained) prior."""
        return self._initial_prior

    def _update_crit(self, N_prev, volumes):
        return int(N_prev*1.5)

    def _conv_crit(self, v_old, v_new):
        return v_new/v_old > 0.8

    def run(
        self,
        max_rounds: int = 10,
    ):
        """

        Args:
            train_args (dict): Training keyword arguments.
            head (swyft.Module instance or type): Head network (optional).
            tail (swyft.Module instance or type): Tail network (optional).
            head_args (dict): Keyword arguments for head network instantiation.
            tail_args (dict): Keyword arguments for tail network instantiation.
            max_rounds (int): Maximum number of rounds per invokation of `run`, default 10.
        """

        while self._status != 5 and len(self._priors) < max_rounds:
            # Define dataset
            if self._status == 0:
                logging.info("Starting round %i"%(len(self._datasets)+1))
                logging.debug("Step 0: Initializing dataset for round %i"%(len(self._datasets)+1))
                if len(self._datasets) == 0:
                    N = self._config["Ninit"]
                else:
                    N_prev = len(self._datasets[-1])
                    N = self._update_crit(N_prev, self.volumes)
                logging.debug("  dataset size N = %i"%N)
                prior = self._initial_prior if len(self._priors) == 0 else self._priors[-1]
                dataset = Dataset(N, prior, store = self._store, simhook = self._simhook)

                self._datasets.append(dataset)
                self._status = 1

            # Perform simulations
            elif self._status == 1:
                logging.debug("Step 1: Perform simulations for round %i"%(len(self._datasets)))
                if self._datasets[-1].requires_sim:
                    #if self._store._simulator is not None:
                    self._datasets[-1].simulate()
                    #else:
                    #    return
                else:
                    self._status = 2

            # Perform training
            elif self._status == 2:
                logging.debug("Step 2: Training for round %i"%(len(self._datasets)))
                posteriors = Posteriors(dataset)
                posteriors.infer(self._partitions, device = self._device, train_args = self._config["train_args"])

                self._posteriors.append(posteriors)
                self._status = 3

            # Generate new constrained prior
            elif self._status == 3:
                logging.debug("Step 3: Generate new bounds from round %i"%(len(self._datasets)))
                posteriors = self._posteriors[-1]
                bound = Bound.from_Posteriors(self._partitions, posteriors, self._obs, th = -13)
                new_prior = prior.rebounded(bound)

                self._priors.append(new_prior)
                self._status = 4

            # Check convergence
            elif self._status == 4:
                logging.debug("Step 4: Convergence check for round %i"%(len(self._datasets)))
                v_old = self._datasets[-1].prior.bound.volume
                v_new = self._priors[-1].bound.volume
                logging.debug("  old volume = %.4g"%v_old)
                logging.debug("  new volume = %.4g"%v_new)
                converged = self._conv_crit(v_old, v_new)
                self._status = 5 if converged else 0

    @property
    def volumes(self):
        return [self._datasets[0].prior.bound.volume] + [prior.bound.volume for prior in self._priors]

    @property
    def N(self):
        return [len(dataset) for dataset in self._datasets]

    @property
    def converged(self):
        return self._status == 5

    @property
    def requires_sim(self):
        return self._status == 2


#    def state_dict(self):
#        """Return `state_dict`."""
#        return dict(
#            base_prior=self._base_prior.state_dict(),
#            obs=self._obs,
#            history=self._history_state_dict(self._history),
#            converged=self.converged,
#            config=self._config,
#        )
#
#    @classmethod
#    def from_state_dict(cls, state_dict, model, noise=None, store=None, device="cpu"):
#        """Instantiate NestedRatios from saved `state_dict`."""
#        base_prior = Prior.from_state_dict(state_dict["base_prior"])
#        obs = state_dict["obs"]
#        config = state_dict["config"]
#
#        nr = cls(
#            model, base_prior, obs, noise=noise, store=store, device=device, **config
#        )
#
#        nr.converged = state_dict["converged"]
#        nr._history = cls._history_from_state_dict(state_dict["history"])
#        return nr
