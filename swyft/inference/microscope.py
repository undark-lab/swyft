import logging
from typing import Callable

import swyft
from swyft.inference.posteriors import Posteriors
from swyft.networks import DefaultHead, DefaultTail
from swyft.utils import all_finite, format_param_list

log = logging.getLogger(__name__)


class RoundStatus:
    Initialized = 0
    Simulated = 1
    Done = 2


class MissingModelError(Exception):
    pass


class Microscope:
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
        head=DefaultHead,
        tail=DefaultTail,
        new_simulation_factor: float = 1.5,
        new_simulation_term: int = 0,
        convergence_ratio=0.8,
        epsilon_threshold=-13,
        train_args: dict = {},
        head_args: dict = {},
        tail_args: dict = {},
    ):
        """Initialize swyft.

        next_round_num_samples = new_simulation_factor * last_round_num_samples + new_simulation_term

        Args:
            partitions,
            prior (Prior): Prior model.
            obs (dict): Target observation
            store (Store): Storage for simulator results.  If none, create MemoryStore.
            simhook (function): Noise model, optional.
            device (str): Device.
            Ninit (int): Initial number of training points.
            head
            tail
            new_simulation_factor (float >= 1)
            new_simulation_term (int >= 0)
            convergence_ratio (float > 0.): Convergence ratio between new_volume / old_volume.
            epsilon_threshold: log ratio cutoff
        """
        assert new_simulation_factor >= 1.0
        assert new_simulation_term >= 0
        assert convergence_ratio > 0.0
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
            train_args=train_args,
            head_args=head_args,
            tail_args=tail_args,
            head=head,
            tail=tail,
            convergence_ratio=convergence_ratio,
            new_simulation_factor=new_simulation_factor,
            new_simulation_term=new_simulation_term,
            epsilon_threshold=epsilon_threshold,
        )
        self._initial_prior = prior  # Initial prior

        self._datasets = []
        self._posteriors = []
        self._next_priors = []
        self._initial_n_simulations = len(store)
        self._n_simulations = []

    def status(self):
        pass

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

    @property
    def constrained_prior(self):
        """Last prior before criterion."""
        return swyft.Posteriors.from_Microscope(self)._prior

    @property
    def elapsed_rounds(self):
        return len(self._next_priors)

    def _calculate_new_N(self, N_prev):
        return int(N_prev * self._config["new_simulation_factor"]) + int(
            self._config["new_simulation_term"]
        )

    def _convergence_criterion(self, v_old, v_new):
        return v_new / v_old > self._config["convergence_ratio"]

    def focus(
        self,
        max_rounds: int = 10,
        custom_new_n: Callable = None,
    ) -> int:
        """[summary]

        Args:
            max_rounds (int, optional): Maximum number of rounds per invocation of `focus`. Defaults to 10.

        Returns:
            elapsed_rounds (int)
        """
        while self._status != 5 and len(self._next_priors) < max_rounds:
            # Define dataset
            if self._status == 0:
                log.debug("Starting round %i" % (len(self._datasets) + 1))
                log.debug(
                    "Step 0: Initializing dataset for round %i"
                    % (len(self._datasets) + 1)
                )
                if len(self._datasets) == 0:
                    N = self._config["Ninit"]
                elif custom_new_n is not None:
                    N = custom_new_n(self)
                else:
                    N_prev = len(self._datasets[-1])
                    N = self._calculate_new_N(N_prev)
                log.debug("  dataset size N = %i" % N)
                prior = (
                    self._initial_prior
                    if len(self._next_priors) == 0
                    else self._next_priors[-1]
                )
                dataset = swyft.Dataset(
                    N, prior, store=self._store, simhook=self._simhook
                )

                self._n_simulations.append(len(self._store))
                self._datasets.append(dataset)
                self._status = 1

            # Perform simulations
            elif self._status == 1:
                log.debug(
                    "Step 1: Perform simulations for round %i" % (len(self._datasets))
                )
                if self._datasets[-1].requires_sim:
                    self._datasets[-1].simulate()
                    if self._datasets[-1].requires_sim:
                        return
                else:
                    self._status = 2

            # Perform training
            elif self._status == 2:
                print(f"Round {self.elapsed_rounds}")
                log.debug("Step 2: Training for round %i" % (len(self._datasets)))
                dataset = self._datasets[-1]
                posteriors = Posteriors(dataset)
                posteriors.infer(
                    self._partitions,
                    device=self._device,
                    train_args=self._config["train_args"],
                    head=self._config["head"],
                    tail=self._config["tail"],
                    head_args=self._config["head_args"],
                    tail_args=self._config["tail_args"],
                )

                self._posteriors.append(posteriors)
                self._status = 3

            # Generate new constrained prior
            elif self._status == 3:
                log.debug(
                    "Step 3: Generate new bounds from round %i" % (len(self._datasets))
                )
                posteriors = self._posteriors[-1]
                bound = swyft.Bound.from_Posteriors(
                    self._partitions,
                    posteriors,
                    self._obs,
                    th=self._config["epsilon_threshold"],
                )
                prior = self._datasets[-1].prior
                new_prior = prior.rebounded(bound)

                self._next_priors.append(new_prior)
                self._status = 4

            # Check convergence
            elif self._status == 4:
                log.debug(
                    "Step 4: Convergence check for round %i" % (len(self._datasets))
                )
                v_old = self._datasets[-1].prior.bound.volume
                v_new = self._next_priors[-1].bound.volume
                log.debug("  old volume = %.4g" % v_old)
                log.debug("  new volume = %.4g" % v_new)
                converged = self._convergence_criterion(v_old, v_new)
                self._status = 5 if converged else 0

    @property
    def volumes(self):
        return [self._datasets[0].prior.bound.volume] + [
            prior.bound.volume for prior in self._next_priors
        ]

    @property
    def N(self):
        return [len(dataset) for dataset in self._datasets]

    @property
    def n_simulations(self):
        return [ns - self._initial_n_simulations for ns in self._n_simulations]

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
