import logging
from warnings import warn

import numpy as np

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

from .cache import DirectoryCache, MemoryCache
from .estimation import Points, RatioEstimator
from .intensity import Prior
from .network import DefaultHead, DefaultTail
from .types import Dict
from .utils import all_finite, format_param_list


class MissingModelError(Exception):
    pass


class Marginals:
    """Marginal container"""

    def __init__(self, ratio, prior):
        """Marginal container initialization.

        Args:
            re (RatioEstimator)
            prior (Prior)
        """
        self._re = ratio
        self._prior = prior

    @property
    def prior(self):
        """(Constrained) prior of the marginal."""
        return self._prior

    @property
    def ratio(self):
        """Ratio estimator for marginals."""
        return self._re

    def __call__(self, obs, n_samples=100000):
        """Return weighted marginal samples.

        Args:
            obs (dict): Observation.
            n_samples (int): Number of samples.

        Returns:
            dict containing samples.

        Note: Observations must be restricted to constrained prior space to
        lead to valid results.
        """
        # FIXME: Make return of log_priors conditional on constrained prior
        # being factorizable
        return self._re.posterior(obs, self._prior, n_samples=n_samples)

    def state_dict(self):
        """Return state_dict."""
        return dict(re=self._re.state_dict(), prior=self._prior.state_dict())

    @classmethod
    def from_state_dict(cls, state_dict):
        """Instantiate Marginals based on state_dict."""
        return Marginals(
            RatioEstimator.from_state_dict(state_dict["re"]),
            Prior.from_state_dict(state_dict["prior"]),
        )

    def gen_constr_prior(self, obs, th=-10):
        """Generate new constrained prior based on ratio estimator and target observation.

        Args:
            obs (dict): Observation.
            th (float): Cutoff maximum log likelihood ratio. Default is -10,
                        which correspond roughly to 4 sigma.

        Returns:
            Prior: Constrained prior.
        """
        return self._prior.get_masked(obs, self._re, th=th)

    def sample_posterior(
        self, obs: dict, n_samples: int, excess_factor: int = 10, maxiter: int = 100
    ) -> Dict[str, np.ndarray]:
        """Samples from each marginal using rejection sampling.

        Args:
            obs (dict): target observation
            n_samples (int): number of samples in each marginal to output
            excess_factor (int, optional): n_samples_to_reject = excess_factor * n_samples . Defaults to 100.
            maxiter (int, optional): maximum loop attempts to draw n_samples. Defaults to 10.

        Returns:
            Dict[str, np.ndarray]: keys are marginal tuples, values are samples
        
        Reference:
            Section 23.3.3
            Machine Learning: A Probabilistic Perspective
            Kevin P. Murphy
        """
        draw = self._re.posterior(obs, self._prior, n_samples=excess_factor * n_samples)
        maximum_log_likelihood_estimates = {
            k: np.log(np.max(v)) for k, v in draw["weights"].items()
        }
        remaining_param_tuples = set(draw["weights"].keys())
        collector = {k: [] for k in remaining_param_tuples}
        out = {}

        # Do the rejection sampling.
        # When a particular key hits the necessary samples, stop calculating on it to reduce cost.
        # Send that key to out.
        counter = 0
        while counter < maxiter:
            # Calculate chance to keep a sample
            log_prob_to_keep = {
                pt: np.log(draw["weights"][pt]) - maximum_log_likelihood_estimates[pt]
                for pt in remaining_param_tuples
            }

            # Draw and determine if samples are kept
            to_keep = {
                pt: np.less_equal(np.log(np.random.rand(excess_factor * n_samples)), v)
                for pt, v in log_prob_to_keep.items()
            }

            # Collect samples for every tuple of parameters, if there are enough, add them to out.
            for param_tuple in remaining_param_tuples:
                collector[param_tuple].append(
                    np.stack(
                        [
                            draw["params"][name][to_keep[param_tuple]]
                            for name in param_tuple
                        ],
                        axis=-1,
                    )
                )
                concatenated = np.concatenate(collector[param_tuple])[:n_samples]
                if len(concatenated) == n_samples:
                    out[param_tuple] = concatenated

            # Remove the param_tuples which we already have in out, thus not to calculate for them anymore.
            for param_tuple in out.keys():
                if param_tuple in remaining_param_tuples:
                    remaining_param_tuples.remove(param_tuple)

            if len(remaining_param_tuples) > 0:
                draw = self._re.posterior(
                    obs, self._prior, n_samples=excess_factor * n_samples
                )
            else:
                return out
            counter += 1
        warn(
            f"Max iterations {maxiter} reached there were not enough samples produced in {remaining_param_tuples}."
        )
        return out


class NestedRatios:
    """Main SWYFT interface class."""

    def __init__(
        self,
        model,
        prior,
        obs,
        noise=None,
        cache=None,
        device="cpu",
        Ninit=3000,
        Nmax=100000,
        density_factor=2.0,
        volume_conv_th=0.1,
    ):
        """Initialize swyft.

        Args:
            model (function): Simulator function.
            prior (Prior): Prior model.
            obs (dict): Target observation
            noise (function): Noise model, optional.
            cache (Cache): Storage for simulator results.  If none, create MemoryCache.
            Ninit (int): Initial number of training points.
            Nmax (int): Maximum number of training points per round.
            density_factor (float > 1): Increase of training point density per round.
            volume_conv_th (float > 0.): Volume convergence threshold.
            device (str): Device.
        """
        assert density_factor > 1.0
        assert volume_conv_th > 0.0
        if not all_finite(obs):
            raise ValueError("obs must be finite.")

        # Not stored
        self._model = model
        self._noise = noise
        self._cache_reference = cache
        self._device = device

        # Stored in state_dict()
        self._obs = obs
        self._config = dict(
            Ninit=Ninit,
            Nmax=Nmax,
            density_factor=density_factor,
            volume_conv_th=volume_conv_th,
        )
        self._converged = False
        self._base_prior = prior  # Initial prior
        self._history = []

    def converged(self):
        return self._converged

    @property
    def _cache(self):
        if self._cache_reference is None:
            self._cache_reference = MemoryCache.from_simulator(
                self._model, self._base_prior
            )
        return self._cache_reference

    def R(self):
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
        D = len(param_list)

        r = 0

        while (not self.converged()) and (r < max_rounds):
            logging.info("NRE round: R = %i" % (self.R() + 1))

            ##################################################
            # Step 1 - get prior and number of training points
            ##################################################
            prior_R, N_R = self._get_prior_R()

            ####################################
            # Step 2 - Update cache and simulate
            ####################################
            self._cache.grow(prior_R, N_R)
            if self.requires_sim():
                logging.info(
                    "Additional simulations are required after growing the cache."
                )
                if self._model is not None:
                    self._cache.simulate(self._model)
                else:
                    logging.warning(
                        "No model specified. Run simulator directly on cache."
                    )
                    return

            ################
            # Step 3 - Infer
            ################
            marginals_R = self._train(
                prior_R,
                param_list,
                head=head,
                tail=tail,
                head_args=head_args,
                tail_args=tail_args,
                train_args=train_args,
                N=N_R,
            )
            constr_prior_R = marginals_R.gen_constr_prior(self._obs)

            # Update object history
            self._history.append(
                dict(marginals=marginals_R, constr_prior=constr_prior_R, N=N_R,)
            )
            r += 1

            # Drop previous marginals
            if (not keep_history) and (self.R() > 1):
                self._history[-2]["marginals"] = None
                self._history[-2]["constr_prior"] = None

            # Check convergence
            logging.debug(
                "constr_prior_R : prior_R volume = %.4g : %.4g"
                % (constr_prior_R.volume(), prior_R.volume())
            )
            if (
                np.log(prior_R.volume()) - np.log(constr_prior_R.volume())
                < self._config["volume_conv_th"]
            ):
                logging.info("Volume converged.")
                self._converged = True

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
        logging.info("Generating marginals for:", str(param_list))
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
        logging.info("Generating marginals for: %s" % str(param_list))
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
        if self.R() == 0:
            prior = self._base_prior
        else:
            prior = self._history[-1]["constr_prior"]
            logging.debug("Constrained prior volume = %.4f" % prior.volume())

        param_list = format_param_list(param_list, all_params=self._cache.params)

        ####################################
        # Step 1 - Update cache and simulate
        ####################################
        self._cache.grow(prior, N)
        if self.requires_sim():
            logging.info("Additional simulations are required after growing the cache.")
            if self._model is not None:
                self._cache.simulate(self._model)
            else:
                logging.warning("No model specified. Run simulator directly on cache.")
                return

        #################
        # Step 2 - Train!
        #################
        marginals = self._train(
            prior=prior,
            N=N,
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
                "marginals": Marginals.from_state_dict(v["marginals"]),
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
            converged=self._converged,
            config=self._config,
        )

    @classmethod
    def from_state_dict(cls, state_dict, model, noise=None, cache=None, device="cpu"):
        """Instantiate NestedRatios from saved `state_dict`."""
        base_prior = Prior.from_state_dict(state_dict["base_prior"])
        obs = state_dict["obs"]
        config = state_dict["config"]

        nr = NestedRatios(
            model, base_prior, obs, noise=noise, cache=cache, device=device, **config
        )

        nr._converged = state_dict["converged"]
        nr._history = cls._history_from_state_dict(state_dict["history"])
        return nr

    def _get_prior_R(self):
        # Method does not change internal states

        param_list = self._cache.params
        D = len(param_list)
        if self.R() == 0:  # First round
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
        self, prior, param_list, N, train_args, head, tail, head_args, tail_args,
    ):
        """Perform amortized inference on constrained priors."""
        if self._cache.requires_sim:
            raise MissingModelError("Some points in the cache have not been simulated yet.")

        logging.info("Starting neural network training.")
        indices = self._cache.sample(prior, N)
        points = Points(indices, self._cache, self._noise)

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

        return Marginals(re, prior)
