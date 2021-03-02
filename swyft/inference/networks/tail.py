# pylint: disable=no-member,
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from swyft.nn.linear import LinearWithChannel
from swyft.nn.module import Module
from swyft.nn.normalization import OnlineNormalizationLayer


def _get_z_shape(param_list):
    return (len(param_list), max([len(c) for c in param_list]))


# TODO: Remove redundant combine functions
def _combine(params, param_list):
    """Combine parameters according to parameter list. Supports one batch dimension."""
    shape = params[list(params)[0]].shape
    device = params[list(params)[0]].device
    z_shape = _get_z_shape(param_list)
    if len(shape) == 0:  # No batching
        z = torch.zeros(z_shape).to(device)
        for i, c in enumerate(param_list):
            pars = torch.stack([params[k] for k in c]).T
            z[i, : pars.shape[0]] = pars
    else:  # Batching
        n = shape[0]
        z = torch.zeros((n,) + z_shape).to(device)
        for i, c in enumerate(param_list):
            pars = torch.stack([params[k] for k in c]).T
            z[:, i, : pars.shape[1]] = pars
    return z


class DefaultTail(Module):
    def __init__(
        self,
        n_features,
        param_list,
        n_tail_features=2,
        p=0.0,
        hidden_layers=[256, 256, 256],
        online_norm=True,
        param_transform=None,
        tail_features=False,
    ):
        super().__init__(
            n_features,
            param_list,
            n_tail_features=n_tail_features,
            p=p,
            hidden_layers=hidden_layers,
            online_norm=online_norm,
            param_transform=param_transform,
            tail_features=tail_features,
        )
        self.param_list = param_list

        n_channels, pdim = _get_z_shape(param_list)
        self.n_channels = n_channels
        self.tail_features = tail_features

        # Feature compressor
        if self.tail_features:
            n_hidden = 256
            self.fcA = LinearWithChannel(n_channels, n_features, n_hidden)
            self.fcB = LinearWithChannel(n_channels, n_hidden, n_hidden)
            self.fcC = LinearWithChannel(n_channels, n_hidden, n_tail_features)
        else:
            n_tail_features = n_features

        # Pre-network parameter transformation hook
        self.param_transform = param_transform

        # Online normalization of (transformed) parameters
        if online_norm:
            self.onl_z = OnlineNormalizationLayer(torch.Size([n_channels, pdim]))
        else:
            self.onl_z = lambda z: z

        # Ratio estimator
        if isinstance(p, float):
            p = [p for i in range(len(hidden_layers))]
        ratio_estimator_config = [
            LinearWithChannel(n_channels, pdim + n_tail_features, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(p=p[0]),
        ]
        for i in range(len(hidden_layers) - 1):
            ratio_estimator_config += [
                LinearWithChannel(n_channels, hidden_layers[i], hidden_layers[i + 1]),
                nn.ReLU(),
                nn.Dropout(p=p[i + 1]),
            ]
        ratio_estimator_config += [LinearWithChannel(n_channels, hidden_layers[-1], 1)]
        self.ratio_estimator = nn.Sequential(*ratio_estimator_config)

        self.af = nn.ReLU()

    def forward(self, f, params):
        """Forward pass tail network.  Can handle one batch dimension.

        Args:
            f (tensor): feature vectors with shape (n_batch, n_features)
            params (dict): parameter dictionary, with parameter shape (n_batch,)

        Returns:
            lnL (tensor): lnL ratio with shape (n_batch, len(param_list))
        """
        # Parameter transform hook
        if self.param_transform is not None:
            params = self.param_transform(params)

        # Feature compressors independent per channel
        f = f.unsqueeze(1).repeat(
            1, self.n_channels, 1
        )  # (n_batch, n_channels, n_features)
        if self.tail_features:
            f = self.af(self.fcA(f))
            f = self.af(self.fcB(f))
            f = self.fcC(f)

        # Channeled density estimator
        z = _combine(params, self.param_list)
        z = self.onl_z(z)

        x = torch.cat([f, z], -1)
        x = self.ratio_estimator(x)
        x = x.squeeze(-1)
        return x


class GenericTail(nn.Module):
    def __init__(
        self,
        num_observation_features: int,
        parameter_list: list,
        get_ratio_estimator: Callable[[int, int], nn.Module],
        get_observation_embedding: Optional[Callable[[int, int], nn.Module]] = None,
        get_parameter_embedding: Optional[Callable[[int, int], nn.Module]] = None,
        online_z_score_obs: bool = True,
        online_z_score_par: bool = True,
    ):
        """Returns an object suitable for use as a tail in NestedRatios.

        For the various get_* callables, we recommend use of the functools.partial function.

        Args:
            num_observation_features (int): dimensionality of observation
            parameter_list (list): list of parameter names
            get_ratio_estimator (Callable[[int, int], nn.Module]): function taking num_channels, dim_observation_embedding + dim_parameter_embedding to torch Module
            get_observation_embedding (Optional[Callable[[int, int], nn.Module]], optional): function taking num_channels, num_observation_features to torch Module. Defaults to None.
            get_parameter_embedding (Optional[Callable[[int, int], nn.Module]], optional): function taking num_channels, num_parameters to torch Module. Defaults to None.
            online_z_score_obs (bool, optional): perform standard scoring of observation before embedding. Defaults to True.
            online_z_score_par (bool, optional): perform standard scoring of parameter before embedding. Defaults to True.
        """
        super().__init__()
        self.register_buffer(
            "num_observation_features", torch.tensor(num_observation_features)
        )
        self.parameter_list = parameter_list
        # self.register_buffer("parameter_list", torch.tensor(parameter_list))  # How to save the params list??
        num_channels, num_parameters = _get_z_shape(self.parameter_list)
        self.register_buffer("num_channels", torch.tensor(num_channels))
        self.register_buffer("num_parameters", torch.tensor(num_parameters))

        self.online_normalization_observations = (
            OnlineNormalizationLayer((num_channels, num_observation_features))
            if online_z_score_obs
            else nn.Identity()
        )
        self.online_normalization_parameters = (
            OnlineNormalizationLayer((num_channels, num_parameters))
            if online_z_score_par
            else nn.Identity()
        )

        if get_observation_embedding is None:
            self.embed_observation = torch.nn.Identity()
        else:
            self.embed_observation = get_observation_embedding(
                num_channels, num_observation_features
            )
        _, _, dim_observation_embedding = self.embed_observation(
            torch.zeros(10, num_channels, num_observation_features)
        ).shape

        if get_parameter_embedding is None:
            self.embed_parameter = torch.nn.Identity()
        else:
            self.embed_parameter = get_parameter_embedding(num_channels, num_parameters)
        _, _, dim_parameter_embedding = self.embed_parameter(
            torch.zeros(10, num_channels, num_parameters)
        ).shape

        self.ratio_estimator = get_ratio_estimator(
            num_channels,
            dim_observation_embedding + dim_parameter_embedding,
        )

    def _channelize_observation(self, observation):
        shape = observation.shape
        return observation.unsqueeze(-2).expand(
            *shape[:-1], self.num_channels, shape[-1]
        )

    def forward(self, observation: torch.Tensor, parameters: Dict[str, torch.Tensor]):
        obs = self._channelize_observation(observation)
        par = _combine(parameters, self.parameter_list)

        obs_zscored = self.online_normalization_observations(obs)
        par_zscored = self.online_normalization_parameters(par)

        obs_embedded = self.embed_observation(obs_zscored)
        par_embedded = self.embed_parameter(par_zscored)

        both = torch.cat([obs_embedded, par_embedded], dim=-1)
        out = self.ratio_estimator(both)
        return out.squeeze(-1)
