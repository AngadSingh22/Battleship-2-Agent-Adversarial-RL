from __future__ import annotations

import torch as th
from torch import nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BattleshipFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # 5 layers of 3x3 conv gives an 11x11 receptive field, covering the 10x10 board globally.
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros((1, n_input_channels, *observation_space.shape[1:]))
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class BattleshipCnnPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=BattleshipFeatureExtractor,
        )


# ---------------------------------------------------------------------------
# Gap 12: Recurrent policy (formulation solution §3, Eq. 3)
# ---------------------------------------------------------------------------
# The formulation defines a recurrent (finite-memory) attacker policy:
#
#   z_t = f_θ(z_{t-1}, φ(o_t, a_{t-1}))
#   π_θ(a_t | z_t, mask_t)
#
# where z_t is a learned hidden state that accumulates evidence across steps,
# approximating the belief-state update b_{t+1} without expensive enumeration.
#
# We implement this via RecurrentPPO (sb3-contrib) which uses an LSTM over
# a shared feature extractor. The observation encoder φ(o_t, a_{t-1}) is
# realised by concatenating a one-hot encoding of the previous action with
# the CNN embedding — handled through the RecurrentPPO's built-in LSTM.
# ---------------------------------------------------------------------------

class BattleshipRecurrentFeatureExtractor(BaseFeaturesExtractor):
    """CNN encoder φ(o_t) for the recurrent policy.

    The LSTM in RecurrentPPO provides the z_t recurrence; this module handles
    only the spatial observation encoding. The previous action a_{t-1} is
    passed as part of the observation by wrapping the env with a
    ``PreviousActionObsWrapper`` (to be added to ``battleship_rl/envs/``).
    """

    def __init__(self, observation_space, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros((1, n_input_channels, *observation_space.shape[1:]))
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def make_recurrent_policy_kwargs(features_dim: int = 128) -> dict:
    """Return policy_kwargs for RecurrentPPO with the battleship CNN encoder.

    Usage::

        from sb3_contrib import RecurrentPPO
        from battleship_rl.agents.policies import make_recurrent_policy_kwargs

        model = RecurrentPPO(
            policy="MlpLstmPolicy",          # or a custom MaskableLstmPolicy
            env=vec_env,
            policy_kwargs=make_recurrent_policy_kwargs(),
        )

    Note: ``sb3_contrib`` does not yet ship a fully maskable LSTM policy
    (MaskableRecurrentActorCriticPolicy).  Until it does, the recommended
    workaround is:
      1. Use ``RecurrentPPO`` with ``MlpLstmPolicy`` and handle masking via
         a Gymnasium ``ActionMasker`` wrapper (masks applied in env).
      2. Or patch MaskableActorCriticPolicy with LSTM support manually.
    This module documents the architecture so the switch is a one-line change.
    """
    return {
        "features_extractor_class": BattleshipRecurrentFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
        "shared_lstm": True,
    }
