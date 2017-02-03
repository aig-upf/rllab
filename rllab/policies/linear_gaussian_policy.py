from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.policies.base import Policy
from rllab.misc.overrides import overrides
from rllab.misc import ext
import numpy as np

class LinearGaussianPolicy(Policy, Serializable):
    def __init__(
            self,
            env_spec,
    ):
        Serializable.quick_init(self, locals())

        # for the moment a simple diagonal gaussian distribution
        action_dim = env_spec.action_space.flat_dim
        self._dist = DiagonalGaussian(action_dim)

        super(LinearGaussianPolicy, self).__init__(env_spec=env_spec)

        self._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var],
        )

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_params_internal(self, **tags):
        return []

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        means, log_stds = self._f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        pass

    @property
    def distribution(self):
        # Just a placeholder
        return Delta()
