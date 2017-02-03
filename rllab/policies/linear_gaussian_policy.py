from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.policies.base import Policy
from rllab.misc.overrides import overrides
from rllab.misc import ext
import theano.tensor as TT
import numpy as np


# For state dims nx1 and control dims dx1
# Parameters are a matrix dxn and a vector dx1, which corresponds to the mean
# of the Gaussian, and a dx1 vector, or diagonal of the covariance

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

        obs_var = env_spec.observation_space.new_tensor_variable(
            'observations',
            # It should have 1 extra dimension since we want to represent a list of observations
            extra_dims=1
        )

        # here we should obtain symbolic expressions 
        self._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var],
        )

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        # compute linear comb of flat_obs
        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        # compute linear comb of flat_obs
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
