from rllab.policies.linear_gaussian_policy import LinearGaussianPolicy
from rllab.envs.normalized_env import normalize
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
import numpy as np
import theano
import theano.tensor as TT
from lasagne.updates import adam

env = normalize(DoublePendulumEnv())
policy = LinearGaussianPolicy(env.spec)

print(policy)
