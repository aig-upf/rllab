

from rllab.algos.npireps import NPIREPS
from rllab.sampler.pi_sampler import PISampler
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(DoublePendulumEnv())
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = NPIREPS(
    env=env,
    policy=policy,
    baseline=baseline,
    sampler_cls=PISampler
)
algo.train()
