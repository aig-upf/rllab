

#from rllab.algos.pireps import PIREPS
from rllab.algos.reps import REPS
from rllab.sampler.pi_sampler import PISampler
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(PointEnv())
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = REPS(
    env=env,
    policy=policy,
    baseline=baseline,
    sampler_cls=PISampler
)
algo.train()
