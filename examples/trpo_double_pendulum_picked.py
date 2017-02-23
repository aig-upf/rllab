

from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

def run_task(*_):
    env = normalize(DoublePendulumEnv())
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        plot=True
    )
    algo.train()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=5,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    plot=True
)
