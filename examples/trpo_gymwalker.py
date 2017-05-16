from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.kl_double_pendulum_env import KLDoublePendulumEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.KL_gaussian_mlp_policy import KL_GaussianMLPPolicy
import lasagne.nonlinearities as NL
import lasagne.init as LI
import rllab.misc.logger as logger
from time import gmtime, strftime
from rllab.envs.gym_env import GymEnv

N=100
delta = 0.01
seed=1738
keyword = 'trpoWalker'

def run_task(*_):

    
    
    
    env = normalize(GymEnv("BipedalWalker-v2", record_video=False, record_log=False))
    
    policy = KL_GaussianMLPPolicy(
        env_spec=env.spec,
        learn_std = False,
        init_std=1., #set the std of the NN equal to the uncontrolled std
        hidden_sizes=(32,32),
        hidden_nonlinearity=NL.tanh, #NL.rectify
        hidden_W_init_mean=LI.GlorotUniform(),
        hidden_W_init_std=LI.GlorotUniform() #LI.Orthogonal('relu')
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        plot=True,
        step_size=delta,
        max_path_length=env.horizon,
        batch_size=N*env.horizon
    )
    logger.log("    seed " + str(seed))
    logger.log("    N " + str(N))
    logger.log("    delta " + str(delta))
    
    algo.train()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=5,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=seed,
    plot=True,
    exp_prefix=keyword,
    exp_name='trpo'+strftime("%Y-%m-%d %H:%M:%S", gmtime())+'seed '+str(seed)+'delta '+str(delta)+'N '+str(N)
)