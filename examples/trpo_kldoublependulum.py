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

N=200
delta = 0.23
seed=1738
keyword = 'trpo'

def run_task(*_):

    
    
    
    env = normalize(KLDoublePendulumEnv())
    
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
        max_path_length=100,
        batch_size=N*100,
        step_size=delta,
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